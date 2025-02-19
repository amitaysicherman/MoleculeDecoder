import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel, PretrainedConfig, AutoModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import T5Config
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _shift_right(input_ids, decoder_start_token_id, pad_token_id):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VectorT5(PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)

        # Initialize MolFormer
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.output_dim)

        # Create encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # Create decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers
        )

        self.decoder_start_embedding = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Add positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)

    def _get_mol_embeddings(self, input_ids, token_attention_mask):
        """
        Process multiple molecules in parallel through MolFormer
        """
        batch_size, max_seq_len, seq_len = input_ids.shape

        # Reshape for parallel processing
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = token_attention_mask.view(-1, seq_len)

        # Process through MolFormer in chunks to avoid OOM
        chunk_size = 2048  # Adjust based on your GPU memory
        all_embeddings = []

        for i in range(0, flat_input_ids.size(0), chunk_size):
            chunk_input_ids = flat_input_ids[i:i + chunk_size]
            chunk_attention_mask = flat_attention_mask[i:i + chunk_size]

            with torch.no_grad():
                outputs = self.molformer(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask
                )
                all_embeddings.append(outputs.pooler_output)

        # Combine chunks
        embeddings = torch.cat(all_embeddings, dim=0)
        embeddings = embeddings.view(batch_size, max_seq_len, -1)
        return embeddings

    def _prepare_decoder_inputs(self, tgt_embeddings):
        """Prepare decoder inputs by shifting target embeddings right"""
        decoder_inputs = torch.cat(
            [self.decoder_start_embedding.expand(tgt_embeddings.size(0), -1, -1),
             tgt_embeddings[:, :-1]], dim=1
        )
        return decoder_inputs

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
            self,
            src_embeddings,
            src_mol_attention_mask,
            tgt_embeddings,
            tgt_mol_attention_mask,
            v2m,
            output_tokens,
            return_seq=False
    ):
        # Project embeddings and add positional encoding
        encoder_hidden_states = self.input_projection(src_embeddings)
        encoder_hidden_states = self.pos_encoder(encoder_hidden_states)

        # Create source padding mask (True means padding position)
        src_key_padding_mask = ~src_mol_attention_mask.bool()

        # Encoder
        memory = self.encoder(
            src=encoder_hidden_states,
            src_key_padding_mask=src_key_padding_mask
        )

        # Prepare decoder inputs
        decoder_inputs = self.input_projection(tgt_embeddings)
        decoder_inputs = self._prepare_decoder_inputs(decoder_inputs)
        decoder_inputs = self.pos_encoder(decoder_inputs)

        # Create masks for decoder
        tgt_key_padding_mask = ~tgt_mol_attention_mask.bool()
        tgt_mask = self.generate_square_subsequent_mask(decoder_inputs.size(1)).to(decoder_inputs.device)

        # Decoder
        decoder_outputs = self.decoder(
            tgt=decoder_inputs,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        sequence_output = self.output_projection(decoder_outputs)

        sequence_output = sequence_output[:, :1, :]

        decoder_input_ids = _shift_right(output_tokens, v2m.config.decoder_start_token_id, v2m.config.pad_token_id)
        decoder_input_ids = decoder_input_ids[:, 0, :]
        decoder_output = v2m.decoder(encoder_hidden_states=sequence_output, input_ids=decoder_input_ids)
        lm_logits = v2m.lm_head(decoder_output.last_hidden_state)
        labels = output_tokens[:, 0, :].clone()
        labels[labels == v2m.config.pad_token_id] = -100

        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), ignore_index=-100)

        # # Calculate loss
        # loss_fct = nn.MSELoss(reduction='none')
        # position_losses = loss_fct(sequence_output, tgt_embeddings)
        #
        #
        #
        #
        #
        #
        #
        #
        # mask = tgt_mol_attention_mask.unsqueeze(-1).expand_as(position_losses)
        # position_losses = position_losses * mask
        # total_active_elements = mask.sum()
        #
        # loss= position_losses.sum() / total_active_elements
        if return_seq:
            return loss, lm_logits.argmax(-1)
        return loss


if __name__ == "__main__":
    import torch
    from torch.optim import AdamW

    # MolFormer output dimension is 1024
    config = T5Config(
        d_model=8,  # Transformer hidden dimension
        d_ff=16,  # Feed-forward dimension
        num_layers=1,  # Number of encoder/decoder layers
        num_heads=1,  # Number of attention heads
        dropout=0.1,  # Dropout rate
        input_dim=10,  # MolFormer output dimension
        output_dim=10  # To match input dimension for loss calculation
    )

    # Initialize model
    model = VectorT5(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Create dummy data
    batch_size = 2
    seq_length = 3
    embedding_dim = 10  # MolFormer output dimension

    # Create random embeddings to simulate MolFormer output
    src_embeddings = torch.randn(batch_size, seq_length, embedding_dim).to(device)
    tgt_embeddings = torch.randn(batch_size, seq_length, embedding_dim).to(device)

    # Create attention masks (1 for valid tokens, 0 for padding)
    src_mol_attention_mask = torch.ones(batch_size, seq_length).to(device)
    tgt_mol_attention_mask = torch.ones(batch_size, seq_length).to(device)

    # Add some padding to test mask functionality
    src_mol_attention_mask[:, -1] = 0  # Last position is padding
    tgt_mol_attention_mask[:, -1] = 0  # Last position is padding

    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()

        loss = model(
            src_embeddings=src_embeddings,
            src_mol_attention_mask=src_mol_attention_mask,
            tgt_embeddings=tgt_embeddings,
            tgt_mol_attention_mask=tgt_mol_attention_mask,
        )

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
