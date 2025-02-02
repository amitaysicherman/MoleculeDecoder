import torch
import torch.nn as nn
from transformers import T5PreTrainedModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from train_mlm import ReactionMolsDataset
num_epochs = 10

class VectorT5(T5PreTrainedModel):
    def __init__(self, config: T5Config, input_dim: int, output_dim: int):
        super().__init__(config)

        # Modify config for vector inputs
        config.vocab_size = 1  # Not used but required
        config.d_model = 512  # Can be adjusted

        # Input/Output projections
        self.input_projection = nn.Linear(input_dim, config.d_model)
        self.output_projection = nn.Linear(config.d_model, output_dim)

        # T5 encoder and decoder
        self.encoder = T5Stack(config)
        self.decoder = T5Stack(config)
        self.eos_embedding = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Initialize weights
        self.init_weights()

    def _append_eos(self, x, attention_mask=None):
        if attention_mask is None:
            return x, attention_mask

        # Find first padding position in each sequence
        first_pad_pos = attention_mask.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1)

        # Replace first padding token with EOS embedding
        eos_pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # Shape: (1, seq_len)
        is_eos = (eos_pos == first_pad_pos)  # Shape: (batch_size, seq_len)

        # Create EOS embeddings for each sequence
        eos = self.eos_embedding.expand(x.shape[0], -1, -1)  # Shape: (batch_size, 1, hidden_size)
        x = torch.where(is_eos.unsqueeze(-1), eos, x)

        return x, attention_mask

    def forward(
            self,
            input_vectors,
            decoder_input_vectors=None,
            attention_mask=None,
            decoder_attention_mask=None,
            labels=None,
            return_dict=None,
    ):
        # Project input vectors
        encoder_hidden_states = self.input_projection(input_vectors)

        # Add EOS to encoder inputs
        encoder_hidden_states, attention_mask = self._append_eos(
            encoder_hidden_states, attention_mask
        )

        # Encoder
        encoder_outputs = self.encoder(
            inputs_embeds=encoder_hidden_states,
            attention_mask=attention_mask,
            return_dict=return_dict,
            use_cache=False,
        )

        if decoder_input_vectors is None:
            # During inference, start with EOS token
            decoder_inputs = self.eos_embedding.repeat(input_vectors.shape[0], 1, 1)
            decoder_attention_mask = torch.ones(input_vectors.shape[0], 1, device=input_vectors.device)
        else:
            # During training, use shifted target sequence
            decoder_inputs = self.input_projection(decoder_input_vectors)

        # Decoder
        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
            use_cache=False,
        )

        sequence_output = self.output_projection(decoder_outputs[0])

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(sequence_output, labels)

        if return_dict:
            return {
                'loss': loss,
                'logits': sequence_output,
                'encoder_last_hidden_state': encoder_outputs[0],
                'decoder_last_hidden_state': decoder_outputs[0],
            }

        return (loss, sequence_output) if loss is not None else sequence_output


# Training example
def train_vector_t5():
    # Model configuration
    config = T5Config(
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
    )

    # Initialize model and dataset
    model = VectorT5(config, input_dim=768, output_dim=768)  # MolFormer hidden size
    dataset = ReactionMolsDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            outputs = model(
                input_vectors=batch['input_vectors'],
                decoder_input_vectors=batch['decoder_input_vectors'],
                attention_mask=batch['attention_mask'],
                decoder_attention_mask=batch['decoder_attention_mask'],
                labels=batch['labels'],
                return_dict=True,
            )

            loss = outputs['loss']
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Usage
if __name__ == "__main__":
    train_vector_t5()