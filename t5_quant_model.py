from transformers import T5PreTrainedModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
import torch
import torch.nn as nn
from typing import Optional, Tuple
import copy
from transformers import PreTrainedModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_quantization_embeddings(
        embeddings_path: str = "results_pubchem/residual_vq_lr_64_512_768_10000_0.001_100_best.pt"):
    embeddings = torch.load(embeddings_path, map_location='cpu')
    embeddings = [embeddings[f'layers.{i}._codebook.embed'][0] for i in range(64)]
    # add zerp embedding for padding token
    for i in range(len(embeddings)):
        embeddings[i] = torch.cat([embeddings[i], torch.zeros(1, 768)], dim=0)

    return embeddings


def _shift_right(emb, decoder_start_embedding):
    decoder_inputs = torch.cat(
        [decoder_start_embedding.expand(emb.size(0), -1, -1),
         emb[:, :-1]], dim=1
    )
    return decoder_inputs


class EmbeddingSum(nn.Module):
    def __init__(self, k, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.k = k
        embedding = load_quantization_embeddings()

        self.embeddings = nn.ModuleList([
            nn.Embedding.from_pretrained(embedding[i])
            for i in range(k)
        ])
        for embedding_layer in self.embeddings:
            embedding_layer.weight.requires_grad = False

        self.embedding_sum = torch.Tensor([x.abs().sum() for x in embedding]).to(device)
        self.embedding_sum = self.embedding_sum / self.embedding_sum.sum()

    def forward(self, input_ids):
        embedded = None
        for i, embedding_layer in enumerate(self.embeddings):
            indexes_in_range = range(i, input_ids.shape[1], self.k)
            res = embedding_layer(input_ids[..., indexes_in_range])
            if embedded is None:
                embedded = res
            else:
                embedded += res
        return embedded


class T5ForResidualQuantization(PreTrainedModel):
    def __init__(self, config: T5Config, num_quantization: int):

        config.dropout = 0.1
        super().__init__(config)
        self.num_quantization = num_quantization

        # Modified shared embedding
        self.shared = EmbeddingSum(
            k=num_quantization,
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model
        )

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

        self.lm_head = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model, bias=False)
            for _ in range(num_quantization)
        ])
        self.decoder_start_embedding = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


    def get_mask_slice(self, mask):
        step=self.num_quantization
        return mask[:, ::step]

    def forward(self, input_ids, input_mask, labels, label_mask):
        inputs_embeds = self.shared(input_ids)
        inputs_embeds = self.pos_encoder(inputs_embeds)
        input_mask = self.get_mask_slice(input_mask)
        src_key_padding_mask = ~input_mask.bool()

        # Encoder
        memory = self.encoder(
            src=inputs_embeds,
            src_key_padding_mask=src_key_padding_mask
        )

        output_embeds = self.shared(labels)
        output_embeds = _shift_right(output_embeds, self.decoder_start_embedding)
        output_embeds = self.pos_encoder(output_embeds)
        label_mask = self.get_mask_slice(label_mask)
        tgt_key_padding_mask = ~label_mask.bool()
        tgt_mask = self.generate_square_subsequent_mask(output_embeds.size(1)).to(output_embeds.device)
        decoder_outputs = self.decoder(
            tgt=output_embeds,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        sequence_output = decoder_outputs  # [0]

        # Apply multiple LM heads
        logits = []
        for emb_layer, lm_head_layer in zip(self.shared.embeddings, self.lm_head):
            y = lm_head_layer(sequence_output)
            # Calculate the dot product between the sequence output and the embedding layer
            dot_product = torch.matmul(sequence_output, emb_layer.weight.t())
            logits.append(dot_product)

        logits = torch.stack(logits, dim=2)  # (batch, seq_len, num_quantization, vocab_size)
        batch_size = logits.shape[0]
        seq_length = logits.shape[1]
        logits = logits.view(batch_size, seq_length * self.num_quantization, -1)
        # print(logits.shape)
        # print(logits.argmax(dim=-1))
        # print(labels)
        w = self.shared.embedding_sum  # shape (num_quantization)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=512, reduction='none')
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            loss[labels.view(-1) == 512] = 0
            loss = loss.view(batch_size, seq_length, self.num_quantization)
            loss = (loss * w).sum(dim=-1).mean()

        return (loss, logits) if loss is not None else logits
