from transformers import T5PreTrainedModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
import torch
import torch.nn as nn
from typing import Optional, Tuple
import copy



def _shift_right(emb, pad_token):
    return torch.cat(
        [pad_token.expand(emb.size(0), -1, -1),
         emb[:, :-1]], dim=1
    )


class EmbeddingSum(nn.Module):
    def __init__(self, k, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.k = k

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim)
            for _ in range(k)
        ])

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


class T5ForResidualQuantization(T5PreTrainedModel):
    def __init__(self, config: T5Config, num_quantization: int):
        super().__init__(config)
        self.num_quantization = num_quantization

        # Modified shared embedding
        self.shared = EmbeddingSum(
            k=num_quantization,
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model
        )

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.ModuleList([
            nn.Linear(config.d_model, config.vocab_size, bias=False)
            for _ in range(num_quantization)
        ])

        # Initialize weights
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle inputs embedding
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.shared(input_ids)

        # Handle decoder inputs embedding
        if decoder_inputs_embeds is None and labels is not None:
            decoder_inputs_embeds = self.shared(labels)
        decoder_inputs_embeds = _shift_right(decoder_inputs_embeds, self.vocab_size)

        # Encode if needed
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=None,  # We already embedded
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=None,  # We already embedded
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Apply multiple LM heads
        logits = []
        for lm_head_layer in self.lm_head:
            logits.append(lm_head_layer(sequence_output))

        # Stack and reshape logits
        logits = torch.stack(logits, dim=2)  # (batch, seq_len, num_quantization, vocab_size)
        batch_size = logits.shape[0]
        seq_length = logits.shape[1]
        logits = logits.view(batch_size, seq_length * self.num_quantization, -1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return (loss, logits) if loss is not None else logits


# Example usage:
def create_model(num_quantization: int):
    config = T5Config.from_pretrained('t5-base')
    model = T5ForResidualQuantization(config, num_quantization)
    return model
