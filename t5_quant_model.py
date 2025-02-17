from transformers import T5PreTrainedModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
import torch
import torch.nn as nn
from typing import Optional, Tuple


class T5ForResidualQuantization(T5PreTrainedModel):
    def __init__(self, config: T5Config, num_quantization: int):
        super().__init__(config)
        self.num_quantization = num_quantization

        # Modified shared embedding
        self.shared = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.d_model)
            for _ in range(num_quantization)
        ])

        self.encoder = T5Stack(config, self.shared[0])  # Use first embedding for encoder

        decoder_config = config.copy()
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        self.decoder = T5Stack(decoder_config, self.shared[0])  # Use first embedding for decoder

        # Modified LM head
        self.lm_head = nn.ModuleList([
            nn.Linear(config.d_model, config.vocab_size, bias=False)
            for _ in range(num_quantization)
        ])

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.shared[0]  # Return first embedding for compatibility

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings

    def embed_tokens(self, input_ids):
        """Custom embedding function for residual quantization"""
        # Reshape input_ids from (batch, seq_len*num_quantization) to (batch, seq_len, num_quantization)
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1] // self.num_quantization
        input_ids = input_ids.view(batch_size, seq_length, self.num_quantization)

        # Apply each embedding and sum
        embedded = torch.zeros(
            batch_size,
            seq_length,
            self.config.d_model,
            device=input_ids.device
        )

        for i, embedding_layer in enumerate(self.shared):
            embedded += embedding_layer(input_ids[..., i])

        return embedded

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
            inputs_embeds = self.embed_tokens(input_ids)

        # Handle decoder inputs embedding
        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = self.embed_tokens(decoder_input_ids)

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