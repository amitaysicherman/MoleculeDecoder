from transformers import PreTrainedModel, T5PreTrainedModel, T5Config, GenerationConfig, GenerationMixin
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Tuple, Union
import torch


class SimpleT5Decoder(T5PreTrainedModel, GenerationMixin):
    def __init__(self, d_model: T5PreTrainedModel, pad_token_id, eos_token_id, decoder_start_token_id, bos_token_id):
        super().__init__(d_model.config)
        d_model.config.is_encoder_decoder = False
        d_model.config.is_decoder = True
        self.decoder = d_model.decoder
        self.lm_head = d_model.lm_head
        self.generation_config = GenerationConfig(
            max_length=75,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            bos_token_id=bos_token_id,

        )

        # Save config
        self.config = d_model.config

    def forward(
            self,
            encoder_hidden_states: torch.Tensor,
            decoder_input_ids: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:

        # Ensure encoder_hidden_states has correct shape [batch, seq_len, hidden_size]
        if len(encoder_hidden_states.shape) == 2:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

        # Get decoder outputs
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Get language model logits
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past_key_values=None,
            attention_mask=None,
            encoder_hidden_states=None,
            **kwargs
    ):
        """
        Implement this method to make the model compatible with huggingface's generate() method
        """
        # if encoder_hidden_states is None, we need to get it from kwargs
        if encoder_hidden_states is None:
            encoder_hidden_states = kwargs.get("encoder_outputs", None)
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states = encoder_hidden_states[0]

        return {
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "encoder_hidden_states": encoder_hidden_states,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }


#

if __name__ == "__main__":
    from train_decoder import create_model, _shift_right

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model, tokenizer = create_model()
    d_model.load_state_dict(torch.load("results_pubchem/checkpoint-90000/pytorch_model.bin", map_location="cpu"),
                            strict=False)
    d_model = d_model.to(device).eval()

    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    ]
    tokens = tokenizer(test_smiles, padding="max_length", truncation=True, max_length=75, return_tensors="pt")
    print(tokens)

    mol_outputs = d_model.molformer(**tokens)
    encoder_hidden_states = d_model.proj(mol_outputs.pooler_output).unsqueeze(1)
    model = SimpleT5Decoder(d_model, 2, 1,
                            2, 2).eval()
    #
    # Generate
    # print(tokens["input_ids"][:, 0:1])
    # input_tokens = torch.LongTensor([[0,  4,  4,]]).to(device)
    generated_ids = model.generate(
        encoder_hidden_states=encoder_hidden_states,
        max_length=25,
        num_beams=25,
        early_stopping=True,
        top_k=2,
        do_sample=True,
        num_return_sequences=10
    )
    print(generated_ids)
    generated_smiles = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_smiles)
    print(test_smiles[0] in generated_smiles)
    tokens["labels"] = tokens["input_ids"].clone()
    model_output = d_model(**tokens)
    preds = model_output.logits.argmax(-1)
    print(preds)
    preds_smiles = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print(preds_smiles)
