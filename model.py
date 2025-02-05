import torch
import torch.nn as nn
from transformers import T5PreTrainedModel, T5Config, AutoModel, T5ForConditionalGeneration
from train_decoder import create_model, _shift_right
import copy
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorT5(T5PreTrainedModel):
    def __init__(self, config: T5Config, input_dim: int, output_dim: int):
        super().__init__(config)

        # Initialize MolFormer
        self.molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True
        ).to(device).eval()
        self.tokens_decoder = create_model()[0]
        self.tokens_decoder.load_state_dict(
            torch.load('results/checkpoint-100000/pytorch_model.bin', map_location='cpu'), strict=False)
        self.tokens_decoder = self.tokens_decoder.to(device).eval()
        for param in self.molformer.parameters():
            param.requires_grad = False
        # Input/Output projections

        self.input_projection = nn.Linear(input_dim, config.d_model)
        self.output_projection = nn.Linear(config.d_model, output_dim)

        # T5 encoder and decoder
        T5 = T5ForConditionalGeneration(config)
        self.encoder = T5.get_encoder()
        self.decoder = T5.get_decoder()
        del T5
        del self.encoder.embed_tokens
        del self.decoder.embed_tokens

        self.eos_embedding = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.decoder_start_embedding = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Initialize weights
        self.init_weights()

    def _get_mol_embeddings(self, input_ids, token_attention_mask):
        """
        Process multiple molecules in parallel through MolFormer
        """
        batch_size, max_seq_len, seq_len = input_ids.shape

        # Reshape for parallel processing
        flat_input_ids = input_ids.view(-1, seq_len)  # (batch_size * max_seq_len, 75)
        flat_attention_mask = token_attention_mask.view(-1, seq_len)  # (batch_size * max_seq_len, 75)

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
        embeddings = torch.cat(all_embeddings, dim=0)  # (batch_size * max_seq_len, hidden_size)

        # Reshape back to original dimensions
        embeddings = embeddings.view(batch_size, max_seq_len, -1)  # (batch_size, max_seq_len, hidden_size)
        return embeddings

    # def _append_eos(self, x, attention_mask=None):
    #     if attention_mask is None:
    #         return x, attention_mask
    #
    #     # Find first padding position in each sequence
    #     first_pad_pos = attention_mask.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1)
    #
    #     # Replace first padding token with EOS embedding
    #     eos_pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # Shape: (1, seq_len)
    #     is_eos = (eos_pos == first_pad_pos)  # Shape: (batch_size, seq_len)
    #
    #     # Create EOS embeddings for each sequence
    #     eos = self.eos_embedding.expand(x.shape[0], -1, -1)  # Shape: (batch_size, 1, hidden_size)
    #     x = torch.where(is_eos.unsqueeze(-1), eos, x)
    #
    #     return x, attention_mask
    #
    def _prepare_decoder_inputs(self, tgt_embeddings):
        """Prepare decoder inputs by shifting target embeddings right"""
        # Create decoder input by shifting right
        decoder_inputs = torch.cat(
            [self.decoder_start_embedding.expand(tgt_embeddings.size(0), -1, -1), tgt_embeddings[:, :-1]], dim=1
        )

        return decoder_inputs

    def forward(
            self,
            src_input_ids,
            src_token_attention_mask,
            src_mol_attention_mask,
            tgt_input_ids=None,
            tgt_token_attention_mask=None,
            tgt_mol_attention_mask=None,
            return_dict=None,
    ):
        # Get embeddings for source and target molecules
        src_embeddings = self._get_mol_embeddings(
            src_input_ids,
            src_token_attention_mask,
        )
        # Project embeddings to model dimension
        encoder_hidden_states = self.input_projection(src_embeddings)
        # # Add EOS to encoder inputs
        # encoder_hidden_states, src_attention_mask = self._append_eos(
        #     encoder_hidden_states,
        #     src_attention_mask
        # )

        # Encoder
        encoder_outputs = self.encoder(
            inputs_embeds=encoder_hidden_states,
            attention_mask=src_mol_attention_mask,
            return_dict=return_dict,
            use_cache=False,
        )
        tgt_embeddings = self._get_mol_embeddings(
            tgt_input_ids,
            tgt_token_attention_mask,
        )
        decoder_inputs = self.input_projection(tgt_embeddings)

        decoder_inputs = self._prepare_decoder_inputs(decoder_inputs)

        # Decoder
        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=tgt_mol_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=src_mol_attention_mask,
            return_dict=return_dict,
            use_cache=False,
        )

        sequence_output = self.output_projection(decoder_outputs[0])

        sequence_output_single_molecule = sequence_output[:, 0:1, :]  # current in forwaeed only one molecule
        tgt_input_ids_single_molecule = tgt_input_ids[:, 0, :]
        tokens_decoder_input_ids = _shift_right(tgt_input_ids_single_molecule, self.tokens_decoder.config.decoder_start_token_id,
                                                self.tokens_decoder.config.pad_token_id)
        tokens_decoder_output = self.tokens_decoder.decoder(encoder_hidden_states=sequence_output_single_molecule,
                                                            input_ids=tokens_decoder_input_ids)
        tokens_lm_logits = self.tokens_decoder.lm_head(tokens_decoder_output.last_hidden_state)
        # loss = F.cross_entropy(tokens_lm_logits.view(-1, tokens_lm_logits.size(-1)), tgt_input_ids_single_molecule.view(-1),
        #                        ignore_index=-self.tokens_decoder.config.pad_token_id)
        loss = F.cross_entropy(
            tokens_lm_logits.reshape(-1, tokens_lm_logits.size(-1)),
            tgt_input_ids_single_molecule.reshape(-1),
            ignore_index=self.tokens_decoder.config.pad_token_id,
        )
        # Get target embeddings for labels
        # labels = self._get_mol_embeddings(
        #     tgt_input_ids,
        #     tgt_token_attention_mask,
        # )
        #
        # loss_fct = nn.MSELoss(reduction='none')
        # # Calculate MSE loss for each position
        # position_losses = loss_fct(sequence_output, labels)
        #
        # # Apply mask to zero out loss for padding positions
        # if tgt_mol_attention_mask is not None:
        #     # Expand mask to match loss dimensions if needed
        #     mask = tgt_mol_attention_mask.unsqueeze(-1).expand_as(position_losses)
        #     position_losses = position_losses * mask
        #
        #     # Calculate mean loss over non-padding positions
        #     total_active_elements = mask.sum()
        #     if total_active_elements > 0:
        #         loss = position_losses.sum() / total_active_elements
        #     else:
        #         loss = position_losses.sum() * 0.0  # Return 0 loss if all positions are masked
        # else:
        #     # If no mask provided, take mean over all positions
        #     loss = position_losses.mean()

        if return_dict:
            return {
                'loss': loss,
                'logits': sequence_output,
                'encoder_last_hidden_state': encoder_outputs[0],
                'decoder_last_hidden_state': decoder_outputs[0],
            }

        return loss, sequence_output
