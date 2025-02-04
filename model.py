import torch
import torch.nn as nn
from transformers import T5PreTrainedModel, T5Config, AutoModel
from transformers.models.t5.modeling_t5 import T5Stack

class VectorT5(T5PreTrainedModel):
    def __init__(self, config: T5Config, input_dim: int, output_dim: int):
        super().__init__(config)
        
        # Initialize MolFormer
        self.molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True
        )
        for param in self.molformer.parameters():
            param.requires_grad = False
        
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
    
    def _get_mol_embeddings(self, input_ids, token_attention_mask, mol_attention_mask):
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
        
        return embeddings, mol_attention_mask

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
        decoder_inputs = torch.zeros_like(tgt_embeddings)
        decoder_inputs[:, 1:] = tgt_embeddings[:, :-1].clone()
        # First token is zero (will be replaced with EOS token in model)
        return decoder_inputs
    def forward(
        self,
        src_input_ids,
        src_token_attention_mask,
        src_mol_attention_mask,
        tgt_input_ids=None,
        tgt_token_attention_mask=None,
        tgt_mol_attention_mask=None,
        labels=None,
        return_dict=None,
    ):
        # Get embeddings for source and target molecules
        src_embeddings, src_attention_mask = self._get_mol_embeddings(
            src_input_ids, 
            src_token_attention_mask, 
            src_mol_attention_mask
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
            attention_mask=src_attention_mask,
            return_dict=return_dict,
            use_cache=False,
        )
        
        if tgt_input_ids is None:
            # During inference, start with EOS token
            decoder_inputs = self.eos_embedding.repeat(src_input_ids.shape[0], 1, 1)
            decoder_attention_mask = torch.ones(src_input_ids.shape[0], 1, device=src_input_ids.device)
        else:
            # During training, use target sequence
            tgt_embeddings, tgt_attention_mask = self._get_mol_embeddings(
                tgt_input_ids,
                tgt_token_attention_mask,
                tgt_mol_attention_mask
            )
            decoder_inputs = self.input_projection(tgt_embeddings)

        decoder_inputs = self._prepare_decoder_inputs(decoder_inputs)

        # Decoder
        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs,
            attention_mask=tgt_attention_mask if tgt_input_ids is not None else decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=src_attention_mask,
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