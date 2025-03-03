from transformers import BertGenerationDecoder, BertGenerationConfig, BertGenerationEncoder
from torch import nn
from vector_quantize_pytorch import ResidualVQ


class Encoder(BertGenerationEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.pooling = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask, encoder_hidden_states=None, **kwargs):
        if encoder_hidden_states is not None:
            outputs = super().forward(input_ids, attention_mask=attention_mask,
                                      encoder_hidden_states=encoder_hidden_states)
        else:
            outputs = super().forward(input_ids, attention_mask=attention_mask)
        pooled_output = self.pooling(outputs.last_hidden_state[:, 0]).unsqueeze(1)
        return pooled_output


class Decoder(BertGenerationDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, input_ids, attention_mask, encoder_hidden_states, labels, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask,
                                  encoder_hidden_states=encoder_hidden_states, labels=labels)
        return outputs


class AutoEncoder(nn.Module):
    def __init__(self, config_encoder, config_decoder):
        super().__init__()
        self.encoder = Encoder(config_encoder)
        self.decoder = Decoder(config_decoder)

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        encoder_outputs = self.encoder(input_ids, attention_mask)
        decoder_outputs = self.decoder(input_ids, attention_mask, encoder_hidden_states=encoder_outputs, labels=labels)
        return decoder_outputs


class VAE(nn.Module):
    def __init__(self, config_encoder, config_decoder, kld_weight=0.1):
        super().__init__()
        self.encoder = Encoder(config_encoder)
        self.decoder = Decoder(config_decoder)
        self.kld_weight = kld_weight
        self.latent_dim = config_decoder.hidden_size
        self.mean = nn.Linear(config_encoder.hidden_size, self.latent_dim)
        self.log_var = nn.Linear(config_encoder.hidden_size, self.latent_dim)

    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick: sample from the distribution
        having mean and log_var using the parameter-free sampling.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        encoder_output = self.encoder(input_ids, attention_mask)

        hidden = encoder_output.squeeze(1)  # (batch_size, hidden_size)
        mean = self.mean(hidden)
        log_var = self.log_var(hidden)
        z = self.reparameterize(mean, log_var).unsqueeze(1)  # (batch_size, 1, hidden_size)
        decoder_outputs = self.decoder(input_ids, attention_mask, encoder_hidden_states=z, labels=labels)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        total_loss = decoder_outputs.loss + self.kld_weight * kl_loss
        decoder_outputs.loss = total_loss
        return decoder_outputs


class VQ(nn.Module):
    def __init__(self, config_encoder, config_decoder, vq_weight=0.1, codebook_size=512, num_quantizers=8):
        super().__init__()
        self.encoder = Encoder(config_encoder)
        self.decoder = Decoder(config_decoder)
        self.vq_weight = vq_weight
        self.vq = ResidualVQ(
            dim=config_encoder.hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,

        )

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        encoder_output = self.encoder(input_ids, attention_mask)
        hidden = encoder_output.squeeze(1)  # (batch_size, hidden_size)
        quantized, indices, commit_loss = self.vq(hidden)
        quantized = quantized.unsqueeze(1)  # (batch_size, 1, hidden_size)
        decoder_outputs = self.decoder(input_ids, attention_mask,
                                       encoder_hidden_states=quantized,
                                       labels=labels)

        total_loss = decoder_outputs.loss + (commit_loss.mean()) * self.vq_weight
        decoder_outputs.loss = total_loss

        return decoder_outputs


hidden_sizes = {'s': 128, 'm': 512, 'l': 1024}
num_layers = {'s': 2, 'm': 6, 'l': 12}
num_heads = {'s': 2, 'm': 4, 'l': 8}
vq_codebook_sizes = {'s': 128, 'm': 512, 'l': 1024}
vq_num_quantizers = {'s': 4, 'm': 8, 'l': 16}


def get_model(name, size, tokenizer):
    size_args = dict(
        hidden_size=hidden_sizes[size],
        num_hidden_layers=num_layers[size],
        num_attention_heads=num_heads[size],
        intermediate_size=hidden_sizes[size] * 2,
    )
    common_args = dict(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        is_encoder_decoder=True,
    )
    encoder_config = BertGenerationConfig(**common_args, **size_args, is_decoder=False)
    decoder_config = BertGenerationConfig(**common_args, **size_args, is_decoder=True, add_cross_attention=True)
    if name == 'ae':
        return AutoEncoder(encoder_config, decoder_config)
    elif name == 'vae':
        return VAE(encoder_config, decoder_config)
    elif name == 'vq':
        return VQ(encoder_config, decoder_config, codebook_size=vq_codebook_sizes[size],
                  num_quantizers=vq_num_quantizers[size])


if __name__ == "__main__":
    from autoencoder.data import AutoEncoderDataset
    from torch.utils.data import DataLoader
    import torch
    from torch.optim import AdamW

    dataset = AutoEncoderDataset()
    dataset.data = dataset.data[:2]
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    tokenizer = dataset.tokenizer
    model = get_model('ae', 's', tokenizer)
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = AdamW(model.parameters(), lr=1e-4)

    for _ in range(1000):
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels)
            outputs.loss.backward()
            optimizer.step()
            print(outputs.loss.item())
