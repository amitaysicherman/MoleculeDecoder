import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoModel, T5Config, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import functional as F
from vector_quantize_pytorch import ResidualVQ


def _shift_right(input_ids, decoder_start_token_id, pad_token_id):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class SMILESDataset(Dataset):
    def __init__(self, tokenizer, smiles_file="pubchem-canonical/CID-SMILES-CANONICAL.smi"):
        self.smiles = []
        with open(smiles_file) as f:
            for line in f:
                lines_smiles = line.strip().split()[1]
                self.smiles.append(lines_smiles)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        tokens = self.tokenizer(smile, padding="max_length", truncation=True, max_length=75, return_tensors="pt")
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        labels = tokens["input_ids"].clone()
        # replace pad tokens with -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        tokens["labels"] = labels
        return tokens


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation
    """
    predictions, labels = eval_pred
    # Get argmax of predictions
    predictions = np.argmax(predictions, axis=-1)
    mask = labels != -100

    total_tokens = mask.sum()
    correct_tokens = ((predictions == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens

    return {
        "token_accuracy": token_accuracy,
    }


class MolFormerT5Decoder(PreTrainedModel):
    def __init__(self, config, q_cp=""):
        super().__init__(config)
        # Load and freeze MolFormer
        self.molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True,
            deterministic_eval=True
        )
        self.config = config
        for param in self.molformer.parameters():
            param.requires_grad = False
        # Simple projection if needed
        if self.molformer.config.hidden_size != config.d_model:
            self.proj = nn.Linear(self.molformer.config.hidden_size, config.d_model)
        else:
            self.proj = nn.Identity()
        # Initialize T5 decoder
        T5 = T5ForConditionalGeneration(config)
        self.decoder = T5.get_decoder()
        self.lm_head = T5.lm_head
        if q_cp:
            _, _, _, num_quantizers, codebook_size, input_dim, _, _, _, _ = q_cp.split("/")[-1].split("_")
            self.q_model = ResidualVQ(
                dim=int(input_dim),
                num_quantizers=int(num_quantizers),
                codebook_size=int(codebook_size),
                ema_update=False,  # Use gradient descent instead of EMA
            )
            self.q_model.load_state_dict(torch.load(q_cp, map_location=torch.device('cpu')))
            for param in self.q_model.parameters():
                param.requires_grad = False
            self.q_model.eval()
        else:
            self.q_model = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get MolFormer embedding
        mol_outputs = self.molformer(input_ids, attention_mask=attention_mask)
        encoder_outputs = self.proj(mol_outputs.pooler_output)
        if self.q_model:
            encoder_outputs, *_ = self.q_model(encoder_outputs)
        encoder_outputs = encoder_outputs.unsqueeze(1)
        decoder_input_ids = _shift_right(input_ids, self.config.decoder_start_token_id, self.config.pad_token_id)
        decoder_output = self.decoder(encoder_hidden_states=encoder_outputs, input_ids=decoder_input_ids)
        lm_logits = self.lm_head(decoder_output.last_hidden_state)

        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), ignore_index=-100)
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
        )


def create_model(q_cp=""):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    # Initialize config
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=768,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=6,
        is_encoder_decoder=True,
        is_decoder=True,
        num_heads=8,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    model = MolFormerT5Decoder(config, q_cp)
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # number of non trainable parameters
    print(f"Number of non-trainable parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(model)

    if q_cp:
        print(f"Loading model to cuntinue training")
        model.load_state_dict(
            torch.load("results_pubchem/checkpoint-90000/pytorch_model.bin", map_location=torch.device('cpu')))
    return model, tokenizer


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--q_cp", type=str, default="results_pubchem/residual_vq_lr_64_512_768_10000_0.001_100_best.pt")
    args = parser.parse_args()

    model, tokenizer = create_model(args.q_cp)

    dataset = SMILESDataset(tokenizer)
    train_size = len(dataset) - 100_000
    eval_size = 100_000
    train_dataset, eval_dataset = random_split(
        dataset, [train_size, eval_size]
    )
    suf = "_pubchem"
    if args.q_cp:
        suf += "_quantized"
    training_args = TrainingArguments(
        output_dir=f"./results{suf}",
        num_train_epochs=10 if not args.q_cp else 5,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        learning_rate=1e-4,  # Constant learning rate
        logging_dir=f'./logs{suf}',
        logging_steps=1_000 if not args.q_cp else 500,
        save_steps=5_000 if not args.q_cp else 500,
        eval_accumulation_steps=2,
        eval_steps=5_000 if not args.q_cp else 500,
        evaluation_strategy="steps",
        report_to=["tensorboard"],
        lr_scheduler_type="constant",  # Use constant learning rate
        load_best_model_at_end=True,
        metric_for_best_model="token_accuracy",
        save_safetensors=False,
        label_names=["labels"],

    )

    # Initialize trainer with evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained("path/to/save/model")
