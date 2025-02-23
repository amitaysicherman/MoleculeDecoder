import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoModel, T5Config, AutoTokenizer, PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack

from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import functional as F
from vector_quantize_pytorch import ResidualVQ


def _shift_right(input_ids, decoder_start_token_id, pad_token_id):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
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


def compute_metrics(eval_pred, debug=False):
    """
    Compute metrics for evaluation
    """
    predictions, labels = eval_pred

    # Get argmax of predictions
    predictions = np.argmax(predictions, axis=-1)
    if debug:
        print()
        print(predictions[0].tolist())
        print(labels[0].tolist())
    mask = labels != -100
    total_tokens = mask.sum()
    correct_tokens = ((predictions == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens
    if debug:
        print(f"Token accuracy: ({correct_tokens} / {total_tokens}) = {token_accuracy}")
    correct_or_pad = (predictions == labels) | (~mask)
    correct_samples = correct_or_pad.all(axis=-1).sum()
    total_samples = len(labels)
    sample_accuracy = correct_samples / total_samples

    return {
        "token_accuracy": token_accuracy,
        "sample_accuracy": sample_accuracy,
    }


class MolFormerT5Decoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Load and freeze MolFormer
        self.config = config
        self.embedder = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            deterministic_eval=True,
            trust_remote_code=True
        ).eval()
        for param in self.embedder.parameters():
            param.requires_grad = False

        if self.embedder.config.hidden_size != config.d_model:
            self.proj = nn.Linear(self.embedder.config.hidden_size, config.d_model)
        else:
            self.proj = nn.Identity()
        # Initialize T5 decoder
        # import T5 Stack
        embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        self.decoder = T5Stack(config, embed_tokens)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        mol_outputs = self.embedder(input_ids, attention_mask=attention_mask)
        encoder_outputs = self.proj(mol_outputs.pooler_output)
        encoder_outputs = encoder_outputs.unsqueeze(1)
        decoder_input_ids = _shift_right(input_ids, self.config.decoder_start_token_id, self.config.pad_token_id)
        decoder_output = self.decoder(encoder_hidden_states=encoder_outputs, input_ids=decoder_input_ids)
        lm_logits = self.lm_head(decoder_output.last_hidden_state)
        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), ignore_index=-100)
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
        )


def create_model(debug=False):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    # Initialize config

    if debug:
        config = T5Config(
            vocab_size=len(tokenizer),
            d_model=256,
            d_ff=512,
            num_layers=4,
            is_encoder_decoder=False,
            is_decoder=True,
            num_heads=4,
            decoder_start_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        config = T5Config(
            vocab_size=len(tokenizer),
            d_model=768,
            d_ff=2048,
            is_encoder_decoder=False,
            is_decoder=True,
            num_layers=6,
            num_decoder_layers=6,
            num_heads=8,
            decoder_start_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,

        )
    if debug:
        print(config)
    model = MolFormerT5Decoder(config)
    # print number of parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    print(
        f"Total parameters: {total_params:,},(trainable: {trainable_params:,}, non-trainable: {non_trainable_params:,})")
    return model, tokenizer


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    model, tokenizer = create_model(args.debug)

    dataset = SMILESDataset(tokenizer)
    DEBUG = args.debug
    if DEBUG:
        train_size = 1
        eval_size = 1
        dataset.smiles = dataset.smiles[:train_size + eval_size]
        train_dataset, eval_dataset = random_split(
            dataset, [train_size, eval_size]
        )
        eval_dataset = train_dataset
    else:
        train_size = len(dataset) - 100_000
        eval_size = 100_000
        train_dataset, eval_dataset = random_split(
            dataset, [train_size, eval_size]
        )

    suf = "_decoder"
    training_args = TrainingArguments(
        output_dir=f"./results{suf}",
        num_train_epochs=10 if not DEBUG else 10000,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        learning_rate=1e-4 if not DEBUG else 1e-5,
        logging_dir=f'./logs{suf}',
        logging_steps=1_000 if not DEBUG else 10,
        save_steps=5_000 if not DEBUG else 50000000,
        eval_accumulation_steps=2,
        eval_steps=5_000 if not DEBUG else 10,
        evaluation_strategy="steps",
        report_to=["tensorboard"] if not DEBUG else [],
        lr_scheduler_type="linear",
        warmup_steps=5_000 if not DEBUG else 500,
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
        compute_metrics=lambda x: compute_metrics(x, debug=DEBUG),
    )
    scores = trainer.evaluate()
    print(scores)

    trainer.train(resume_from_checkpoint=False)
