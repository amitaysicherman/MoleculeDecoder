import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    Trainer, TrainingArguments
)
import numpy as np
from autoencoder.data import get_tokenizer
from autoencoder.data import smiles_to_tokens


class TranslationDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        base_dir = "USPTO"

        # Read source and target files
        with open(f"{base_dir}/reactants-{split}.txt") as f:
            self.reactants = f.read().splitlines()
        with open(f"{base_dir}/products-{split}.txt") as f:
            self.products = f.read().splitlines()
        with open(f"{base_dir}/reagents-{split}.txt") as f:
            self.reagents = f.read().splitlines()

    def __len__(self):
        return len(self.products)

    def __getitem__(self, idx):
        reac = self.reactants[idx]
        reag = self.reagents[idx]
        source = f"{reac} .. {reag}"
        target = self.products[idx]

        source_tokens = " ".join(smiles_to_tokens(source))
        target_tokens = " ".join(smiles_to_tokens(target))
        source_encoding = self.tokenizer.encode(
            source_tokens,
            max_length=self.max_length,
        )
        target_encoding = self.tokenizer.encode(
            target_tokens,
            max_length=self.max_length,
        )
        label = target_encoding['input_ids'].clone()
        label[label == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids': source_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
            'labels': label
        }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[0]
    predictions = np.argmax(predictions, axis=-1)
    is_pad = labels == -100
    correct_or_pad = (predictions == labels) | is_pad
    perfect_match_accuracy = correct_or_pad.all(axis=1).mean()
    correct_not_pad = (predictions == labels) & ~is_pad
    token_accuracy = correct_not_pad.sum() / (~is_pad).sum()

    return {
        "perfect_match_accuracy": perfect_match_accuracy,
        "token_accuracy": token_accuracy
    }
import os
import glob
def get_last_cp(base_dir):
    if not os.path.exists(base_dir):
        return None
    all_checkpoints = glob.glob(f"{base_dir}/checkpoint-*")
    if not all_checkpoints:
        return None
    cp_steps = [int(cp.split("-")[-1]) for cp in all_checkpoints]
    last_cp = max(cp_steps)
    return f"{base_dir}/checkpoint-{last_cp}"


def main(retro=False):
    # Build vocabulary and create tokenizer
    tokenizer = get_tokenizer()
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=512,  # Hidden size
        d_ff=2048,  # Intermediate feed-forward size
        num_layers=6,  # Number of encoder/decoder layers
        num_heads=4,  # Number of attention heads
        is_encoder_decoder=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )

    # Initialize model from configuration
    model = T5ForConditionalGeneration(config)
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Prepare datasets
    src_prefix = "src" if not retro else "tgt"
    tgt_prefix = "tgt" if not retro else "src"
    train_dataset = TranslationDataset(
        "train",
        tokenizer
    )
    eval_dataset = TranslationDataset(
        "valid",
        tokenizer
    )

    # Training arguments
    name_suffix = "retro" if retro else "forward"
    training_args = TrainingArguments(
        output_dir=f"./results_{name_suffix}",
        evaluation_strategy="steps",
        learning_rate=2e-4,  # Higher learning rate since we're training from scratch
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        eval_accumulation_steps=10,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=100,  # More epochs since we're training from scratch
        logging_steps=100,
        save_steps=1000,
        warmup_steps=2000,
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
        eval_steps=1000,
        metric_for_best_model="eval_token_accuracy",
        save_only_model=True,
        save_safetensors=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    # score = trainer.evaluate()
    # print(score)

    # Train the model
    trainer.train(resume_from_checkpoint=False)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--retro", action="store_true")
    args = parser.parse_args()
    retro = args.retro
    main(retro)
