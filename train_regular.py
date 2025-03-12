import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    Trainer, TrainingArguments
)
import numpy as np
from transformers import AutoTokenizer
from autoencoder.data import preprocess_smiles
import random

class TranslationDataset(Dataset):
    def __init__(self, split, is_retro=False, parouts=0):
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.max_len = 10
        self.max_mol_len = 75
        self.skip_unk = True
        base_dir = "USPTO"
        self.is_retro = is_retro
        # Read source and target files
        if not parouts:
            with open(f"{base_dir}/reactants-{split}.txt") as f:
                self.reactants = f.read().splitlines()
            with open(f"{base_dir}/products-{split}.txt") as f:
                self.products = f.read().splitlines()
            with open(f"{base_dir}/reagents-{split}.txt") as f:
                self.reagents = f.read().splitlines()
        else:
            with open(f"PaRoutes/{split}.src") as f:
                self.reactants = f.read().splitlines()
                self.reactants = [r.split(".")[0] for r in self.reactants]
            with open(f"PaRoutes/{split}.tgt") as f:
                self.products = f.read().splitlines()
            self.reagents = ["" for _ in self.reactants]
        assert len(self.reactants) == len(self.products) == len(self.reagents)

    def __len__(self):
        return len(self.products)

    def mols_smiles_to_tokens(self, mols_smiles):
        mols = mols_smiles.strip().split(".")

        if len(mols) > self.max_len:
            return None
        mols = [preprocess_smiles(m) for m in mols]
        if None in mols:
            return None
        mols=".".join(mols)
        tokens = self.tokenizer(mols, padding="max_length", truncation=True, max_length=self.max_mol_len,
                                return_tensors="pt")
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        if tokens['attention_mask'][-1] != 0:
            return None
        if self.skip_unk:
            if self.tokenizer.unk_token_id in tokens['input_ids']:
                return None
        return tokens

    def __getitem__(self, idx):
        do_again = True
        while do_again:

            reac = self.reactants[idx]
            react_tokens = self.mols_smiles_to_tokens(reac)

            reag = self.reagents[idx]
            reag_tokens = self.mols_smiles_to_tokens(reag)

            prod = self.products[idx]
            prod_tokens = self.mols_smiles_to_tokens(prod)

            if react_tokens is None or reag_tokens is None or prod_tokens is None:
                idx = np.random.randint(0, len(self.products))
            else:
                do_again = False
        if self.is_retro:
            source_encoding = prod_tokens
            target_encoding = react_tokens
        else:
            source_encoding = {k: torch.cat([react_tokens[k], reag_tokens[k]], dim=0) for k in react_tokens}
            target_encoding = prod_tokens

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
        "sample_accuracy": perfect_match_accuracy,
        "token_accuracy": token_accuracy
    }


def main(retro=False,batch_size=256,parouts=0):
    # Build vocabulary and create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
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
    train_dataset = TranslationDataset(
        "train",
        is_retro=retro,
        parouts=parouts
    )
    val_dataset = TranslationDataset(
        "valid",
        is_retro=retro,
        parouts=parouts
    )
    train_subset_random_indices = random.sample(range(len(train_dataset)), len(val_dataset))
    train_subset = torch.utils.data.Subset(train_dataset, train_subset_random_indices)


    # Training arguments
    name_suffix = "retro" if retro else "forward"
    training_args = TrainingArguments(
        output_dir=f"./res_{name_suffix}",
        evaluation_strategy="steps",
        learning_rate=2e-4,  # Higher learning rate since we're training from scratch
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=10,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=100,  # More epochs since we're training from scratch
        logging_steps=100,
        save_steps=1000,
        warmup_steps=2000,
        logging_dir=f"./logs_{name_suffix}",
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
        eval_steps=1000,
        metric_for_best_model="eval_validation_token_accuracy",
        save_only_model=True,
        save_safetensors=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={'validation': val_dataset, "train": train_subset},
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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--parouts", type=int, default=0)
    args = parser.parse_args()
    retro = args.retro
    main(retro, args.batch_size, args.parouts)
