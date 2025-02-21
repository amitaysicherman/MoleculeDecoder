import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast
)
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from collections import Counter
import numpy as np
from pathlib import Path

tokenizer_file = "USPTO/tokenizer/"


def build_vocab_from_files(src_file: str, tgt_file: str, min_freq: int = 2):
    """Build vocabulary from source and target files, with caching to disk"""
    # Check if tokenizer already exists
    if Path(tokenizer_file).exists():
        print(f"Loading existing tokenizer from {tokenizer_file}")
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_file)

    print(f"Building new tokenizer from {src_file} and {tgt_file}")
    counter = Counter()

    # Read source file
    with open(src_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)

    # Read target file
    with open(tgt_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)

    # Create vocabulary dictionary with special tokens first
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }

    # Add frequent tokens to vocabulary
    idx = len(vocab)
    for token, count in counter.most_common():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1

    print(f"Vocabulary size: {len(vocab)}")

    # Create the tokenizer
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))

    # Set normalizer to none since we want to keep the exact whitespace tokenization
    tokenizer.normalizer = normalizers.Sequence([])

    # Set pre-tokenizer to just split on whitespace
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # Create the wrapped tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>"
    )

    # Save tokenizer to disk
    print(f"Saving tokenizer to {tokenizer_file}")
    wrapped_tokenizer.save_pretrained(tokenizer_file)

    return wrapped_tokenizer


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read source and target files
        with open(src_file, 'r', encoding='utf-8') as f:
            self.source_texts = [line.strip() for line in f]
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.target_texts = [line.strip() for line in f]

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source = self.source_texts[idx]
        target = self.target_texts[idx]

        # Add special tokens and tokenize
        source = f"{self.tokenizer.bos_token} {source} {self.tokenizer.eos_token}"
        target = f"{self.tokenizer.bos_token} {target} {self.tokenizer.eos_token}"

        # Tokenize inputs
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize targets
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        label = target_encoding['input_ids'].clone()
        label[label == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': label.squeeze()
        }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[0]
    # Get predicted token indices
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


def main(retro=False):
    # Build vocabulary and create tokenizer
    tokenizer = build_vocab_from_files(
        "USPTO/src-train.txt",
        "USPTO/tgt-train.txt",
        min_freq=2
    )

    # Initialize model configuration
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=768,  # Hidden size
        d_kv=64,  # Size of key/value heads
        d_ff=2048,  # Intermediate feed-forward size
        num_layers=6,  # Number of encoder/decoder layers
        num_heads=8,  # Number of attention heads
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
    src_prefix="src" if not retro else "tgt"
    tgt_prefix="tgt" if not retro else "src"
    train_dataset = TranslationDataset(
        f"USPTO/{src_prefix}-train.txt",
        f"USPTO/{tgt_prefix}-train.txt",
        tokenizer
    )
    eval_dataset = TranslationDataset(
        f"USPTO/{src_prefix}-valid.txt",
        f"USPTO/{tgt_prefix}-valid.txt",
        tokenizer
    )

    # Training arguments
    name_suffix = "retro" if retro else "forward"
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./results_{name_suffix}",
        evaluation_strategy="steps",
        learning_rate=2e-4,  # Higher learning rate since we're training from scratch
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        eval_accumulation_steps=10,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,  # More epochs since we're training from scratch
        logging_steps=100,
        save_steps=1000,
        warmup_steps=2000,
        lr_scheduler_type="constant",
        load_best_model_at_end=True,
        eval_steps=1000,
        metric_for_best_model="eval_token_accuracy",
    )

    # Initialize data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    score = trainer.evaluate()
    print(score)

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")


# def translate(text, model, tokenizer, max_length=128):
#     # Add special tokens
#     text = f"{tokenizer.bos_token} {text} {tokenizer.eos_token}"
#
#     # Tokenize input
#     inputs = tokenizer(
#         text,
#         max_length=max_length,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )
#
#     # Generate translation
#     outputs = model.generate(
#         input_ids=inputs['input_ids'],
#         attention_mask=inputs['attention_mask'],
#         max_length=max_length,
#         num_beams=4,
#         early_stopping=True
#     )
#
#     # Decode translation
#     translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return translation
#
#
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--retro", action="store_true")
    args = parser.parse_args()
    retro = args.retro
    main(retro)
