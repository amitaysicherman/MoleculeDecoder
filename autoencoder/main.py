import os.path

from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split

from autoencoder.data import AutoEncoderDataset
from autoencoder.model import get_model
import numpy as np
import glob


def args_to_name(args):
    return f"{args.model}_{args.size}"


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation
    """
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=-1)
    # use shift right to get the labels
    predictions = predictions[:, :-1]
    labels = labels[:, 1:]

    mask = labels != -100
    total_tokens = mask.sum()
    correct_tokens = ((predictions == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens
    correct_or_pad = (predictions == labels) | (~mask)
    correct_samples = correct_or_pad.all(axis=-1).sum()
    total_samples = len(labels)
    sample_accuracy = correct_samples / total_samples
    return {
        "token_accuracy": token_accuracy,
        "sample_accuracy": sample_accuracy,
    }


def check_checkpoint(base_dir):
    """
    Check if the base_dir contains a checkpoint
    """
    if os.path.exists(base_dir):
        return bool(glob.glob(f"{base_dir}/checkpoint-*"))
    else:
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ae")
    parser.add_argument("--size", type=str, default="s")
    args = parser.parse_args()

    dataset = AutoEncoderDataset()
    # split the dataset into training and validation

    val_size = 100_000
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    factor = 4 if args.size == 'l' else 1
    model = get_model(args.model, args.size, dataset.tokenizer)
    name = args_to_name(args)
    output_dir = f"res_auto/{name}"
    log_dir = f"logs_auto/{name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=1024 // factor,
        per_device_eval_batch_size=1024 // factor,
        eval_accumulation_steps=1,
        save_total_limit=1,
        save_safetensors=False,
        logging_dir=log_dir,
        evaluation_strategy="steps",
        report_to=["tensorboard"],
        logging_steps=500,
        save_steps=5000 * factor,
        eval_steps=5000 * factor,
        load_best_model_at_end=True,
        greater_is_better=True,
        learning_rate=1e-4/factor,
        lr_scheduler_type='constant',
        label_names=['labels'],
        seed=42,
        save_only_model=True,
        metric_for_best_model="eval_token_accuracy",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train(resume_from_checkpoint=check_checkpoint(output_dir))
