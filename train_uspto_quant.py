import os
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from t5_quant_model import T5ForResidualQuantization
from transformers import T5Config
from typing import Dict, List
import numpy as np

quantization_codebook_size = 512  # Adjust based on your data


class ResidualQuantizationDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str, max_length: int = 5*64):
        """
        Args:
            src_path: Path to source tokens file
            tgt_path: Path to target tokens file
            max_length: Maximum sequence length
        """
        self.max_length = max_length

        # Read source and target files
        with open(src_path, 'r') as f:
            self.src_data = [
                [int(token) for token in line.strip().split()]
                for line in f
            ]

        with open(tgt_path, 'r') as f:
            self.tgt_data = [
                [int(token) for token in line.strip().split()]
                for line in f
            ]

        assert len(self.src_data) == len(self.tgt_data), "Source and target files must have same number of lines"

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        src_tokens = self.src_data[idx]
        tgt_tokens = self.tgt_data[idx]

        # Pad sequences if needed
        if len(src_tokens) < self.max_length:
            src_tokens = src_tokens + [quantization_codebook_size] * (self.max_length - len(src_tokens))
        else:
            src_tokens = src_tokens[:self.max_length]

        if len(tgt_tokens) < self.max_length:
            tgt_tokens = tgt_tokens + [quantization_codebook_size] * (self.max_length - len(tgt_tokens))
        else:
            tgt_tokens = tgt_tokens[:self.max_length]

        return {
            'input_ids': torch.tensor(src_tokens, dtype=torch.long),
            'labels': torch.tensor(tgt_tokens, dtype=torch.long),
            'attention_mask': torch.ones(len(src_tokens), dtype=torch.long)
        }


def create_datasets(base_dir: str):
    """Create train, validation, and test datasets"""
    datasets = {}

    for split in ['train', 'valid', 'test']:
        src_path = os.path.join(base_dir, f'src-{split}.txt')
        tgt_path = os.path.join(base_dir, f'tgt-{split}.txt')
        datasets[split] = ResidualQuantizationDataset(src_path, tgt_path)

    return datasets


def compute_metrics(eval_pred):
    """Custom metrics computation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    # Mask out padding tokens
    mask = labels != quantization_codebook_size
    correct = ((predictions == labels) & mask).sum()
    total = mask.sum()

    return {
        'accuracy': correct / total,
    }


def main():
    # Configuration
    base_dir = "USPTO_Q"
    num_quantization = 64  # Adjust based on your data

    # Create datasets
    datasets = create_datasets(base_dir)

    config = T5Config(
        vocab_size=quantization_codebook_size,
        d_model=768,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=6,
        is_encoder_decoder=True,
        is_decoder=True,
        num_heads=8,
        decoder_start_token_id=[quantization_codebook_size],
    )
    model = T5ForResidualQuantization(config, num_quantization)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        warmup_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_safetensors=False,
        lr_scheduler_type="constant",  # Use constant learning rate
        eval_accumulation_steps=2,
        learning_rate=1e-4,  # Constant learning rate
        save_total_limit=1,
        report_to=["tensorboard"],
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['valid'],
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Evaluate on test set
    test_results = trainer.evaluate(datasets['test'])
    print(f"Test results: {test_results}")

    # Save final model
    trainer.save_model("./final_model")


if __name__ == "__main__":
    main()

