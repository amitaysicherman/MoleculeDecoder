import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from transformers import Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter
import glob
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    """
    Custom collate function to handle padding and stacking in the DataLoader.
    This ensures that sequences are padded to the max length in the batch,
    and embeddings are stacked together.
    """
    # Extract the input_ids, attention_mask, labels, and encoder_hidden_states from the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    encoder_hidden_states = [item['encoder_hidden_states'] for item in batch]

    # Pad the input sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 for ignoring padding in loss
    encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)  # Stack embeddings into a batch

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "encoder_hidden_states": encoder_hidden_states
    }
CHUCK_SIZE = 8096


class MoleculeDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.embedding_files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.embedding_files)

    def __getitem__(self, idx):
        # Load embedding
        embedding = np.load(self.embedding_files[idx])
        embedding = torch.from_numpy(embedding).float()
        emb_index = int(self.embedding_files[idx].split("_")[-1].split(".")[0])
        smile_file = self.embedding_files[idx].replace(f"_{emb_index}.npy", ".smi").replace("ZINK_NP", "ZINK")
        with open(smile_file) as f:
            smiles_ids = f.read().splitlines()
        smiles = [smiles_id.split()[0] for smiles_id in smiles_ids]
        if len(smiles) > CHUCK_SIZE:
            end_index = min(len(smiles), (emb_index + 1) * CHUCK_SIZE)
            smiles = smiles[emb_index * CHUCK_SIZE:end_index]
        tokens = self.tokenizer(
            smiles,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "encoder_hidden_states": embedding,
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": tokens["input_ids"].squeeze()
        }


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation
    """
    predictions, labels = eval_pred
    # Get argmax of predictions
    predictions = np.argmax(predictions, axis=-1)

    # Create mask for padding tokens
    mask = labels != -100

    # Calculate token accuracy (ignoring padding)
    total_tokens = mask.sum()
    correct_tokens = ((predictions == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens

    # Calculate sequence accuracy (all tokens correct)
    sequence_matches = [(pred == label).all() for pred, label in
                        zip(predictions[mask.reshape(predictions.shape[0], -1)],
                            labels[mask.reshape(labels.shape[0], -1)])]
    sequence_accuracy = np.mean(sequence_matches)

    return {
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy
    }


def main():
    # Load MolFormer tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

    # Initialize model config (matching roughly 50M params)
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=768,  # Same as MolFormer
        d_ff=2048,
        num_layers=6,  # Reduced from MolFormer's 12 to match param count
        num_decoder_layers=6,
        num_heads=8,
        is_decoder=True,
        is_encoder_decoder=False,  # We're only using the decoder
    )

    # Initialize model using T5ForConditionalGeneration
    model = T5ForConditionalGeneration(config)

    # Setup tensorboard
    writer = SummaryWriter('runs/molecule_decoder')

    # Create full dataset
    full_dataset = MoleculeDataset(
        data_dir="ZINK_NP",
        tokenizer=tokenizer
    )

    # Split dataset into train and evaluation
    train_size = int(0.9 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(
        full_dataset, [train_size, eval_size]
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-4,  # Constant learning rate
        logging_dir='./logs',
        logging_steps=100,
        save_steps=1000,
        eval_steps=500,  # Evaluate every 500 steps
        evaluation_strategy="steps",
        report_to=["tensorboard"],
        lr_scheduler_type="constant",  # Use constant learning rate
        load_best_model_at_end=True,
        metric_for_best_model="token_accuracy",
    )

    # Initialize trainer with evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=custom_collate_fn,
    )

    # Train model
    trainer.train()

    # Final evaluation
    final_metrics = trainer.evaluate()
    print("Final Evaluation Metrics:", final_metrics)

    # Save final model
    trainer.save_model("./molecule_decoder_final")
    writer.close()


if __name__ == "__main__":
    main()