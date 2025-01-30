import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from transformers import Trainer, TrainingArguments
from torch.utils.tensorboard import SummaryWriter
import glob

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
        print(f"Loaded {self.embedding_files[idx]}")
        print(f"Loaded {smile_file}")
        # print all shapes
        print(f"embedding: {embedding.shape}")
        print(f"input_ids: {tokens['input_ids'].shape}")
        print(f"attention_mask: {tokens['attention_mask'].shape}")
        print(f"labels: {tokens['input_ids'].shape}")
        return {
            "encoder_hidden_states": embedding,
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": tokens["input_ids"].squeeze()
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

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        learning_rate=1e-4,  # Constant learning rate
        logging_dir='./logs',
        logging_steps=100,
        save_steps=1000,
        report_to=["tensorboard"],
        lr_scheduler_type="constant",  # Use constant learning rate
    )

    # Create dataset
    train_dataset = MoleculeDataset(
        data_dir="ZINK_NP",
        tokenizer=tokenizer
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train model
    trainer.train()

    # Save final model
    trainer.save_model("./molecule_decoder_final")
    writer.close()


def generate_molecules(model, embeddings, tokenizer, max_length=512):
    """
    Generate SMILES strings from embeddings
    """
    with torch.no_grad():
        outputs = model.generate(
            encoder_hidden_states=embeddings,
            max_length=max_length,
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True
        )

        # Decode outputs to SMILES strings
        smiles = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return smiles


if __name__ == "__main__":
    main()
