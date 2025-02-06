import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from datetime import datetime
import logging
from transformers import AutoModel


def get_mol_embeddings(molformer, input_ids, token_attention_mask):
    """
    Process multiple molecules in parallel through MolFormer
    """
    batch_size, max_seq_len, seq_len = input_ids.shape

    # Reshape for parallel processing
    flat_input_ids = input_ids.view(-1, seq_len)  # (batch_size * max_seq_len, 75)
    flat_attention_mask = token_attention_mask.view(-1, seq_len)  # (batch_size * max_seq_len, 75)

    # Process through MolFormer in chunks to avoid OOM
    chunk_size = 2048  # Adjust based on your GPU memory
    all_embeddings = []

    for i in range(0, flat_input_ids.size(0), chunk_size):
        chunk_input_ids = flat_input_ids[i:i + chunk_size]
        chunk_attention_mask = flat_attention_mask[i:i + chunk_size]

        with torch.no_grad():
            outputs = molformer(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask
            )
            all_embeddings.append(outputs.pooler_output)

    # Combine chunks
    embeddings = torch.cat(all_embeddings, dim=0)  # (batch_size * max_seq_len, hidden_size)

    # Reshape back to original dimensions
    embeddings = embeddings.view(batch_size, max_seq_len, -1)  # (batch_size, max_seq_len, hidden_size)
    return embeddings


class Trainer:
    def __init__(
            self,
            model,
            train_dataset,
            val_dataset,
            batch_size=32,
            num_epochs=10,
            lr=1e-4,
            device="cuda",
            output_dir="outputs"
    ):
        self.model = model.to(device)
        self.molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            deterministic_eval=True,
            trust_remote_code=True
        ).to(device).eval()
        for param in self.molformer.parameters():
            param.requires_grad = False

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=False
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=False
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
        )
        self.device = device
        self.num_epochs = num_epochs

        # Setup logging
        self.output_dir = output_dir
        self.experiment_dir = os.path.join(
            output_dir,
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Log hyperparameters
        self.log_hyperparameters({
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr
        })

    def setup_logging(self):
        """Setup logging configuration"""
        self.log_file = os.path.join(self.experiment_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

    def log_hyperparameters(self, hyperparams):
        """Log hyperparameters to a JSON file"""
        hyper_file = os.path.join(self.experiment_dir, 'hyperparameters.json')
        with open(hyper_file, 'w') as f:
            json.dump(hyperparams, f, indent=2)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        with tqdm(self.train_loader, desc=f"Epoch {epoch}") as pbar:
            for i, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                self.optimizer.zero_grad()
                intput_embeddings = get_mol_embeddings(
                    self.molformer,
                    batch['src_input_ids'],
                    batch['src_token_attention_mask']
                )
                output_embeddings = get_mol_embeddings(
                    self.molformer,
                    batch['tgt_input_ids'],
                    batch['tgt_token_attention_mask']
                )
                loss = self.model(
                    src_embeddings=intput_embeddings,
                    tgt_embeddings=output_embeddings,
                    src_mol_attention_mask=batch['src_mol_attention_mask'],
                    tgt_mol_attention_mask=batch['tgt_mol_attention_mask']
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (i + 1)})

            # Log metrics
            if i % 100 == 0:
                logging.info(f"Epoch {epoch} - Batch {i}/{len(self.train_loader)} - Loss: {loss.item() :.4f}")

        avg_loss = total_loss / len(self.train_loader)
        logging.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                intput_embeddings = get_mol_embeddings(
                    self.molformer,
                    batch['src_input_ids'],
                    batch['src_token_attention_mask']
                )
                output_embeddings = get_mol_embeddings(
                    self.molformer,
                    batch['tgt_input_ids'],
                    batch['tgt_token_attention_mask']
                )
                loss= self.model(
                    src_embeddings=intput_embeddings,
                    tgt_embeddings=output_embeddings,
                    src_mol_attention_mask=batch['src_mol_attention_mask'],
                    tgt_mol_attention_mask=batch['tgt_mol_attention_mask']
                )



                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        logging.info(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
            val_loss = self.validate()

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    checkpoint,
                    os.path.join(self.experiment_dir, 'best_model.pt')
                )
                logging.info(f"Saved new best model with validation loss: {val_loss:.4f}")

            # Regular checkpoint every 5 epochs
            if epoch % 5 == 0:
                torch.save(
                    checkpoint,
                    os.path.join(self.experiment_dir, f'checkpoint_epoch_{epoch}.pt')
                )
                logging.info(f"Saved checkpoint at epoch {epoch}")
