import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import json
from datetime import datetime
import logging

class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        batch_size=32,
        num_epochs=10,
        lr=1e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        device="cuda",
        output_dir="outputs"
    ):
        self.model = model.to(device)
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
            weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        self.scaler = GradScaler()
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
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
            'lr': lr,
            'weight_decay': weight_decay,
            'gradient_accumulation_steps': gradient_accumulation_steps,
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
        
    def _prepare_decoder_inputs(self, tgt_embeddings):
        """Prepare decoder inputs by shifting target embeddings right"""
        # Create decoder input by shifting right
        decoder_inputs = torch.zeros_like(tgt_embeddings)
        decoder_inputs[:, 1:] = tgt_embeddings[:, :-1].clone()
        # First token is zero (will be replaced with EOS token in model)
        return decoder_inputs
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch}") as pbar:
            for i, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # First get target embeddings for labels and decoder input
                with torch.no_grad():
                    tgt_embeddings, tgt_attention_mask = self.model._get_mol_embeddings(
                        batch['tgt_input_ids'],
                        batch['tgt_token_attention_mask'],
                        batch['tgt_mol_attention_mask']
                    )

                # Create shifted decoder inputs
                decoder_inputs = self._prepare_decoder_inputs(tgt_embeddings)

                # Forward pass
                outputs = self.model(
                    src_input_ids=batch['src_input_ids'],
                    src_token_attention_mask=batch['src_token_attention_mask'],
                    src_mol_attention_mask=batch['src_mol_attention_mask'],
                    tgt_input_ids=batch['tgt_input_ids'],
                    tgt_token_attention_mask=batch['tgt_token_attention_mask'],
                    tgt_mol_attention_mask=batch['tgt_mol_attention_mask'],
                    labels=tgt_embeddings,  # Use target embeddings as labels
                    return_dict=True,
                )

                loss = outputs['loss'] / self.gradient_accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            if (i + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix({'loss': total_loss / (i + 1)})

            # Log metrics
            if i % 100 == 0:
                logging.info(f"Epoch {epoch} - Batch {i}/{len(self.train_loader)} - Loss: {loss.item() * self.gradient_accumulation_steps:.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        logging.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get target embeddings for labels
                tgt_embeddings, _ = self.model._get_mol_embeddings(
                    batch['tgt_input_ids'],
                    batch['tgt_token_attention_mask'],
                    batch['tgt_mol_attention_mask']
                )
                
                outputs = self.model(
                    src_input_ids=batch['src_input_ids'],
                    src_token_attention_mask=batch['src_token_attention_mask'],
                    src_mol_attention_mask=batch['src_mol_attention_mask'],
                    tgt_input_ids=batch['tgt_input_ids'],
                    tgt_token_attention_mask=batch['tgt_token_attention_mask'],
                    tgt_mol_attention_mask=batch['tgt_mol_attention_mask'],
                    labels=tgt_embeddings,
                    return_dict=True,
                )
                total_loss += outputs['loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        logging.info(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
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
