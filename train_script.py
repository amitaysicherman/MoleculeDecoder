import torch
from transformers import T5Config
import argparse
import os

from dataset import ReactionMolsDataset
from model import VectorT5
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Vector T5 Model')
    parser.add_argument('--data_dir', type=str, default="USPTO",
                        help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str, default="outputs",
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--grad_acc_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--size', type=str, default="small",
                        help='Size of the model (small or large)')
    return parser.parse_args()


def get_model(debug=False, size="large"):
    if debug:
        config = T5Config(
            d_model=64,
            d_kv=32,
            d_ff=128,
            num_layers=2,
            num_heads=2,
            input_dim=768,
            output_dim=768,
            dropout=0.0
        )
    else:
        if size == "small":
            config = T5Config(
                d_model=256,
                d_kv=32,
                d_ff=512,
                num_layers=4,
                num_heads=4,
                input_dim=768,
                output_dim=768,
                dropout=0.1
            )
        else:
            config = T5Config(
                d_model=768,
                d_ff=2048,
                num_layers=6,
                num_heads=8,
                input_dim=768,
                output_dim=768,
                dropout=0.1
            )
    model = VectorT5(config)
    return model


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize datasets
    train_dataset = ReactionMolsDataset(base_dir=args.data_dir, split="train", debug=args.debug)
    val_dataset = ReactionMolsDataset(base_dir=args.data_dir, split="valid", debug=args.debug)

    # Initialize model
    model = get_model(debug=args.debug)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size if not args.debug else 1,
        num_epochs=args.epochs if not args.debug else 200,
        lr=args.lr,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=args.output_dir
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
