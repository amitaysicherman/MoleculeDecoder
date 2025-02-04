import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os


class ReactionMolsDataset(Dataset):
    def __init__(self, base_dir="USPTO", split="train", debug=False):
        if not os.path.exists(base_dir):
            raise ValueError(f"Directory {base_dir} does not exist")
        if split not in ["train", "valid", "test"]:
            raise ValueError(f"Split {split} not recognized")

        self.base_dir = base_dir
        self.split = split
        self.max_seq_len = 10  # Maximum number of molecules per reaction

        # Load data
        with open(f"{base_dir}/src-{split}.txt") as f:
            src_lines = f.read().splitlines()
        self.src_lines = [line.replace(" ", "").split(".") for line in src_lines]
        src_too_long = [len(s) > self.max_seq_len for s in self.src_lines]
        with open(f"{base_dir}/tgt-{split}.txt") as f:
            tgt_lines = f.read().splitlines()
        self.tgt_lines = [line.replace(" ", "").split(".") for line in tgt_lines]
        tgt_too_long = [len(t) > self.max_seq_len for t in self.tgt_lines]
        line_too_long = [s or t for s, t in zip(src_too_long, tgt_too_long)]
        self.src_lines = [s for s, too_long in zip(self.src_lines, line_too_long) if not too_long]
        self.tgt_lines = [t for t, too_long in zip(self.tgt_lines, line_too_long) if not too_long]
        if debug:
            self.src_lines = self.src_lines[:1]
            self.tgt_lines = self.tgt_lines[:1]
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True
        )


    def _tokenize_and_pad(self, smiles_list):
        """Tokenize list of SMILES and pad to max_seq_len"""
        # Tokenize each SMILES string
        tokens = [
            self.tokenizer(
                s,
                padding="max_length",
                truncation=True,
                max_length=75,
                return_tensors="pt"
            ) for s in smiles_list if s  # Skip empty strings
        ]

        # Pad number of molecules to max_seq_len
        num_mols = len(tokens)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:num_mols] = 1

        # Pad with empty tokenizations if needed
        while len(tokens) < self.max_seq_len:
            tokens.append(
                self.tokenizer(
                    "",
                    padding="max_length",
                    truncation=True,
                    max_length=75,
                    return_tensors="pt"
                )
            )

        # Stack all input_ids and attention_masks
        input_ids = torch.stack([t['input_ids'].squeeze(0) for t in tokens])
        token_attention_mask = torch.stack([t['attention_mask'].squeeze(0) for t in tokens])

        return {
            'input_ids': input_ids,  # Shape: (max_seq_len, 75)
            'token_attention_mask': token_attention_mask,  # Shape: (max_seq_len, 75)
            'mol_attention_mask': attention_mask,  # Shape: (max_seq_len,)
        }

    def __getitem__(self, idx):
        src, tgt = self.src_lines[idx], self.tgt_lines[idx]

        # Filter out empty strings and take only valid SMILES
        src = [s for s in src if s.strip()]
        tgt = [t for t in tgt if t.strip()]

        # Get tokenized inputs for source and target
        src_tokens = self._tokenize_and_pad(src)
        tgt_tokens = self._tokenize_and_pad(tgt)

        return {
            'src_input_ids': src_tokens['input_ids'],  # (max_seq_len, 75)
            'src_token_attention_mask': src_tokens['token_attention_mask'],  # (max_seq_len, 75)
            'src_mol_attention_mask': src_tokens['mol_attention_mask'],  # (max_seq_len,)
            'tgt_input_ids': tgt_tokens['input_ids'],  # (max_seq_len, 75)
            'tgt_token_attention_mask': tgt_tokens['token_attention_mask'],  # (max_seq_len, 75)
            'tgt_mol_attention_mask': tgt_tokens['mol_attention_mask'],  # (max_seq_len,)
        }

    def __len__(self):
        return len(self.src_lines)


if __name__ == "__main__":
    # Test the dataset
    dataset = ReactionMolsDataset("USPTO", "train")
    print(f"Dataset size: {len(dataset)}")

    # Get first item
    item = dataset[0]
    print("\nShapes of first item:")
    for k, v in item.items():
        print(f"{k}: {v.shape}")

    # Test data loader
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print("\nShapes of first batch:")
    for k, v in batch.items():
        print(f"{k}: {v.shape}")