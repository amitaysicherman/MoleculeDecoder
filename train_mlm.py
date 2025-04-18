from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import torch

from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReactionMolsDataset(Dataset):
    def __init__(self, base_dir="USPTO", split="train"):
        self.base_dir = base_dir
        self.split = split
        self.max_seq_len = 10
        with open(f"{base_dir}/src-{split}.txt") as f:
            src_lines = f.read().splitlines()
        self.src_lines = [line.replace(" ", "").split(".") for line in src_lines]
        with open(f"{base_dir}/tgt-{split}.txt") as f:
            tgt_lines = f.read().splitlines()
        self.tgt_lines = [line.replace(" ", "").split(".") for line in tgt_lines]
        self.molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True
        )
        for param in self.molformer.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

    def __len__(self):
        return len(self.src_lines)

    # def apply_molformer(self, smiles):
    #     tokens = self.tokenizer(smiles, padding="max_length", truncation=True, max_length=75, return_tensors="pt")
    #     with torch.no_grad():
    #         outputs = self.molformer(**tokens)
    #     return outputs.pooler_output  # shape: (seq_len, hidden_size)
    #
    # def __getitem__(self, idx):
    #     src, tgt = self.src_lines[idx], self.tgt_lines[idx]
    #     src_emb = self.apply_molformer(src)
    #     tgt_emb = self.apply_molformer(tgt)
    #     src_emb = src_emb.unsqueeze(0)
    #     tgt_emb = tgt_emb.unsqueeze(0)
    #     return src_emb, tgt_emb
    def _get_embeddings(self, smiles_list):
        """Tokenizes and gets embeddings from MoLFormer"""
        tokens = self.tokenizer(smiles_list, padding="max_length", truncation=True, max_length=75, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}  # Move tokens to GPU

        if self.molformer.device != device:
            self.molformer = self.molformer.to(device)  # Move MolFormer to GPU once

        with torch.no_grad():
            outputs = self.molformer(**tokens)
        return outputs.pooler_output, tokens  # (seq_len, hidden_dim)

    def _pad_and_mask(self, embeddings):
        """Pads embeddings to max_seq_len and creates a proper attention mask"""
        seq_len = embeddings.shape[0]  # Get actual length
        embeddings = F.pad(embeddings, (0, 0, 0, self.max_seq_len - seq_len))  # Pad to max_seq_len
        mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        mask[:seq_len] = 1  # Mark real tokens as 1
        return embeddings, mask

    def __getitem__(self, idx):
        src, tgt = self.src_lines[idx], self.tgt_lines[idx]

        # Get embeddings and masks
        src_emb, src_tokens = self._get_embeddings(src)
        tgt_emb, tgt_tokens = self._get_embeddings(tgt)

        # Pad embeddings and create proper masks
        src_emb, src_mask = self._pad_and_mask(src_emb)
        tgt_emb, tgt_mask = self._pad_and_mask(tgt_emb)

        # Create decoder input (shift target embeddings right)
        decoder_input = torch.zeros_like(tgt_emb)
        decoder_input[:, 1:] = tgt_emb[:, :-1]

        # Create decoder mask (shift target mask right)
        decoder_mask = torch.zeros_like(tgt_mask)
        decoder_mask[1:] = tgt_mask[:-1]
        decoder_mask[0] = 1  # First token is always attended

        return {
            'input_vectors': src_emb,
            'decoder_input_vectors': decoder_input,
            'labels': tgt_emb,
            'attention_mask': src_mask,
            'decoder_attention_mask': decoder_mask,
            'decoder_input_tokens': tgt_tokens['input_ids'],
        }


if __name__ == "__main__":
    dataset = ReactionMolsDataset()
    print(dataset[0])
    for k, v in dataset[0].items():
        print(k, v.shape)
