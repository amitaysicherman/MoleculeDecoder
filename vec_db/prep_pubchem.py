from transformers import AutoModel, AutoTokenizer
from autoencoder.data import preprocess_smiles
import torch
import argparse
import numpy as np
from npy_append_array import NpyAppendArray
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_len = 75
batch_size = 2048

parser = argparse.ArgumentParser()
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index", type=int, default=-1)
args = parser.parse_args()

encoder = AutoModel.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct",
    deterministic_eval=True,
    trust_remote_code=True,
    use_safetensors=False
)

tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

with open("pubchem-canonical/CID-SMILES-CANONICAL.smi", 'r', encoding='utf-8') as f:
    data = f.read().splitlines()

start_index = args.start_index
end_index = args.end_index
if end_index == -1:
    end_index = len(data)
output_file = f"pubchem-canonical/db/{start_index}_{end_index}.npy"
output_text_file = f"pubchem-canonical/db/{start_index}_{end_index}.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
if os.path.exists(output_text_file):
    os.remove(output_text_file)
if os.path.exists(output_file):
    os.remove(output_file)
chunk_data = data[args.start_index:args.end_index]
for i in range(0, len(chunk_data), batch_size):
    smiles = chunk_data[i:i + batch_size]
    smiles = [preprocess_smiles(s) for s in smiles]
    if len(smiles) == 0:
        continue
    tokens = [tokenizer(s, padding="none", return_tensors="pt")[0] for s in smiles]
    too_long = [i for i, t in enumerate(tokens) if len(t.input_ids) > max_len]
    if len(too_long) > 0:
        tokens = [t for i, t in enumerate(tokens) if i not in too_long]
        smiles = [s for i, s in enumerate(smiles) if i not in too_long]
    tokens = tokenizer.pad(tokens, padding="max_length", max_length=max_len, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = encoder(**tokens)
    embeddings = outputs.pooler_output.detach().cpu().numpy()
    with NpyAppendArray(output_file, delete_if_exists=False) as npaa:
        npaa.append(embeddings)
    with open(output_text_file, 'a') as f:
        for s in smiles:
            f.write(s + "\n")
