from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import argparse
import os
import random
from tqdm import tqdm
random.seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index", type=int, default=10000000)
parser.add_argument("--samples", type=int, default=1000000)
parser.add_argument("--batch_size", type=int, default=1024)
args = parser.parse_args()
start_index = args.start_index
end_index = args.end_index
samples = args.samples
batch_size = args.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
molformer = AutoModel.from_pretrained(
    "ibm/MoLFormer-XL-both-10pct",
    deterministic_eval=True,
    trust_remote_code=True
).to(device).eval()
with open("pubchem-canonical/CID-SMILES-CANONICAL.smi") as f:
    smiles = f.readlines()
print(f"Total number of smiles: {len(smiles)}")
print(f"Start index: {start_index}")
print(f"End index: {end_index}")
smiles = smiles[start_index:end_index]
random.shuffle(smiles)
smiles = smiles[:samples]
smiles = [s.split(" ")[1].strip() for s in smiles]

all_outputs = []
for i in tqdm(range(0, len(smiles), batch_size)):
    smiles_batch = smiles[i:i + batch_size]
    input_tokens = tokenizer(smiles_batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
    input_ids = input_tokens["input_ids"].to(device)
    attention_mask = input_tokens["attention_mask"].to(device)
    mol_outputs = molformer(input_ids, attention_mask=attention_mask)
    np_output = mol_outputs.pooler_output.detach().cpu().numpy()
    all_outputs.append(np_output)
np_output = np.concatenate(all_outputs, axis=0)
output_name = f"molformer_pubchem_{start_index}_{end_index}.npy"

np.save(output_name, np_output)
