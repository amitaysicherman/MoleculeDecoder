#  wget "https://ibm.ent.box.com/index.php?rm=box_download_shared_file&vanity_name=MoLFormer-data&file_id=f_1096127714688" -O zink.zip
#  unzip zink.zip
import os
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
print(model)
model.eval().to(device)
# print number of parameters
print(f"Number of parameters: {model.num_parameters(),}")

tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)


def split_into_max_len(smiles):
    max_len = 8096
    chunks = []
    if len(smiles) < max_len:
        return [smiles]
    for i in range(0, len(smiles), max_len):
        end = min(i + max_len, len(smiles))
        chunks.append(smiles[i:end])
    return chunks


chunks = os.listdir("ZINK")
os.makedirs("ZINK_NP", exist_ok=True)
pbar = tqdm(chunks)
total_smiles = 0
for chunk in pbar:
    with open(f"ZINK/{chunk}") as f:
        smiles_ids = f.read().splitlines()
    smiles = [smiles_id.split()[0] for smiles_id in smiles_ids]
    total_smiles += len(smiles)
    smiles_list = split_into_max_len(smiles)
    for i, smiles in enumerate(smiles_list):
        inputs = tokenizer(smiles, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeds = outputs.pooler_output.cpu().numpy()
        chunk_name = chunk.split(".")[0]
        np.save(f"ZINK_NP/{chunk_name}_{i}.npy", embeds)
    pbar.set_description(f"Total SMILES: {total_smiles}")
