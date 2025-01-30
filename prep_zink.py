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

chunks = os.listdir("ZINK")
pbar = tqdm(chunks)
total_smiles = 0
for chunk in pbar:
    with open(f"ZINK/{chunk}") as f:
        smiles_ids = f.read().splitlines()
    smiles = [smiles_id.split()[0] for smiles_id in smiles_ids]
    total_smiles += len(smiles)
    inputs = tokenizer(smiles, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeds = outputs.pooler_output.cpu().numpy()
    np.save(f"ZINK/{chunk}.npy", embeds)
    pbar.set_description(f"Total SMILES: {total_smiles}")
