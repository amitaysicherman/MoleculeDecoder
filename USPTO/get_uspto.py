import pickle
import os
import pandas as pd
from rdkit import Chem

output_file = 'USPTO/data.pickle'
if not os.path.exists(output_file):
    cmd = 'wget "https://az.app.box.com/index.php?rm=box_download_shared_file&shared_name=7eci3nd9vy0xplqniitpk02rbg9q2zcq&file_id=f_854847813119" -O USPTO/data.pickle'
    os.system(cmd)

with open('USPTO/data.pickle', 'rb') as f:
    data: pd.DataFrame = pickle.load(f)


def mol_to_smiles(mol):
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


data['reactants_smiles'] = data['reactants_mol'].apply(mol_to_smiles)
data['products_smiles'] = data['products_mol'].apply(mol_to_smiles)
data['reagents_smiles'] = data['reagents_mol'].apply(mol_to_smiles)

for split in ['train', 'valid', 'test']:
    data_split = data[data['set'] == split]
    with open(f'USPTO/reactants-{split}.txt', 'w') as f:
        for _, row in data_split.iterrows():
            f.write(f"{row['reactants_smiles']}\n")
    with open(f'USPTO/products-{split}.txt', 'w') as f:
        for _, row in data_split.iterrows():
            f.write(f"{row['products_smiles']}\n")
    with open(f'USPTO/reagents-{split}.txt', 'w') as f:
        for _, row in data_split.iterrows():
            f.write(f"{row['reagents_smiles']}\n")


