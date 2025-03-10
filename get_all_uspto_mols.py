from autoencoder.data import preprocess_smiles
from tqdm import tqdm
all_mols = set()
train_mols = set()
valid_mols = set()
test_mols = set()
name_to_set= {
    "train": train_mols,
    "valid": valid_mols,
    "test": test_mols
}
base_dir = "USPTO"
for split in ["train", "valid", "test"]:
    for side in ["products", "reactants"]:
        with open(f"{base_dir}/{side}-{split}.txt") as f:
            lines = f.read().splitlines()
        mols = [line.replace(" ", "").split(".") for line in lines]
        mols = [[preprocess_smiles(mol) for mol in m] for m in tqdm(mols)]
        all_mols.update({m for mol in mols for m in mol})
        name_to_set[split].update({m for mol in mols for m in mol})



# print("All:", len(all_mols))
# print("Train:", len(train_mols))
# print("Valid:", len(valid_mols))
print(f"All: {len(all_mols):,}")
print(f"Train: {len(train_mols):,}")
print(f"Valid: {len(valid_mols):,}")
print(f"Test: {len(test_mols):,}")
valid_no_train = valid_mols - train_mols
test_no_train = test_mols - train_mols
print(f"Valid no train: {len(valid_no_train):,}")
print(f"Test no train: {len(test_no_train),}")
# with open("USPTO/all_mols.txt", "w") as f:
#     f.write("\n".join(all_mols))
