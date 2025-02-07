all_mols = []
base_dir="USPTO"
for split in ["train","valid","test"]:
    for side in ["src","tgt"]:
        with open(f"{base_dir}/{side}-{split}.txt") as f:
            lines = f.read().splitlines()
        mols = [line.replace(" ", "").split(".") for line in lines]
        all_mols.extend([m for mol in mols for m in mol])
print(len(all_mols))
with open("USPTO/all_mols.txt", "w") as f:
    f.write("\n".join(all_mols))