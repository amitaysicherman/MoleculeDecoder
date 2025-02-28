import torch
from train_decoder import create_model
import tqdm
import numpy as np

USPTO = "USPTO"
pubchem = "pubchem"

mode = USPTO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from train_decoder import create_model

decoder_model, tokenizer = create_model()
state_dict = torch.load("results_decoder/checkpoint-195000/pytorch_model.bin", map_location=torch.device('cpu'))
decoder_model.load_state_dict(state_dict, strict=True)
decoder_model = decoder_model.to(device).eval()
if mode == USPTO:
    with open("USPTO/all_mols.txt", "r") as f:
        all_uspto_mols = f.read().splitlines()
        # all_uspto_mols = [s.strip().split()[1] for s in all_uspto_mols]
else:  # pubchem
    with open("pubchem-canonical/CID-SMILES-CANONICAL.smi", "r") as f:
        all_uspto_mols = f.read().splitlines()
        all_uspto_mols = [s.strip().split()[1] for s in all_uspto_mols]

is_correct = []
tokens_accuracy = []
pbar = tqdm.tqdm(all_uspto_mols, total=len(all_uspto_mols))
for smiles in pbar:
    tokens = tokenizer([smiles], padding="max_length", truncation=True, max_length=75, return_tensors="pt")
    if tokens["input_ids"][0, -1] != tokenizer.pad_token_id:
        print(f"Skipping {smiles} because it is too long")
        continue
    labels = tokens["input_ids"].clone()
    # replace pad tokens with -100
    labels[labels == tokenizer.pad_token_id] = -100
    tokens["labels"] = labels
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = decoder_model(**tokens)
    preds = output.logits.argmax(-1)

    mask = labels != -100

    total_tokens = mask.sum()
    correct_tokens = ((preds == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens
    print()
    print(preds[0].tolist())
    print(labels[0].tolist())
    print(f"Token accuracy: ({correct_tokens} / {total_tokens}) = {token_accuracy}")
    pred_smiles = tokenizer.decode(preds[0], skip_special_tokens=True)
    is_correct.append(pred_smiles == smiles)
    tokens_accuracy.append(token_accuracy.item())
    pbar.set_postfix({"correct": np.mean(is_correct), "token_accuracy": np.mean(tokens_accuracy)})
