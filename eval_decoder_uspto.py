import torch
from train_decoder import create_model
import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

decoder_model, tokenizer = create_model()
decoder_model.load_state_dict(
    torch.load("results/checkpoint-55000/pytorch_model.bin", map_location=torch.device('cpu')), strict=True)
decoder_model = decoder_model.to(device).eval()




with open("pubchem-canonical/CID-SMILES-CANONICAL.smi", "r") as f:
    all_uspto_mols = f.read().splitlines()
    all_uspto_mols = [s.strip().split()[1] for s in all_uspto_mols]


is_correct = []
pbar= tqdm.tqdm(all_uspto_mols,total=len(all_uspto_mols))
for smiles in pbar:
    tokens = tokenizer([smiles], padding="max_length", truncation=True, max_length=75, return_tensors="pt")
    labels = tokens["input_ids"].clone()
    # replace pad tokens with -100
    labels[labels == tokenizer.pad_token_id] = -100
    tokens["labels"] = labels
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = decoder_model(**tokens)
    preds= output.logits.argmax(-1)


    mask = labels != -100

    total_tokens = mask.sum()
    correct_tokens = ((preds == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens


    pred_smiles = tokenizer.decode(preds[0], skip_special_tokens=True)
    is_correct.append(pred_smiles == smiles)
    pbar.set_postfix({"correct": sum(is_correct)/len(is_correct), "token_accuracy": token_accuracy.item()})