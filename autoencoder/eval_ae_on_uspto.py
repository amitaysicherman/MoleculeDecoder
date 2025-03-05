from autoencoder.train_mvm import ReactionMolsDataset
import torch
from autoencoder.data import get_tokenizer
from autoencoder.train_mvm import device, get_last_cp
from tqdm import tqdm
mem = set()

dataset = ReactionMolsDataset()
from autoencoder.model import get_model

tokenizer = get_tokenizer()
model = get_model('ae', "m", tokenizer)
state_dict = torch.load(f"{get_last_cp('res_auto/ae_m')}/pytorch_model.bin", map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
for param in model.parameters():
    param.requires_grad = False
model.to(device).eval()
pbar = tqdm(dataset)
src_correct = 0
src_total = 0
tgt_correct = 0
tgt_total = 0
for data in pbar:
    for i in range(10):
        for prefix in ['src', 'tgt']:
            if data[f'{prefix}_mol_attention_mask'][i] == 0:
                continue
            mol = data[f'{prefix}_input_ids'][i]
            mol_mask = data[f'{prefix}_token_attention_mask'][i]

            mol_key = tuple(mol[mol_mask == 1].detach().cpu().numpy().tolist())
            if mol_key in mem:
                continue
            mem.add(mol_key)
            mol = mol.unsqueeze(0).to(device)
            mol_mask = mol_mask.unsqueeze(0).to(device)
            labels = mol.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            output = model(mol, mol_mask, labels)
            prediction = output.logits.argmax(-1).squeeze().detach()
            prediction = prediction[mol_mask[0] == 1].cpu().numpy().tolist()[:-1]
            get_labels = labels[0][mol_mask[0] == 1].cpu().numpy().tolist()[1:]
            if prefix == 'src':
                src_correct += prediction == get_labels
                src_total += 1
            else:
                tgt_correct += prediction == get_labels
                tgt_total += 1
            # if prediction != get_labels:
            #     prediction = tokenizer.decode(prediction)
            #     get_labels = tokenizer.decode(get_labels)
                # print(f"{prefix} Prediction: {prediction} Original: {get_labels}")
    pbar.set_description(f"Src Accuracy: {src_correct / src_total:.2f}({src_correct}/{src_total}) Tgt Accuracy: {tgt_correct / tgt_total:.2f}({tgt_correct}/{tgt_total})")