import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vector_quantize_pytorch import ResidualVQ  # Import the ResidualVQ model
import argparse
import numpy as np
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_with_decoder(model):
    from train_decoder import _shift_right, create_model
    decoder_model, tokenizer = create_model()
    decoder_model.load_state_dict(
        torch.load("results_pubchem/checkpoint-90000/pytorch_model.bin", map_location=torch.device('cpu')), strict=True)
    decoder_model = decoder_model.to(device).eval()
    with open("pubchem-canonical/CID-SMILES-CANONICAL.smi", "r") as f:
        all_uspto_mols = f.read().splitlines()
        all_uspto_mols = [s.strip().split()[1] for s in all_uspto_mols]
    random.seed(42)
    all_uspto_mols = random.sample(all_uspto_mols, 250)
    is_correct = []
    token_accuracy = []
    is_correct_not_q = []
    token_accuracy_n = []
    pbar = tqdm(all_uspto_mols, total=len(all_uspto_mols))
    for smiles in pbar:
        tokens = tokenizer([smiles], padding="max_length", truncation=True, max_length=75, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        labels = input_ids.clone()
        # replace pad tokens with -100
        labels[labels == tokenizer.pad_token_id] = -100
        tokens["labels"] = labels
        with torch.no_grad():
            mol_outputs = decoder_model.molformer(input_ids, attention_mask=attention_mask)
            encoder_outputs = decoder_model.proj(mol_outputs.pooler_output)
            for q in [True, False]:
                if q:
                    encoder_outputs_q, *_ = model(encoder_outputs)
                else:
                    encoder_outputs_q = encoder_outputs
                encoder_outputs_q = encoder_outputs_q.unsqueeze(1)
                decoder_input_ids = _shift_right(input_ids, decoder_model.config.decoder_start_token_id,
                                                 decoder_model.config.pad_token_id)
                decoder_output = decoder_model.decoder(encoder_hidden_states=encoder_outputs_q,
                                                       input_ids=decoder_input_ids)
                lm_logits = decoder_model.lm_head(decoder_output.last_hidden_state)
                preds = lm_logits.argmax(-1)
                mask = labels != -100
                total_tokens = mask.sum()
                correct_tokens = ((preds == labels) & mask).sum()
                correct_tokens = correct_tokens / total_tokens
                pred_smiles = tokenizer.decode(preds[0], skip_special_tokens=True)
                if q:
                    token_accuracy.append(correct_tokens.item())
                    is_correct.append(pred_smiles == smiles)
                else:
                    token_accuracy_n.append(correct_tokens.item())
                    is_correct_not_q.append(pred_smiles == smiles)
            pbar.set_postfix({"correct": np.mean(is_correct), "token_accuracy": np.mean(token_accuracy),
                              "correct_not_q": np.mean(is_correct_not_q),
                              "token_accuracy_n": np.mean(token_accuracy_n)})


class VectorQuantizerDataset(torch.utils.data.Dataset):
    def __init__(self):
        step = 10_000_000
        self.np_files = []
        for i in range(0, 100_000_000, step):
            self.np_files.append(f"molformer_pubchem_{i}_{i + step}.npy")
        self.open_file_idx = 0
        self.open_file = np.load(self.np_files[0])

        self.samples_per_file = self.open_file.shape[0]

    def __len__(self):
        return len(self.np_files) * self.samples_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file
        if file_idx != self.open_file_idx:
            self.open_file = np.load(self.np_files[file_idx])
            self.open_file_idx = file_idx
        return torch.tensor(self.open_file[sample_idx]).float()


def get_model(input_dim, num_quantizers, codebook_size):
    model = ResidualVQ(
        dim=input_dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        learnable_codebook=True,  # Make codebook learnable
        ema_update=False,  # Use gradient descent instead of EMA
        kmeans_init=True,  # set to True
        kmeans_iters=100  # number of kmeans iterations to calculate the centroids for the codebook on init
    )
    return model


def get_data_loader(batch_size):
    dataset = VectorQuantizerDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main(num_quantizers, codebook_size, input_dim, batch_size, learning_rate, num_epochs):
    model = get_model(input_dim, num_quantizers, codebook_size)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data_loader = get_data_loader(batch_size)
    model.train()
    save_name_prefix = f"residual_vq_lr4_{num_quantizers}_{codebook_size}"
    best_loss = float("inf")
    for epoch in range(num_epochs):
        pbar = tqdm(data_loader)
        for x in pbar:
            optimizer.zero_grad()
            x = x.to(device)
            _, _, loss = model(x, return_all_codes=False)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item()}")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f"{save_name_prefix + '_best'}.pt")
        # torch.save(model.state_dict(), f"{save_name_prefix}_epoch_{epoch + 1}.pt")
        if epoch % 5 == 0:
            evaluate_with_decoder(model)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_quantizers", type=int, default=10)
    argparser.add_argument("--codebook_size", type=int, default=512)

    argparser.add_argument("--input_dim", type=int, default=768)  # Dimension of Molecular Transformer output
    argparser.add_argument("--batch_size_factor", type=int, default=100)
    argparser.add_argument("--learning_rate", type=float, default=1e-5)
    argparser.add_argument("--num_epochs", type=int, default=100)
    args = argparser.parse_args()
    bs = args.batch_size_factor * args.codebook_size
    bs = min(bs, 10_000 if args.codebook_size >= 512 else 50_000)
    print("Arguments:", args)

    main(args.num_quantizers, args.codebook_size, args.input_dim, bs, args.learning_rate, args.num_epochs)
