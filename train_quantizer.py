import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vector_quantize_pytorch import ResidualVQ  # Import the ResidualVQ model
import argparse
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    )
    return model


def get_data_loader(batch_size):
    dataset = VectorQuantizerDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main(num_quantizers, codebook_size, input_dim, batch_size, batch_size_factor, learning_rate, num_epochs):
    model = get_model(input_dim, num_quantizers, codebook_size)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data_loader=get_data_loader(batch_size)
    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(data_loader)
        for x in pbar:
            optimizer.zero_grad()
            x=x.to(device)
            _, _, loss = model(x, return_all_codes=False)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item()}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    torch.save(model.state_dict(), "residual_vq.pth")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_quantizers", type=int, default=512)
    argparser.add_argument("--codebook_size", type=int, default=10)

    argparser.add_argument("--input_dim", type=int, default=768)  # Dimension of Molecular Transformer output
    # argparser.add_argument("--batch_size", type=int, default=65536)
    argparser.add_argument("--batch_size_factor", type=int, default=100)
    argparser.add_argument("--learning_rate", type=float, default=1e-4)
    argparser.add_argument("--num_epochs", type=int, default=1)
    args = argparser.parse_args()
    bs = args.batch_size_factor * args.codebook_size
    main(args.num_quantizers, args.codebook_size, args.input_dim, args.batch_size_factor * args.codebook_size,
         bs, args.learning_rate, args.num_epochs)
