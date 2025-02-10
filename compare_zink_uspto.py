from train_decoder import SMILESDataset
from transformers import AutoTokenizer, AutoModel
import random
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bin_file_path = "ZINK_PROCESSED/smiles.bin"
indices_file_path = "ZINK_PROCESSED/indices.npy"
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
molfomer = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True, deterministic_eval=True)

zink_dataset = SMILESDataset(bin_file_path, indices_file_path, tokenizer)
zink_random_indexes = random.sample(range(len(zink_dataset)), 100)
zink_random_samples = [zink_dataset[i] for i in zink_random_indexes]
zink_vectors = [molfomer(**s) for s in zink_random_samples]
with open("USPTO/all_mols.txt", "r") as f:
    uspto_mols = f.read().splitlines()
uspoto_random_samples = random.sample(uspto_mols, 100)
uspoto_random_samples = [tokenizer(s, padding="max_length", truncation=True, max_length=75, return_tensors="pt") for s
                         in uspoto_random_samples]
uspto_vectors = [molfomer(**s) for s in uspoto_random_samples]


combined_vectors = zink_vectors + uspto_vectors
combined_vectors = torch.cat(combined_vectors).detach().cpu().numpy()
combined_vectors = combined_vectors.reshape(combined_vectors.shape[0], -1)
combined_vectors = TSNE(n_components=2).fit_transform(combined_vectors)
plt.scatter(combined_vectors[:100, 0], combined_vectors[:100, 1], label="Zink")
plt.scatter(combined_vectors[100:, 0], combined_vectors[100:, 1], label="USPTO")
plt.legend()
plt.show()

