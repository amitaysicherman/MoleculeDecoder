import os
from tqdm import tqdm
from vector_quantize_pytorch import ResidualVQ
from transformers import AutoTokenizer, AutoModel
import torch


def get_molformer_embeddings(smiles):
    input_tokens = tokenizer(smiles, padding="max_length", truncation=True, return_tensors="pt", max_length=75)
    input_ids = input_tokens["input_ids"].to(device)
    attention_mask = input_tokens["attention_mask"].to(device)
    with torch.no_grad():
        mol_outputs = molformer(input_ids, attention_mask=attention_mask)
    return mol_outputs.pooler_output.detach()


def get_quantized_embeddings_indexes(embeddings):
    with torch.no_grad():
        quantized = q_model(embeddings)
    quantized_indexes = quantized[1]
    quantized_indexes = quantized_indexes.view(quantized_indexes.size(0), -1)
    return quantized_indexes


def get_line_indexes(line):
    smiles = line.replace(" ", "").strip().split(".")
    embeddings = get_molformer_embeddings(smiles)
    quantized_indexes = get_quantized_embeddings_indexes(embeddings)
    return quantized_indexes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

molformer = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True,
                                      deterministic_eval=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
q_cp = "results_pubchem/residual_vq_lr_64_512_768_10000_0.001_100_best.pt"
_, _, _, num_quantizers, codebook_size, input_dim, _, _, _, _ = q_cp.split("/")[-1].split("_")
q_model = ResidualVQ(
    dim=int(input_dim),
    num_quantizers=int(num_quantizers),
    codebook_size=int(codebook_size),
    ema_update=False,  # Use gradient descent instead of EMA
).eval().to(device)
q_model.load_state_dict(torch.load(q_cp, map_location=torch.device('cpu')))

output_dir = "USPTO_Q"
os.makedirs(output_dir, exist_ok=True)
input_dir = "USPTO"
for type_ in ["src", "tgt"]:
    for split in ["train", "valid", "test"]:
        with open(f"{input_dir}/{type_}-{split}.txt") as f:
            lines = f.readlines()
        with open(f"{output_dir}/{type_}-{split}.txt", "w") as f:
            for line in tqdm(lines):
                quantized_indexes = get_line_indexes(line)
                quantized_indexes = sum(quantized_indexes.cpu().numpy().tolist(), [])
                quantized_indexes_str = " ".join([str(i) for i in quantized_indexes])
                f.write(quantized_indexes_str + "\n")
