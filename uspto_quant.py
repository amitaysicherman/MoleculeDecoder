import os
from tqdm import tqdm
from vector_quantize_pytorch import ResidualVQ
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Tuple


def count_molecules(line: str) -> int:
    return len(line.replace(" ", "").strip().split("."))


def create_batch(lines: List[str], target_batch_size: int = 1024) -> Tuple[List[str], List[int], int]:
    current_batch = []
    molecules_per_line = []
    total_molecules = 0

    for line in lines:
        num_mols = count_molecules(line)
        if total_molecules + num_mols > target_batch_size and current_batch:
            break
        current_batch.append(line)
        molecules_per_line.append(num_mols)
        total_molecules += num_mols

    return current_batch, molecules_per_line, len(current_batch)


def process_batch(lines: List[str], tokenizer, molformer, q_model, device) -> torch.Tensor:
    # Flatten all SMILES in the batch
    flat_smiles = [smile for line in lines for smile in line.replace(" ", "").strip().split(".")]

    # Process the entire batch at once
    input_tokens = tokenizer(flat_smiles, padding="max_length", truncation=True,
                             return_tensors="pt", max_length=75)
    input_ids = input_tokens["input_ids"].to(device)
    attention_mask = input_tokens["attention_mask"].to(device)

    with torch.no_grad():
        mol_outputs = molformer(input_ids, attention_mask=attention_mask)
        embeddings = mol_outputs.pooler_output
        quantized = q_model(embeddings)

    quantized_indexes = quantized[1]
    return quantized_indexes.view(quantized_indexes.size(0), -1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading models...")
    molformer = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct",
                                          trust_remote_code=True,
                                          deterministic_eval=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct",
                                              trust_remote_code=True)

    q_cp = "results_pubchem/residual_vq_lr_64_512_768_10000_0.001_100_best.pt"
    _, _, _, num_quantizers, codebook_size, input_dim, _, _, _, _ = q_cp.split("/")[-1].split("_")
    q_model = ResidualVQ(
        dim=int(input_dim),
        num_quantizers=int(num_quantizers),
        codebook_size=int(codebook_size),
        ema_update=False,
    ).eval().to(device)
    q_model.load_state_dict(torch.load(q_cp, map_location=torch.device('cpu')))

    # Process files
    output_dir = "USPTO_Q"
    os.makedirs(output_dir, exist_ok=True)
    input_dir = "USPTO"
    target_batch_size = 1024

    for type_ in ["src", "tgt"]:
        for split in ["train", "valid", "test"]:
            input_file = f"{input_dir}/{type_}-{split}.txt"
            output_file = f"{output_dir}/{type_}-{split}.txt"

            print(f"\nProcessing {input_file}...")

            # Read all lines
            with open(input_file) as f:
                lines = f.readlines()

            # Process with dynamic batching
            with open(output_file, "w") as f:
                processed_lines = 0
                pbar = tqdm(total=len(lines))

                while processed_lines < len(lines):
                    remaining_lines = lines[processed_lines:]

                    # Create batch based on molecule count
                    batch, molecules_per_line, num_lines = create_batch(
                        remaining_lines, target_batch_size)

                    # Process the batch
                    quantized_indexes = process_batch(
                        batch, tokenizer, molformer, q_model, device)

                    # Split back into individual lines
                    current_idx = 0
                    for num_mols in molecules_per_line:
                        line_indexes = quantized_indexes[current_idx:current_idx + num_mols]
                        current_idx += num_mols

                        # Convert to string and write
                        indexes_str = " ".join(map(str, line_indexes.cpu().numpy().flatten()))
                        f.write(indexes_str + "\n")

                    processed_lines += num_lines
                    pbar.update(num_lines)

                pbar.close()

            print(f"Completed {input_file}")


if __name__ == "__main__":
    main()