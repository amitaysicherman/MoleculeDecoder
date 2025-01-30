import os
from tqdm import tqdm
import numpy as np

# Create output directory
os.makedirs("ZINK_PROCESSED", exist_ok=True)

# Initialize variables
chunks = os.listdir("ZINK")
all_smiles = bytearray()  # Use bytearray for efficient concatenation
indices = []  # List to store starting indices of each SMILE
current_index = 0

# Process all chunks
pbar = tqdm(chunks)
total_smiles = 0

for chunk in pbar:
    # Read SMILES from file
    with open(f"ZINK/{chunk}") as f:
        smiles_ids = f.read().splitlines()

    # Extract just the SMILES strings (first column)
    smiles = [smiles_id.split()[0] for smiles_id in smiles_ids]
    total_smiles += len(smiles)

    # Process each SMILE
    for smile in smiles:
        # Store the starting index
        indices.append(current_index)

        # Convert SMILE to bytes and add to main bytearray
        smile_bytes = smile.encode('utf-8')
        all_smiles.extend(smile_bytes)

        # Update current index
        current_index += len(smile_bytes)

    pbar.set_description(f"Total SMILES: {total_smiles}")

# Convert indices to numpy array with smallest possible type
max_index = max(indices)
if max_index < np.iinfo(np.uint32).max:
    dtype = np.uint32
elif max_index < np.iinfo(np.uint64).max:
    dtype = np.uint64
else:
    raise ValueError("Indices too large for uint64")

indices_array = np.array(indices, dtype=dtype)

# Save the binary data and indices
with open("ZINK_PROCESSED/smiles.bin", "wb") as f:
    f.write(all_smiles)
np.save("ZINK_PROCESSED/indices.npy", indices_array)

print(f"Processed {total_smiles} SMILES")
print(f"Total binary size: {len(all_smiles) / (1024 * 1024):.2f} MB")
print(f"Indices array size: {indices_array.nbytes / (1024 * 1024):.2f} MB")
print(f"Using dtype: {dtype}")


# Optional: Verification code
def verify_data():
    print("\nVerifying data...")
    # Load the saved data
    with open("ZINK_PROCESSED/smiles.bin", "rb") as f:
        loaded_smiles = f.read()
    loaded_indices = np.load("ZINK_PROCESSED/indices.npy")

    # Check a few random SMILES
    for i in range(min(5, len(loaded_indices))):
        start = loaded_indices[i]
        end = loaded_indices[i + 1] if i + 1 < len(loaded_indices) else len(loaded_smiles)
        smile = loaded_smiles[start:end].decode('utf-8')
        print(f"Sample SMILE {i}: {smile}")


verify_data()