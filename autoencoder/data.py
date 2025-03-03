import os

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, processors
from collections import Counter
from pathlib import Path
import re
from rdkit import Chem
from tokenizers.pre_tokenizers import Whitespace
from rdkit import RDLogger
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

RDLogger.DisableLog('rdApp.*')

tokenizer_file = "pubchem-canonical/tokenizer/"



def preprocess_smiles(smiles: str) -> str:
    # Remove stereochemistry and canonicalize SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def smiles_to_tokens(smiles: str) -> list:
    SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    SMILES_REGEX = re.compile(SMILES_TOKENIZER_PATTERN)

    # canonicalize SMILES and remove stereochemistry
    smiles = preprocess_smiles(smiles)
    tokens = [token for token in SMILES_REGEX.findall(smiles)]
    return tokens
def process_line(line):
    smiles = line.strip().split()[1]
    tokens = smiles_to_tokens(smiles)
    return tokens

def get_tokenizer(input_file="pubchem-canonical/CID-SMILES-CANONICAL.smi"):
    if Path(tokenizer_file).exists():
        print(f"Loading existing tokenizer from {tokenizer_file}")
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_file)

    tokens_set = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    cpu_count = os.cpu_count()
    num_workers = min(cpu_count, 32)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for tokens in tqdm(executor.map(process_line, lines), total=len(lines), desc="Processing SMILES", unit="line"):
            tokens_set.update(set(tokens))
        # with tqdm(total=len(lines), desc="Processing SMILES", unit="line") as pbar:
        #     # Submit tasks in parallel
        #     futures = [executor.submit(process_line, line) for line in lines]
        #
        #     # Process results as they complete
        #     for future in futures:
        #         tokens = future.result()
        #         tokens_set.update(set(tokens))
        #         pbar.update(1)  # Update tqdm manually
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    idx = len(vocab)
    for token in tokens_set:
        vocab[token] = idx
        idx += 1
    print(f"Vocabulary size: {len(vocab)}")

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))

    tokenizer.pre_tokenizer = Whitespace()
    # add special tokens

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", vocab["<bos>"]),
            ("<eos>", vocab["<eos>"])
        ]
    )

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>"
    )

    wrapped_tokenizer.save_pretrained(tokenizer_file)
    return wrapped_tokenizer


class AutoEncoderDataset(Dataset):
    def __init__(self, input_file="pubchem-canonical/CID-SMILES-CANONICAL.smi", max_length=75):
        self.tokenizer = get_tokenizer(input_file)
        self.max_length = max_length
        self.data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                _, smiles = line.strip().split()
                tokens = smiles_to_tokens(smiles)
                if len(tokens) > max_length:
                    continue
                self.data.append(" ".join(tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data[idx]
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = source_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


# main:
if __name__ == "__main__":
    dataset = AutoEncoderDataset()

    print(dataset[0])
