from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, processors
from pathlib import Path
import re
from rdkit import Chem
from tokenizers.pre_tokenizers import Whitespace
from rdkit import RDLogger
from tqdm import tqdm
import random

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


from autoencoder.tokenizer import ChemicalTokenizer


def get_tokenizer():
    return ChemicalTokenizer()


def line_to_tokens(line):
    _, smiles = line.strip().split()
    tokens = smiles_to_tokens(smiles)
    return tokens


class AutoEncoderDataset(Dataset):
    def __init__(self, input_file="pubchem-canonical/CID-SMILES-CANONICAL.smi", max_length=75):
        self.tokenizer = get_tokenizer()
        self.max_length = max_length
        with open(input_file, 'r', encoding='utf-8') as f:
            self.data = f.read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        tokens = line_to_tokens(line)
        while len(tokens) > self.max_length:
            idx = random.randint(0, len(self.data) - 1)
            line = self.data[idx]
            tokens = line_to_tokens(line)

        source = " ".join(tokens)

        source_encoding = self.tokenizer.encode(source, max_length=self.max_length)
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

    for _ in dataset:
        pass
