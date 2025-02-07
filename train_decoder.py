import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoModel, T5Config, AutoTokenizer, GenerationConfig, \
    T5PreTrainedModel, PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import functional as F
import mmap
from rdkit import Chem


def _shift_right(input_ids, decoder_start_token_id, pad_token_id):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class SMILESDataset(Dataset):
    def __init__(self, bin_file_path, indices_file_path, tokenizer):
        """
        Initialize the dataset.
        Args:
            bin_file_path (str): Path to the binary file containing SMILES strings
            indices_file_path (str): Path to the .npy file containing indices
        """
        # Load indices
        self.indices = np.load(indices_file_path)
        self.tokenizer = tokenizer
        # Memory map the binary file for efficient access
        self.bin_file = open(bin_file_path, 'rb')
        self.mm = mmap.mmap(self.bin_file.fileno(), 0, access=mmap.ACCESS_READ)

        with open("USPTO/all_mols.txt") as f:
            self.all_uspto_mols = f.read().splitlines()

        # Calculate total size
        self.uspto_size = len(self.all_uspto_mols)
        self.zink_size = len(self.indices)
        self.total_size = self.uspto_size + self.zink_size

    def remove_stereochemistry(self, smiles):
        """
        Remove stereochemistry using RDKit.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        """
        Get a SMILES string at the given index.
        Args:
            idx (int): Index of the SMILES string to retrieve
        Returns:
            str: The SMILES string
        """
        # Get start index

        if idx >= self.zink_size:
            print("USPTO")
            smile = self.all_uspto_mols[idx - self.zink_size]

        else:
            start_idx = self.indices[idx]

            # Get end index (either next index or end of file)
            if idx + 1 < self.total_size:
                end_idx = self.indices[idx + 1]
            else:
                end_idx = len(self.mm)

            # Read and decode the SMILES string
            smile = self.mm[start_idx:end_idx].decode('utf-8')
        smile = self.remove_stereochemistry(smile)
        tokens = self.tokenizer(smile, padding="max_length", truncation=True, max_length=75, return_tensors="pt")
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        labels = tokens["input_ids"].clone()
        # replace pad tokens with -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        tokens["labels"] = labels

        return tokens

    def __del__(self):
        """Cleanup when the dataset is destroyed"""
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'bin_file'):
            self.bin_file.close()


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation
    """
    predictions, labels = eval_pred
    # Get argmax of predictions
    predictions = np.argmax(predictions, axis=-1)
    mask = labels != -100

    total_tokens = mask.sum()
    correct_tokens = ((predictions == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens

    return {
        "token_accuracy": token_accuracy,
    }


class MolFormerT5Decoder(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Load and freeze MolFormer
        self.molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True,
            deterministic_eval=True
        )
        self.config = config
        for param in self.molformer.parameters():
            param.requires_grad = False
        # Simple projection if needed
        if self.molformer.config.hidden_size != config.d_model:
            self.proj = nn.Linear(self.molformer.config.hidden_size, config.d_model)
        else:
            self.proj = nn.Identity()
        # Initialize T5 decoder
        T5 = T5ForConditionalGeneration(config)
        self.decoder = T5.get_decoder()
        self.lm_head = T5.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get MolFormer embedding
        mol_outputs = self.molformer(input_ids, attention_mask=attention_mask)
        encoder_outputs = self.proj(mol_outputs.pooler_output).unsqueeze(1)
        # Run through decoder
        decoder_input_ids = _shift_right(input_ids, self.config.decoder_start_token_id, self.config.pad_token_id)
        decoder_output = self.decoder(encoder_hidden_states=encoder_outputs, input_ids=decoder_input_ids)
        lm_logits = self.lm_head(decoder_output.last_hidden_state)

        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), ignore_index=-100)
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
        )


def create_model():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    # Initialize config
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=768,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=6,
        is_encoder_decoder=True,
        is_decoder=True,
        num_heads=8,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    model = MolFormerT5Decoder(config)
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # number of non trainable parameters
    print(f"Number of non-trainable parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(model)
    return model, tokenizer


# Example usage
if __name__ == "__main__":
    model, tokenizer = create_model()

    # Load the dataset
    bin_file_path = "ZINK_PROCESSED/smiles.bin"
    indices_file_path = "ZINK_PROCESSED/indices.npy"
    dataset = SMILESDataset(bin_file_path, indices_file_path, tokenizer)
    train_size = len(dataset) - 100_000
    eval_size = 100_000
    train_dataset, eval_dataset = random_split(
        dataset, [train_size, eval_size]
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        learning_rate=1e-4,  # Constant learning rate
        logging_dir='./logs',
        logging_steps=1_000,
        save_steps=5_000,
        eval_accumulation_steps=2,
        eval_steps=5_000,  # Evaluate every 500 steps
        evaluation_strategy="steps",
        report_to=["tensorboard"],
        lr_scheduler_type="constant",  # Use constant learning rate
        load_best_model_at_end=True,
        metric_for_best_model="token_accuracy",
        save_safetensors=False,
        label_names=["labels"],

    )

    # Initialize trainer with evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # eval_results = trainer.evaluate()
    # print("Evaluation results:", eval_results)
    trainer.train(resume_from_checkpoint=False)

    # Evaluate the model
    trainer.evaluate()

    # Save the trained model
    model.save_pretrained("path/to/save/model")
