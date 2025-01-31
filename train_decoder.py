import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoModel, T5Config, AutoTokenizer, GenerationConfig, \
    T5PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import functional as F
import mmap


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

        # Calculate total size
        self.total_size = len(self.indices)

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
        start_idx = self.indices[idx]

        # Get end index (either next index or end of file)
        if idx + 1 < self.total_size:
            end_idx = self.indices[idx + 1]
        else:
            end_idx = len(self.mm)

        # Read and decode the SMILES string
        smile = self.mm[start_idx:end_idx].decode('utf-8')
        tokens = self.tokenizer(smile, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        print(tokens)
        labels = tokens["input_ids"].clone()
        labels = _shift_right(labels, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id)
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
    predict_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    with open("predictions.txt", "a") as f:
        f.write("\n".join(predict_text))

    # Create mask for padding tokens
    mask = labels != -100

    # Calculate token accuracy (ignoring padding)
    total_tokens = mask.sum()
    correct_tokens = ((predictions == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens

    # Calculate sequence accuracy (all tokens correct)
    sequence_matches = [(pred == label).all() for pred, label in
                        zip(predictions[mask.reshape(predictions.shape[0], -1)],
                            labels[mask.reshape(labels.shape[0], -1)])]
    sequence_accuracy = np.mean(sequence_matches)
    with open("metrics.txt", "a") as f:
        f.write(f"Token Accuracy: {token_accuracy}, Sequence Accuracy: {sequence_accuracy}\n")
    return {
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy
    }


# def main():
#     # Load MolFormer tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
#
#     # Initialize model config (matching roughly 50M params)
#     config = T5Config(
#         vocab_size=len(tokenizer),
#         d_model=768,  # Same as MolFormer
#         d_ff=2048,
#         num_layers=6,  # Reduced from MolFormer's 12 to match param count
#         num_decoder_layers=6,
#         num_heads=8,
#         is_decoder=True,
#         is_encoder_decoder=False,  # We're only using the decoder
#         decoder_start_token_id=tokenizer.pad_token_id,
#     )
#
#     # Initialize model using T5ForConditionalGeneration
#     model = T5ForConditionalGeneration(config)
#
#     # Setup tensorboard
#     # writer = SummaryWriter('runs/molecule_decoder')
#
#     # Create full dataset
#     full_dataset = SMILESDataset(
#         bin_file_path="ZINK_PROCESSED/smiles.bin",
#         indices_file_path="ZINK_PROCESSED/indices.npy"
#     )
#
#     # Split dataset into train and evaluation
#     train_size = int(0.9 * len(full_dataset))
#     eval_size = len(full_dataset) - train_size
#     train_dataset, eval_dataset = random_split(
#         full_dataset, [train_size, eval_size]
#     )
#
#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir="./results",
#         num_train_epochs=10,
#         per_device_train_batch_size=2,
#         per_device_eval_batch_size=2,
#         learning_rate=1e-4,  # Constant learning rate
#         logging_dir='./logs',
#         logging_steps=100,
#         save_steps=1000,
#         eval_steps=500,  # Evaluate every 500 steps
#         evaluation_strategy="steps",
#         report_to=["tensorboard"],
#         lr_scheduler_type="constant",  # Use constant learning rate
#         load_best_model_at_end=True,
#         metric_for_best_model="token_accuracy",
#     )
#
#     # Initialize trainer with evaluation
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         compute_metrics=compute_metrics,
#     )
#
#     # Train model
#     trainer.train()
#
#     # Final evaluation
#     final_metrics = trainer.evaluate()
#     print("Final Evaluation Metrics:", final_metrics)
#
#     # Save final model
#     trainer.save_model("./molecule_decoder_final")
#     # writer.close()
#

class MolFormerT5Decoder(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Load and freeze MolFormer
        self.molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True
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

    def forward(self, input_ids, attention_mask=None):
        # Get MolFormer embedding
        mol_outputs = self.molformer(input_ids, attention_mask=attention_mask)
        encoder_outputs = self.proj(mol_outputs.pooler_output).unsqueeze(1)
        # Run through decoder
        decoder_input_ids = self._shift_right(input_ids)
        decoder_output = self.decoder(encoder_hidden_states=encoder_outputs, input_ids=decoder_input_ids)
        lm_logits = self.lm_head(decoder_output.last_hidden_state)
        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), input_ids.view(-1))
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
        d_model=8,  # 768,
        d_ff=16,  # 2048,
        num_layers=1,  # 6,
        num_decoder_layers=1,  # 6,
        is_encoder_decoder=True,
        is_decoder=True,
        num_heads=1,  # 8,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    model = MolFormerT5Decoder(config)
    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model, tokenizer


# Example usage
if __name__ == "__main__":
    model, tokenizer = create_model()

    # Load the dataset
    bin_file_path = "ZINK_PROCESSED/smiles.bin"
    indices_file_path = "ZINK_PROCESSED/indices.npy"
    dataset = SMILESDataset(bin_file_path, indices_file_path, tokenizer)
    train_size = len(dataset) - 10
    eval_size = 10
    train_dataset, eval_dataset = random_split(
        dataset, [train_size, eval_size]
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-4,  # Constant learning rate
        logging_dir='./logs',
        logging_steps=100,
        save_steps=1000,
        eval_steps=50,  # Evaluate every 500 steps
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

    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the trained model
    model.save_pretrained("path/to/save/model")
