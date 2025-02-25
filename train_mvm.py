# Molecule vector model
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedModel, AutoModel, T5Config, TrainingArguments, Trainer
import os
from torch.nn import functional as F
from transformers import PreTrainedModel
from train_decoder import create_model
from transformers.models.t5.modeling_t5 import T5Stack
import copy
import numpy as np
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_layers = {"xs": 1, "s": 2, "m": 4, "l": 6, "xl": 12}
n_heads = {"xs": 1, "s": 2, "m": 4, "l": 8, "xl": 12}
ff_dim = {"xs": 256, "s": 512, "m": 1024, "l": 2048, "xl": 4096}


def size_to_config(size):
    return T5Config(
        vocab_size=1,  # not used
        d_model=768,
        d_ff=ff_dim[size],
        num_layers=n_layers[size],
        num_decoder_layers=n_layers[size],
        num_heads=n_heads[size]
    )


def load_smiles_file(file_name):
    with open(file_name) as f:
        lines = f.read().splitlines()
    lines = [line.replace(" ", "").split(".") for line in lines]
    return lines


class ReactionMolsDataset(Dataset):
    def __init__(self, base_dir="USPTO", split="train", debug=False):
        self.max_len = 10
        self.src = load_smiles_file(f"{base_dir}/src-{split}.txt")
        self.tgt = load_smiles_file(f"{base_dir}/tgt-{split}.txt")
        self.src, self.tgt = zip(
            *[(s, t) for s, t in zip(self.src, self.tgt) if len(s) <= self.max_len and len(t) <= self.max_len])
        if debug:
            self.src = self.src[:2]
            self.tgt = self.tgt[:2]
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.empty = {"input_ids": torch.tensor([self.tokenizer.pad_token_id] * 75),
                      "attention_mask": torch.tensor([0] * 75)}

    def _tokenize_and_pad(self, smiles_list):
        tokens = [self.tokenizer(s, padding="max_length", truncation=True, max_length=75, return_tensors="pt") for s in
                  smiles_list if s]
        num_mols = len(tokens)
        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        attention_mask[:num_mols] = 1
        while len(tokens) < self.max_len:
            tokens.append({k: v.clone() for k, v in self.empty.items()})
        input_ids = torch.stack([t['input_ids'].squeeze(0) for t in tokens])
        token_attention_mask = torch.stack([t['attention_mask'].squeeze(0) for t in tokens])
        return {
            'input_ids': input_ids,  # Shape: (max_seq_len, 75)
            'token_attention_mask': token_attention_mask,  # Shape: (max_seq_len, 75)
            'mol_attention_mask': attention_mask,  # Shape: (max_seq_len,)
        }

    def __getitem__(self, idx):
        src, tgt = self.src[idx], self.tgt[idx]
        src_tokens = self._tokenize_and_pad(src)
        tgt_tokens = self._tokenize_and_pad(tgt)
        return {
            'src_input_ids': src_tokens['input_ids'],  # (max_seq_len, 75)
            'src_token_attention_mask': src_tokens['token_attention_mask'],  # (max_seq_len, 75)
            'src_mol_attention_mask': src_tokens['mol_attention_mask'],  # (max_seq_len,)
            'tgt_input_ids': tgt_tokens['input_ids'],  # (max_seq_len, 75)
            'tgt_token_attention_mask': tgt_tokens['token_attention_mask'],  # (max_seq_len, 75)
            'tgt_mol_attention_mask': tgt_tokens['mol_attention_mask'],  # (max_seq_len,)
        }

    def __len__(self):
        return len(self.src)


class MVM(PreTrainedModel):
    def __init__(self, config, alpha=0.5):
        super().__init__(config)
        self.config = config
        self.alpha = alpha
        self.encoder = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            deterministic_eval=True,
            trust_remote_code=True
        )
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder_model, _ = create_model()
        state_dict = torch.load("results_decoder/checkpoint-90000/pytorch_model.bin", map_location=torch.device('cpu'))
        self.decoder_model.load_state_dict(state_dict, strict=True)
        self.decoder_model.eval()
        for param in self.decoder_model.parameters():
            param.requires_grad = False
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.mvm_encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.mvm_decoder = T5Stack(decoder_config)

        self.decoder_start_embedding = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.pad_embedding = nn.Parameter(torch.randn(config.d_model))

    def get_mol_embeddings(self, input_ids, token_attention_mask, mol_attention_mask):
        """
        Process multiple molecules in parallel through MolFormer
        """
        batch_size, max_seq_len, seq_len = input_ids.shape

        flat_input_ids = input_ids.view(-1, seq_len)  # (batch_size * max_seq_len, 75)
        flat_attention_mask = token_attention_mask.view(-1, seq_len)  # (batch_size * max_seq_len, 75)
        flat_mol_attention_mask = mol_attention_mask.view(-1) == 1  # (batch_size * max_seq_len)
        flat_input_ids = flat_input_ids[flat_mol_attention_mask]
        flat_attention_mask = flat_attention_mask[flat_mol_attention_mask]

        chunk_size = 2048  # Adjust based on your GPU memory
        all_embeddings = []
        for i in range(0, flat_input_ids.size(0), chunk_size):
            chunk_input_ids = flat_input_ids[i:i + chunk_size]
            chunk_attention_mask = flat_attention_mask[i:i + chunk_size]
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask
                )
                all_embeddings.append(outputs.pooler_output)

        embeddings = torch.cat(all_embeddings, dim=0)  # (batch_size * max_seq_len, hidden_size)
        final_emb = self.pad_embedding.expand(flat_mol_attention_mask.size(0), -1).clone()
        index_counter = 0
        for i, m_values in enumerate(flat_mol_attention_mask):
            if m_values:
                final_emb[i] = embeddings[index_counter]
                index_counter += 1
        final_emb = final_emb.view(batch_size, max_seq_len, -1)
        return final_emb

    def shift_vectors_right(self, vectors):
        """
        Shift vectors to the right by one position
        """
        padding = self.decoder_start_embedding.expand(vectors.size(0), 1, -1)
        return torch.cat([padding, vectors[:, :-1]], dim=1)

    def forward(self, src_input_ids, src_token_attention_mask, src_mol_attention_mask,
                tgt_input_ids, tgt_token_attention_mask, tgt_mol_attention_mask):
        src_embeddings = self.get_mol_embeddings(src_input_ids, src_token_attention_mask, src_mol_attention_mask)
        tgt_embeddings = self.get_mol_embeddings(tgt_input_ids, tgt_token_attention_mask, tgt_mol_attention_mask)
        mvm_encoder_outputs = self.mvm_encoder(inputs_embeds=src_embeddings)
        mvm_encoder_outputs = mvm_encoder_outputs['last_hidden_state']
        shifted_tgt_embeddings = self.shift_vectors_right(tgt_embeddings)
        mvm_decoder_outputs = self.mvm_decoder(inputs_embeds=shifted_tgt_embeddings,
                                               encoder_hidden_states=mvm_encoder_outputs)
        mvm_decoder_outputs = mvm_decoder_outputs['last_hidden_state']
        # correctly use just the first sequence (single product)
        mvm_decoder_outputs = mvm_decoder_outputs[:, 0, :]

        gt_tgt_embeddings = self.get_mol_embeddings(tgt_input_ids, tgt_token_attention_mask, tgt_mol_attention_mask)
        gt_tgt_embeddings = gt_tgt_embeddings[:, 0, :]
        loss = F.mse_loss(mvm_decoder_outputs, gt_tgt_embeddings)

        tgt_input_ids_decoder = tgt_input_ids[:, 0].clone()
        labels = tgt_input_ids_decoder.clone()

        labels_mask = tgt_token_attention_mask[:, 0]
        labels[labels_mask == 0] = -100

        decoder_output = self.decoder_model(encoder_outputs=mvm_decoder_outputs, labels=labels,
                                            input_ids=tgt_input_ids_decoder)

        decoder_output.decoder_hidden_states = mvm_decoder_outputs
        decoder_output.loss = self.alpha * loss + (1 - self.alpha) * decoder_output.loss

        return decoder_output


def compute_metrics(eval_pred, debug=False):
    """
    Compute metrics for evaluation
    """
    predictions, labels_ = eval_pred
    tgt_input_ids, tgt_token_attention_mask, tgt_mol_attention_mask = labels_
    labels = tgt_input_ids[:, 0]
    labels_mask = tgt_token_attention_mask[:, 0]
    labels[labels_mask == 0] = -100
    # Get argmax of predictions
    predictions = np.argmax(predictions, axis=-1)
    if debug:
        print()
        print(predictions[0].tolist())
        print(labels[0].tolist())
    mask = labels != -100
    total_tokens = mask.sum()
    correct_tokens = ((predictions == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens
    if debug:
        print(f"Token accuracy: ({correct_tokens} / {total_tokens}) = {token_accuracy}")
    correct_or_pad = (predictions == labels) | (~mask)
    correct_samples = correct_or_pad.all(axis=-1).sum()
    total_samples = len(labels)
    sample_accuracy = correct_samples / total_samples

    return {
        "token_accuracy": token_accuracy,
        "sample_accuracy": sample_accuracy,
    }


def main(debug=False, batch_size=1024, num_epochs=10, lr=1e-4, size="m", alpha=0.5):
    train_dataset = ReactionMolsDataset(split="train", debug=debug)
    val_dataset = ReactionMolsDataset(split="valid", debug=debug)
    if debug:
        size = "xs"
    config = size_to_config(size)
    model = MVM(config, alpha)
    if debug:
        print(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    print(
        f"MODEL:MVM. Total parameters: {total_params:,}, (trainable: {trainable_params:,}, non-trainable: {non_trainable_params:,})")
    output_suf = f"{size}_{lr}_{alpha}"
    os.makedirs(f"results_mvm/{output_suf}", exist_ok=True)
    train_args = TrainingArguments(
        output_dir=f"results_mvm/{output_suf}",
        num_train_epochs=num_epochs if not debug else 10000,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=10,
        logging_dir=f"logs_mvm/{output_suf}",
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_accumulation_steps=1,
        report_to="none" if debug else "tensorboard",
        learning_rate=lr,
        lr_scheduler_type='constant',
        label_names=['tgt_input_ids', 'tgt_token_attention_mask', 'tgt_mol_attention_mask'],
        seed=42
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda x: compute_metrics(x, debug=debug)
    )
    # score = trainer.evaluate()
    # print(score)
    resume_from_checkpoint = len(glob.glob(f"results_mvm/{output_suf}/checkpoint*")) > 0
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


# run main to test dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--size", type=str, default="s", choices=['xs', "s", "m", "l", "xl"])
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()
    main(args.debug, args.batch_size, args.num_epochs, args.lr, args.size, args.alpha)
