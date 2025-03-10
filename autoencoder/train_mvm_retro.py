import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
import os
from torch.nn import functional as F
import numpy as np
from transformers import BertConfig, BertModel
import random
from autoencoder.data import smiles_to_tokens, get_tokenizer, preprocess_smiles
from transformers import AutoTokenizer
import glob
from transformers import BertGenerationDecoder, BertGenerationConfig, BertGenerationEncoder
from transformers import AutoModel
from train_decoder import create_model

# hidden_sizes = {'s': 128, 'm': 512, 'l': 512}
num_layers = {'s': 2, 'm': 6, 'l': 24}
num_heads = {'s': 2, 'm': 4, 'l': 8}

# if mps use mps, else if gpu use gpu, else use cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def size_to_configs(size, hidden_size, tokenizer):
    size_args = dict(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers[size],
        num_attention_heads=num_heads[size],
        intermediate_size=hidden_size * 4,
    )
    common_args = dict(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        is_encoder_decoder=True,
    )
    encoder_config = BertGenerationConfig(**common_args, **size_args, is_decoder=False)
    decoder_config = BertGenerationConfig(**common_args, **size_args, is_decoder=True, add_cross_attention=True)
    return encoder_config, decoder_config


class ReactionMolsDataset(Dataset):
    def __init__(self, base_dir="USPTO", split="train", max_mol_len=75, max_len=10, skip_unk=True):
        self.max_len = max_len
        self.max_mol_len = max_mol_len
        self.skip_unk = skip_unk
        with open(f"{base_dir}/reactants-{split}.txt") as f:
            self.reactants = f.read().splitlines()
        with open(f"{base_dir}/products-{split}.txt") as f:
            self.products = f.read().splitlines()
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.empty = {"input_ids": torch.tensor([self.tokenizer.pad_token_id] * 75),
                      "attention_mask": torch.tensor([0] * 75)}

    def preprocess_line(self, line):
        if line == "":
            return {
                'input_ids': torch.zeros(self.max_len, self.max_mol_len,
                                         dtype=torch.long) + self.tokenizer.pad_token_id,
                'token_attention_mask': torch.zeros(self.max_len, self.max_mol_len, dtype=torch.long),
                'mol_attention_mask': torch.zeros(self.max_len, dtype=torch.long)
            }
        mols = line.strip().split(".")

        if len(mols) > self.max_len:
            return None
        mols = [preprocess_smiles(m) for m in mols]
        if None in mols:
            return None
        tokens = [self.tokenizer(m, padding="max_length", truncation=True, max_length=self.max_mol_len,
                                 return_tensors="pt") for m in mols]
        tokens = [{k: v.squeeze(0) for k, v in t.items()} for t in tokens]
        for t in tokens:
            if t['attention_mask'][-1] != 0:
                return None

        if self.skip_unk:
            if any([self.tokenizer.unk_token_id in t['input_ids'] for t in tokens]):
                print("Skipping UNK")
                return None
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
        reactants, products = self.reactants[idx], self.products[idx]
        products, reactants = reactants, products  # Swap reactants and products for retrosynthesis
        continue_run = True
        while continue_run:
            reactants_tokens = self.preprocess_line(reactants)
            products_tokens = self.preprocess_line(products)
            if reactants_tokens is not None and products_tokens is not None:
                continue_run = False
            else:
                idx = random.randint(0, len(self.reactants) - 1)
                reactants, products = self.reactants[idx], self.products[idx]

        return {
            'reactants_input_ids': reactants_tokens['input_ids'],
            'reactants_token_attention_mask': reactants_tokens['token_attention_mask'],
            'reactants_mol_attention_mask': reactants_tokens['mol_attention_mask'],
            'products_input_ids': products_tokens['input_ids'],
            'products_token_attention_mask': products_tokens['token_attention_mask'],
            'products_mol_attention_mask': products_tokens['mol_attention_mask'],
        }

    def __len__(self):
        return len(self.reactants)


class MVM(nn.Module):
    def __init__(self, config_enc, config_dec, encoder=None, decoder=None, is_trainable_encoder=False):
        super().__init__()
        self.config_enc = config_enc
        self.config_dec = config_dec
        self.encoder = encoder
        self.is_trainable_encoder = is_trainable_encoder
        self.decoder = decoder
        self.bert_encoder = BertGenerationEncoder(config_enc)
        self.bert_decoder = BertGenerationDecoder(config_dec)

    def get_mol_embeddings(self, input_ids, token_attention_mask, mol_attention_mask):
        batch_size, max_seq_len, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(-1, seq_len)  # (batch_size * max_seq_len, 75)
        flat_attention_mask = token_attention_mask.view(-1, seq_len)  # (batch_size * max_seq_len, 75)
        flat_mol_attention_mask = mol_attention_mask.view(-1) == 1  # (batch_size * max_seq_len)
        flat_input_ids = flat_input_ids[flat_mol_attention_mask]
        flat_attention_mask = flat_attention_mask[flat_mol_attention_mask]
        chunk_size = 2048 if not self.is_trainable_encoder else 256
        all_embeddings = []
        with torch.set_grad_enabled(self.is_trainable_encoder):
            for i in range(0, flat_input_ids.size(0), chunk_size):
                all_embeddings.append(self.encoder(
                    input_ids=flat_input_ids[i:i + chunk_size],
                    attention_mask=flat_attention_mask[i:i + chunk_size]
                ).pooler_output)
        embeddings = torch.cat(all_embeddings, dim=0)  # (batch_size * max_seq_len, hidden_size)
        final_emb = torch.zeros(flat_mol_attention_mask.size(0), embeddings.size(-1), device=embeddings.device)
        final_emb[flat_mol_attention_mask.nonzero(as_tuple=True)[0]] = embeddings
        final_emb = final_emb.view(batch_size, max_seq_len, -1)
        return final_emb

    def forward(self, reactants_input_ids, reactants_token_attention_mask, reactants_mol_attention_mask,
                products_input_ids, products_token_attention_mask, products_mol_attention_mask):
        reactants_embeddings = self.get_mol_embeddings(reactants_input_ids, reactants_token_attention_mask,
                                                       reactants_mol_attention_mask)

        src_embeddings = reactants_embeddings
        src_mol_attention_mask = reactants_mol_attention_mask
        src_seq_mask = src_mol_attention_mask.float()

        bert_encoder_output = self.bert_encoder(inputs_embeds=src_embeddings, attention_mask=src_seq_mask)[
            'last_hidden_state']

        product_embeddings = self.get_mol_embeddings(products_input_ids, products_token_attention_mask,
                                                     products_mol_attention_mask)

        bert_decoder_output = self.bert_decoder(inputs_embeds=product_embeddings,
                                                attention_mask=products_mol_attention_mask,
                                                encoder_hidden_states=bert_encoder_output, output_hidden_states=True)
        bert_decoder_output = bert_decoder_output['hidden_states'][-1]

        bs, seq_len, _ = bert_decoder_output.size()
        bert_decoder_output_flattened = bert_decoder_output.view(bs * seq_len, -1)
        products_input_ids_flattened = products_input_ids.view(bs * seq_len, -1)
        products_token_attention_mask_flattened = products_token_attention_mask.view(bs * seq_len, -1)
        products_mol_attention_mask_flattened = products_mol_attention_mask.view(bs * seq_len)

        bert_decoder_output_flattened = bert_decoder_output_flattened[products_mol_attention_mask_flattened == 1]
        products_input_ids_flattened = products_input_ids_flattened[products_mol_attention_mask_flattened == 1]
        products_token_attention_mask_flattened = products_token_attention_mask_flattened[
            products_mol_attention_mask_flattened == 1]
        products_labels_flattened = products_input_ids_flattened.clone()
        products_labels_flattened[products_token_attention_mask_flattened == 0] = -100
        chunk_size = 2048 if not self.is_trainable_encoder else 256
        all_loss = []
        all_logits = []
        with torch.set_grad_enabled(self.is_trainable_encoder):
            for i in range(0, products_input_ids_flattened.size(0), chunk_size):
                output = self.decoder(
                    input_ids=products_input_ids_flattened[i:i + chunk_size],
                    attention_mask=products_token_attention_mask_flattened[i:i + chunk_size],
                    encoder_outputs=bert_decoder_output_flattened[i:i + chunk_size],
                    labels=products_labels_flattened[i:i + chunk_size]
                )
                all_loss.append(output.loss)
                all_logits.append(output.logits)
        loss = torch.stack(all_loss).mean()
        logits = torch.cat(all_logits, dim=0)
        final_logits = torch.zeros(products_mol_attention_mask_flattened.size(0), products_labels_flattened.size(-1),
                                   logits.size(-1), device=logits.device)
        final_logits[products_mol_attention_mask_flattened.nonzero(as_tuple=True)[0]] = logits
        final_logits = final_logits.view(bs, seq_len, -1)
        return {"loss": loss, "logits": final_logits}


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation
    """
    predictions, labels_ = eval_pred
    tgt_input_ids, tgt_token_attention_mask, tgt_mol_attention_mask = labels_
    labels = tgt_input_ids  # Shape: (batch_size, max_seq_len, 75)

    labels_mask = tgt_token_attention_mask
    labels[labels_mask == 0] = -100
    labels_mask = labels_mask.astype(bool)
    predictions = np.argmax(predictions, axis=-1)  # Shape: (batch_size, max_seq_len, 75)
    total_tokens = 0
    total_samples = 0
    correct_tokens = 0
    correct_samples = 0
    batch_size, max_seq_len, seq_len = labels.shape
    for i in range(batch_size):
        for j in range(max_seq_len):
            total_tokens += labels_mask[i, j].sum()
            correct_tokens += ((predictions[i, j] == labels[i, j]) & labels_mask[i, j]).sum()
            if labels_mask[i, j].sum() == 0:
                continue
            total_samples += 1
            correct_samples += ((predictions[i, j] == labels[i, j]) | (~labels_mask[i, j])).all()

    token_accuracy = correct_tokens / total_tokens
    sample_accuracy = correct_samples / total_samples
    return {
        "token_accuracy": token_accuracy,
        "sample_accuracy": sample_accuracy,
    }


def get_last_cp(base_dir):
    if not os.path.exists(base_dir):
        return None
    all_checkpoints = glob.glob(f"{base_dir}/checkpoint-*")
    if not all_checkpoints:
        return None
    cp_steps = [int(cp.split("-")[-1]) for cp in all_checkpoints]
    last_cp = max(cp_steps)
    return f"{base_dir}/checkpoint-{last_cp}"


def main(batch_size=32, num_epochs=10, lr=1e-4, size="m", train_encoder=False,
         train_decoder=False, cp=None):
    train_dataset = ReactionMolsDataset(split="train")
    val_dataset = ReactionMolsDataset(split="valid")
    train_subset_random_indices = random.sample(range(len(train_dataset)), len(val_dataset))
    train_subset = torch.utils.data.Subset(train_dataset, train_subset_random_indices)
    encoder_config, decoder_config = size_to_configs(size, 768, train_dataset.tokenizer)
    # Load encoder and decoder

    encoder = AutoModel.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct",
        deterministic_eval=True,
        trust_remote_code=True,
        use_safetensors=False  # Force using PyTorch format instead of safetensors
    )

    decoder, _ = create_model()
    state_dict = torch.load("results_decoder/checkpoint-195000/pytorch_model.bin", map_location=torch.device('cpu'))
    decoder.load_state_dict(state_dict, strict=True)

    encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = train_encoder

    decoder.to(device)
    for param in decoder.parameters():
        param.requires_grad = train_decoder

    model = MVM(config_enc=encoder_config, config_dec=decoder_config, encoder=encoder, decoder=decoder,
                is_trainable_encoder=train_encoder)

    # Initialize MVM model with encoder and decoder
    if cp is not None:
        last_cp = get_last_cp(cp)
        if last_cp is None:
            raise ValueError("Checkpoint path does not exist")
        model_file = f"{last_cp}/pytorch_model.bin"
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
        print(f"Loaded model from {model_file}")
        missing_prefixes = list(set([k.split(".")[0] for k in missing_keys]))
        print(f"Missing prefixes: {missing_prefixes}")
        unexpected_prefixes = list(set([k.split(".")[0] for k in unexpected_keys]))
        print(f"Unexpected prefixes: {unexpected_prefixes}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    print(
        f"MODEL:MVM. Total parameters: {total_params:,}, (trainable: {trainable_params:,}, non-trainable: {non_trainable_params:,})")

    # Update output suffix to include encoder/decoder training info
    output_suf = f"{size}_{lr}"
    if train_encoder:
        output_suf += "_train_enc"
    if train_decoder:
        output_suf += "_train_dec"
    if cp is not None:
        output_suf += "_cp"

    os.makedirs(f"res_auto_mvm_retro/{output_suf}", exist_ok=True)
    train_args = TrainingArguments(
        output_dir=f"res_auto_mvm_retro/{output_suf}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=25,
        logging_dir=f"logs_auto_mvm_retro/{output_suf}",
        logging_steps=1000,
        save_steps=10000,
        evaluation_strategy="steps",
        eval_steps=10000,
        save_total_limit=1,
        load_best_model_at_end=True,
        save_safetensors=False,
        gradient_accumulation_steps=1,
        metric_for_best_model="eval_validation_token_accuracy",
        report_to="tensorboard",
        learning_rate=lr,
        lr_scheduler_type='constant',
        label_names=['products_input_ids', 'products_token_attention_mask', 'products_mol_attention_mask'],
        seed=42,
        save_only_model=True,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset={'validation': val_dataset, "train": train_subset},
        compute_metrics=lambda x: compute_metrics(x),
    )
    print(trainer.evaluate())
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--size", type=str, default="s")
    parser.add_argument("--train_encoder", type=int, default=1)
    parser.add_argument("--train_decoder", type=int, default=1)
    parser.add_argument("--cp", type=str, default=None)
    args = parser.parse_args()

    main(
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.size,
        bool(args.train_encoder),
        bool(args.train_decoder),
        args.cp,
    )
