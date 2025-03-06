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

hidden_sizes = {'s': 128, 'm': 512, 'l': 512}
num_layers = {'s': 2, 'm': 6, 'l': 24}
num_heads = {'s': 2, 'm': 4, 'l': 8}

# if mps use mps, else if gpu use gpu, else use cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def size_to_config(size, hidden_size=512):
    config = BertConfig(
        vocab_size=1,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers[size],
        num_attention_heads=num_heads[size],
        intermediate_size=hidden_sizes[size] * 4,
        max_position_embeddings=25,
    )
    return config


class ReactionMolsDataset(Dataset):
    def __init__(self, base_dir="USPTO", split="train", max_mol_len=75, max_len=10, skip_unk=True, is_molformer=False):
        self.max_len = max_len
        self.max_mol_len = max_mol_len
        self.skip_unk = skip_unk
        self.is_molformer = is_molformer
        with open(f"{base_dir}/reactants-{split}.txt") as f:
            self.reactants = f.read().splitlines()
        with open(f"{base_dir}/products-{split}.txt") as f:
            self.products = f.read().splitlines()
        with open(f"{base_dir}/reagents-{split}.txt") as f:
            self.reagents = f.read().splitlines()
        if is_molformer:
            self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        else:
            self.tokenizer = get_tokenizer()
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
        if not self.is_molformer:
            mols = [smiles_to_tokens(s) for s in mols]
            if None in mols:
                return None
            if any([len(s) > self.max_mol_len for s in mols]):
                return None
            mols = [" ".join(m) for m in mols]
            tokens = [self.tokenizer.encode(m, max_length=self.max_mol_len) for m in mols]
        else:
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
        reactants, products, reagents = self.reactants[idx], self.products[idx], self.reagents[idx]

        continue_run = True
        while continue_run:
            reactants_tokens = self.preprocess_line(reactants)
            products_tokens = self.preprocess_line(products)
            reagents_tokens = self.preprocess_line(reagents)
            if reactants_tokens is not None and products_tokens is not None and reagents_tokens is not None:
                continue_run = False
            else:
                idx = random.randint(0, len(self.reactants) - 1)
                reactants, products, reagents = self.reactants[idx], self.products[idx], self.reagents[idx]

        return {
            'reactants_input_ids': reactants_tokens['input_ids'],
            'reactants_token_attention_mask': reactants_tokens['token_attention_mask'],
            'reactants_mol_attention_mask': reactants_tokens['mol_attention_mask'],
            'products_input_ids': products_tokens['input_ids'],
            'products_token_attention_mask': products_tokens['token_attention_mask'],
            'products_mol_attention_mask': products_tokens['mol_attention_mask'],
            'reagents_input_ids': reagents_tokens['input_ids'],
            'reagents_token_attention_mask': reagents_tokens['token_attention_mask'],
            'reagents_mol_attention_mask': reagents_tokens['mol_attention_mask'],
        }

    def __len__(self):
        return len(self.reactants)


class MVM(nn.Module):
    def __init__(self, config, alpha=0.5):
        super().__init__()
        self.config = config
        self.alpha = alpha

        self.bert_model = BertModel(config)

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size), requires_grad=True)
        self.sep_token = nn.Parameter(torch.randn(1, 1, config.hidden_size), requires_grad=True)
        # self.pad_embedding = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=False)

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
                outputs = encoder(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask
                )
                all_embeddings.append(outputs.squeeze(1))
        embeddings = torch.cat(all_embeddings, dim=0)  # (batch_size * max_seq_len, hidden_size)
        final_emb = torch.zeros(flat_mol_attention_mask.size(0), embeddings.size(-1), device=embeddings.device)
        final_emb[flat_mol_attention_mask.nonzero(as_tuple=True)[0]] = embeddings
        final_emb = final_emb.view(batch_size, max_seq_len, -1)
        return final_emb

    def forward(self, reactants_input_ids, reactants_token_attention_mask, reactants_mol_attention_mask,
                products_input_ids, products_token_attention_mask, products_mol_attention_mask,
                reagents_input_ids, reagents_token_attention_mask, reagents_mol_attention_mask):
        reactants_embeddings = self.get_mol_embeddings(reactants_input_ids, reactants_token_attention_mask,
                                                       reactants_mol_attention_mask)
        reagents_embeddings = self.get_mol_embeddings(reagents_input_ids, reagents_token_attention_mask,
                                                      reagents_mol_attention_mask)

        cls = self.cls_token.expand(reactants_embeddings.size(0), -1, -1)
        sep = self.sep_token.expand(reactants_embeddings.size(0), -1, -1)
        src_embeddings = torch.cat([cls, reactants_embeddings, sep, reagents_embeddings], dim=1)
        ones_mask = torch.ones(src_embeddings.size(0), 1, device=src_embeddings.device)
        src_mol_attention_mask = torch.cat(
            [ones_mask, reactants_mol_attention_mask, ones_mask, reagents_mol_attention_mask], dim=1)
        src_seq_mask = src_mol_attention_mask.float()
        output = self.bert_model(inputs_embeds=src_embeddings, attention_mask=src_seq_mask)
        bert_predict = output['pooler_output']

        gt_tgt_embeddings = self.get_mol_embeddings(products_input_ids, products_token_attention_mask,
                                                    products_mol_attention_mask)
        gt_tgt_embeddings = gt_tgt_embeddings[:, 0, :]
        loss = F.mse_loss(bert_predict, gt_tgt_embeddings)

        tgt_input_ids_decoder = products_input_ids[:, 0].clone()
        tgt_input_ids_decoder_mask = products_token_attention_mask[:, 0]
        labels = tgt_input_ids_decoder.clone()
        labels_mask = products_token_attention_mask[:, 0]
        labels[labels_mask == 0] = -100

        decoder_output = decoder(tgt_input_ids_decoder, tgt_input_ids_decoder_mask,
                                 encoder_hidden_states=bert_predict.unsqueeze(1),
                                 labels=labels)

        decoder_output.decoder_hidden_states = bert_predict
        decoder_output.loss = self.alpha * loss + (1 - self.alpha) * decoder_output.loss

        return decoder_output


def compute_metrics(eval_pred, is_molformer=False):
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

    # use shift right to get the labels
    if not is_molformer:
        predictions = predictions[:, :-1]
        labels = labels[:, 1:]

    mask = labels != -100
    total_tokens = mask.sum()
    correct_tokens = ((predictions == labels) & mask).sum()
    token_accuracy = correct_tokens / total_tokens
    correct_or_pad = (predictions == labels) | (~mask)
    correct_samples = correct_or_pad.all(axis=-1).sum()
    total_samples = len(labels)
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


def main(batch_size=1024, num_epochs=10, lr=1e-4, size="m", alpha=0.5, use_molformer=False):
    train_dataset = ReactionMolsDataset(split="train", is_molformer=use_molformer)
    val_dataset = ReactionMolsDataset(split="valid", is_molformer=use_molformer)
    train_subset_random_indices = random.sample(range(len(train_dataset)), len(val_dataset))
    train_subset = torch.utils.data.Subset(train_dataset, train_subset_random_indices)
    config = size_to_config(size)
    model = MVM(config=config, alpha=alpha)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    print(
        f"MODEL:MVM. Total parameters: {total_params:,}, (trainable: {trainable_params:,}, non-trainable: {non_trainable_params:,})")
    output_suf = f"{size}_{lr}_{alpha}"
    if use_molformer:
        output_suf += "_molformer"
    os.makedirs(f"res_auto_mvm/{output_suf}", exist_ok=True)
    train_args = TrainingArguments(
        output_dir=f"res_auto_mvm/{output_suf}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=10,
        logging_dir=f"logs_auto_mvm/{output_suf}",
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
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
        compute_metrics=lambda x: compute_metrics(x, is_molformer=use_molformer),
    )

    # model.load_state_dict(
    #     torch.load(f"{get_last_cp(f'res_auto_mvm/{output_suf}')}/pytorch_model.bin", map_location=device))
    #
    # print(trainer.evaluate())
    trainer.train(resume_from_checkpoint=get_last_cp(f"res_auto_mvm/{output_suf}") is not None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--size", type=str, default="s")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--molformer", action="store_true")
    args = parser.parse_args()

    from autoencoder.model import get_model

    tokenizer = get_tokenizer()

    if args.molformer:
        from transformers import AutoModel
        from train_decoder import create_model

        encoder = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            deterministic_eval=True,
            trust_remote_code=True,
            use_safetensors=False  # Force using PyTorch format instead of safetensors
        )
        encoder.eval().to(device)
        for param in encoder.parameters():
            param.requires_grad = False

        decoder, _ = create_model()
        state_dict = torch.load("results_decoder/checkpoint-195000/pytorch_model.bin", map_location=torch.device('cpu'))
        decoder.load_state_dict(state_dict, strict=True)
        decoder.eval().to(device)
        for param in decoder.parameters():
            param.requires_grad = False
    else:
        model = get_model('ae', "m", tokenizer)
        state_dict = torch.load(f"{get_last_cp('res_auto/ae_m')}/pytorch_model.bin", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        encoder = model.encoder
        encoder.eval().to(device)
        for param in encoder.parameters():
            param.requires_grad = False
        decoder = model.decoder
        decoder.eval().to(device)
        for param in decoder.parameters():
            param.requires_grad = False

    main(args.batch_size, args.num_epochs, args.lr, args.size, args.alpha, args.molformer)
