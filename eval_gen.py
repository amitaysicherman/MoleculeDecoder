import torch
from train_decoder import create_model
from train_script import get_model as get_concept_model
from dataset import ReactionMolsDataset
from torch.utils.data import DataLoader
from transformers import AutoModel
from trainer import get_mol_embeddings
from train_decoder import _shift_right
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import torch.nn.functional as F

# def generate(
#         model,
#         emb,
#         max_length=50,
#         num_beams=3,
#         num_return_sequences=1,
#         eos_token_id=1,
# ):
#     if num_return_sequences > num_beams:
#         raise ValueError(
#             f"num_return_sequences ({num_return_sequences}) has to be less or equal to num_beams ({num_beams})")
#
#     batch_size = emb.size(0)
#
#     with torch.no_grad():
#         encoder_outputs = model.proj(emb).unsqueeze(1)
#
#     encoder_outputs = encoder_outputs.expand(batch_size, num_beams, -1)
#     encoder_outputs = encoder_outputs.contiguous().view(batch_size * num_beams, 1, -1)
#
#     # Initialize beam search state
#     beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
#     beam_scores[:, 1:] = -1e9  # Initialize all beams except first to very low score
#
#     # Initialize decoder input with start token
#     decoder_input_ids = torch.full(
#         (batch_size * num_beams, 1),
#         model.config.decoder_start_token_id,
#         dtype=torch.long,
#         device=device
#     )
#
#     # Keep track of which sequences are finished
#     done = [False for _ in range(batch_size)]
#     generated_hyps = [[] for _ in range(batch_size)]
#
#     with torch.no_grad():
#         for step in range(max_length):
#             # Forward pass through decoder
#             decoder_outputs = model.decoder(
#                 input_ids=decoder_input_ids,
#                 encoder_hidden_states=encoder_outputs
#             )
#
#             # Get next token logits
#             next_token_logits = model.lm_head(decoder_outputs.last_hidden_state[:, -1, :])
#             next_token_scores = F.log_softmax(next_token_logits, dim=-1)
#
#             # Calculate scores for next tokens
#             vocab_size = next_token_scores.shape[-1]
#
#             # Move tensors to the same device and ensure correct shape
#             beam_scores_for_add = beam_scores.view(-1, 1).to(next_token_scores.device)
#             next_scores = beam_scores_for_add + next_token_scores
#             next_scores = next_scores.view(batch_size, num_beams * vocab_size)
#
#             # Get top-k scores and tokens
#             next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1)
#
#             # Convert token indices
#             next_beam_indices = (next_tokens // vocab_size).to(device)
#             next_tokens = (next_tokens % vocab_size).to(device)
#
#             # Build next beams
#             new_decoder_input_ids = []
#             new_beam_scores = []
#
#             for batch_idx in range(batch_size):
#                 if done[batch_idx]:
#                     continue
#
#                 beam_idx_offset = batch_idx * num_beams
#
#                 for beam_idx in range(num_beams):
#                     beam_token_idx = next_tokens[batch_idx, beam_idx]
#                     beam_score = next_scores[batch_idx, beam_idx]
#                     beam_idx = next_beam_indices[batch_idx, beam_idx]
#
#                     # Get sequence for this beam
#                     beam_decoder_input_ids = decoder_input_ids[beam_idx_offset + beam_idx]
#                     new_decoder_input_ids.append(torch.cat([beam_decoder_input_ids, beam_token_idx.unsqueeze(0)]))
#                     new_beam_scores.append(beam_score)
#
#                     # Check if sequence is complete
#                     if beam_token_idx.item() == eos_token_id:
#                         score = beam_score
#                         generated_hyps[batch_idx].append((score.item(), new_decoder_input_ids[-1]))
#
#                 # Check if all beams for this batch item are done
#                 if len(generated_hyps[batch_idx]) == num_beams:
#                     done[batch_idx] = True
#
#             # Break if all sequences are done
#             if all(done):
#                 break
#
#             # Update beam state
#             decoder_input_ids = torch.stack(new_decoder_input_ids).to(device)
#             beam_scores = torch.tensor(new_beam_scores, device=device).view(batch_size, num_beams)
#
#     # Select top-n hypotheses for each input
#     output_sequences = []
#     for batch_idx in range(batch_size):
#         if not generated_hyps[batch_idx]:
#             # If no complete sequences, take the current best incomplete ones
#             best_beam_indices = beam_scores[batch_idx].argsort(descending=True)[:num_return_sequences]
#             for beam_idx in best_beam_indices:
#                 sequence = decoder_input_ids[batch_idx * num_beams + beam_idx]
#                 output_sequences.append(sequence)
#         else:
#             # Sort completed sequences by score and take top-n
#             sorted_hyps = sorted(generated_hyps[batch_idx], key=lambda x: x[0], reverse=True)
#             for j in range(min(len(sorted_hyps), num_return_sequences)):
#                 score, sequence = sorted_hyps[j]
#                 output_sequences.append(sequence)
#
#             # Pad with copies of the best sequence if we don't have enough
#             while len(output_sequences) < (batch_idx + 1) * num_return_sequences:
#                 output_sequences.append(sequence)  # Use the last sequence
#
#     # Stack all sequences and ensure they're on the right device
#     return torch.stack(output_sequences).to(device)
#
def load_models():
    decoder_model, tokenizer = create_model()
    state_dict = torch.load("results_decoder/checkpoint-195000/pytorch_model.bin", map_location=torch.device('cpu'))
    decoder_model.load_state_dict(state_dict, strict=True)
    decoder_model = decoder_model.to(device).eval()

    concept_model = get_concept_model()
    concept_model.load_state_dict(
        torch.load("outputs/20250219_133220/best_model.pt", map_location=torch.device('cpu'))['model_state_dict'],
        strict=True)
    concept_model = concept_model.to(device).eval()

    molformer = AutoModel.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct",
        deterministic_eval=True,
        trust_remote_code=True
    ).to(device).eval()
    for param in molformer.parameters():
        param.requires_grad = False

    return decoder_model, concept_model, molformer, tokenizer





decoder_model, concept_model, molformer, tokenizer = load_models()
test_dataset = ReactionMolsDataset(base_dir="USPTO", split="valid", debug=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class Metrics:
    def __init__(self):
        self.perfect_match_accuracy = 0
        self.token_accuracy = 0
        self.gt_perfect_match_accuracy = 0
        self.gt_token_accuracy = 0
        self.total_samples = 0
        self.total_tokens = 0

    def update(self, labels, predictions, gt_predictions):
        labels = labels.cpu().numpy().flatten()
        predictions = predictions.detach().cpu().numpy().flatten()
        gt_predictions = gt_predictions.detach().cpu().numpy().flatten()
        is_pad = labels == -100
        predictions = predictions[~is_pad]
        gt_predictions = gt_predictions[~is_pad]
        labels = labels[~is_pad]
        self.token_accuracy += (predictions == labels).sum()
        self.gt_token_accuracy += (gt_predictions == labels).sum()
        labels_smiles = tokenizer.decode(labels, skip_special_tokens=True)
        predictions_smiles = tokenizer.decode(predictions, skip_special_tokens=True)
        gt_predictions = tokenizer.decode(gt_predictions, skip_special_tokens=True)
        self.perfect_match_accuracy += labels_smiles == predictions_smiles
        self.gt_perfect_match_accuracy += labels_smiles == gt_predictions
        self.total_samples += 1
        self.total_tokens += len(labels)

    def to_dict(self):
        return {
            "perfect_match_accuracy": self.perfect_match_accuracy / self.total_samples,
            "token_accuracy": self.token_accuracy / self.total_tokens,
            "gt_perfect_match_accuracy": self.gt_perfect_match_accuracy / self.total_samples,
            "gt_token_accuracy": self.gt_token_accuracy / self.total_tokens
        }


scores = Metrics()
pbar = tqdm(test_loader)
for batch in pbar:
    batch = {k: v.to(device) for k, v in batch.items()}
    intput_embeddings = get_mol_embeddings(
        molformer,
        batch['src_input_ids'],
        batch['src_token_attention_mask']
    )
    output_embeddings = get_mol_embeddings(
        molformer,
        batch['tgt_input_ids'],
        batch['tgt_token_attention_mask']
    )
    if batch['src_input_ids'][0,0,-1] != tokenizer.pad_token_id:
        continue
    if batch['tgt_input_ids'][0,0,-1] != tokenizer.pad_token_id:
        continue
    outputs = concept_model(
        src_embeddings=intput_embeddings,
        tgt_embeddings=output_embeddings,
        src_mol_attention_mask=batch['src_mol_attention_mask'],
        tgt_mol_attention_mask=batch['tgt_mol_attention_mask'],
        v2m=decoder_model,
        output_tokens=batch['tgt_input_ids'],

        return_seq=True
    )

    mol_outputs = outputs[0][0:1]
    encoder_outputs = decoder_model.proj(mol_outputs).unsqueeze(1)
    # Run through decoder
    input_ids = batch['tgt_input_ids'][0][:1]
    decoder_input_ids = _shift_right(input_ids, decoder_start_token_id=tokenizer.pad_token_id, pad_token_id=tokenizer.pad_token_id)
    decoder_output = decoder_model.decoder(encoder_hidden_states=encoder_outputs, input_ids=decoder_input_ids)
    pred_lm_logits = decoder_model.lm_head(decoder_output.last_hidden_state).argmax(dim=-1)

    # decoder_output = decoder_model.decoder(encoder_hidden_states=output_embeddings[:, :1], input_ids=decoder_input_ids)
    # gt_lm_logits = decoder_model.lm_head(decoder_output.last_hidden_state).argmax(dim=-1)

    attention_mask = batch['tgt_token_attention_mask'][0][:1]
    labels = batch['tgt_input_ids'][0][:1].clone()
    labels[attention_mask == 0] = -100

    gt_lm_logits=decoder_model(input_ids, attention_mask, labels).logits.argmax(dim=-1)

    labels = batch['tgt_input_ids'][0][:1].clone()
    labels[attention_mask == 0] = -100

    scores.update(labels, pred_lm_logits, gt_lm_logits)

    pbar.set_postfix(scores.to_dict())

# 2025-02-19 13:32:22,773 - PyTorch version 2.5.1 available.
