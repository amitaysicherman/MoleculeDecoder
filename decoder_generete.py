import torch
import torch.nn.functional as F


def generate(
        model,
        input_ids,
        attention_mask=None,
        max_length=50,
        num_beams=3,
        num_return_sequences=1,
        eos_token_id=1,
):
    """
    Generate sequences using beam search.

    Args:
        model: The MolFormerT5Decoder model
        input_ids: Input token ids (batch_size, sequence_length)
        attention_mask: Attention mask for input_ids
        max_length: Maximum length of generated sequence
        num_beams: Number of beams for beam search
        num_return_sequences: Number of sequences to return for each input (must be <= num_beams)
        eos_token_id: Token ID for end of sequence

    Returns:
        torch.Tensor: Generated sequences (batch_size * num_return_sequences, max_length)
    """
    if num_return_sequences > num_beams:
        raise ValueError(
            f"num_return_sequences ({num_return_sequences}) has to be less or equal to num_beams ({num_beams})")

    device = input_ids.device
    batch_size = input_ids.size(0)

    # Get encoder outputs (will be reused for all beam steps)
    with torch.no_grad():
        mol_outputs = model.molformer(input_ids, attention_mask=attention_mask)
        encoder_outputs = model.proj(mol_outputs.pooler_output).unsqueeze(1)

    # Expand encoder outputs for beam search
    encoder_outputs = encoder_outputs.expand(batch_size, num_beams, -1)
    encoder_outputs = encoder_outputs.contiguous().view(batch_size * num_beams, 1, -1)

    # Initialize beam search state
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
    beam_scores[:, 1:] = -1e9  # Initialize all beams except first to very low score

    # Initialize decoder input with start token
    decoder_input_ids = torch.full(
        (batch_size * num_beams, 1),
        model.config.decoder_start_token_id,
        dtype=torch.long,
        device=device
    )

    # Keep track of which sequences are finished
    done = [False for _ in range(batch_size)]
    generated_hyps = [[] for _ in range(batch_size)]

    with torch.no_grad():
        for step in range(max_length):
            # Forward pass through decoder
            decoder_outputs = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs
            )

            # Get next token logits
            next_token_logits = model.lm_head(decoder_outputs.last_hidden_state[:, -1, :])
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # Calculate scores for next tokens
            vocab_size = next_token_scores.shape[-1]

            # Move tensors to the same device and ensure correct shape
            beam_scores_for_add = beam_scores.view(-1, 1).to(next_token_scores.device)
            next_scores = beam_scores_for_add + next_token_scores
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            # Get top-k scores and tokens
            next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1)

            # Convert token indices
            next_beam_indices = (next_tokens // vocab_size).to(device)
            next_tokens = (next_tokens % vocab_size).to(device)

            # Build next beams
            new_decoder_input_ids = []
            new_beam_scores = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    continue

                beam_idx_offset = batch_idx * num_beams

                for beam_idx in range(num_beams):
                    beam_token_idx = next_tokens[batch_idx, beam_idx]
                    beam_score = next_scores[batch_idx, beam_idx]
                    beam_idx = next_beam_indices[batch_idx, beam_idx]

                    # Get sequence for this beam
                    beam_decoder_input_ids = decoder_input_ids[beam_idx_offset + beam_idx]
                    new_decoder_input_ids.append(torch.cat([beam_decoder_input_ids, beam_token_idx.unsqueeze(0)]))
                    new_beam_scores.append(beam_score)

                    # Check if sequence is complete
                    if beam_token_idx.item() == eos_token_id:
                        score = beam_score
                        generated_hyps[batch_idx].append((score.item(), new_decoder_input_ids[-1]))

                # Check if all beams for this batch item are done
                if len(generated_hyps[batch_idx]) == num_beams:
                    done[batch_idx] = True

            # Break if all sequences are done
            if all(done):
                break

            # Update beam state
            decoder_input_ids = torch.stack(new_decoder_input_ids).to(device)
            beam_scores = torch.tensor(new_beam_scores, device=device).view(batch_size, num_beams)

    # Select top-n hypotheses for each input
    output_sequences = []
    for batch_idx in range(batch_size):
        if not generated_hyps[batch_idx]:
            # If no complete sequences, take the current best incomplete ones
            best_beam_indices = beam_scores[batch_idx].argsort(descending=True)[:num_return_sequences]
            for beam_idx in best_beam_indices:
                sequence = decoder_input_ids[batch_idx * num_beams + beam_idx]
                output_sequences.append(sequence)
        else:
            # Sort completed sequences by score and take top-n
            sorted_hyps = sorted(generated_hyps[batch_idx], key=lambda x: x[0], reverse=True)
            for j in range(min(len(sorted_hyps), num_return_sequences)):
                score, sequence = sorted_hyps[j]
                output_sequences.append(sequence)

            # Pad with copies of the best sequence if we don't have enough
            while len(output_sequences) < (batch_idx + 1) * num_return_sequences:
                output_sequences.append(sequence)  # Use the last sequence

    # Stack all sequences and ensure they're on the right device
    return torch.stack(output_sequences).to(device)


# Example usage:
"""
model.eval()
generated_ids = generate(
    model,
    input_ids,
    attention_mask=attention_mask,
    max_length=50,
    num_beams=5,
    num_return_sequences=3,  # Return top 3 sequences for each input
    pad_token_id=0,
    eos_token_id=1
)

# Generated ids will have shape (batch_size * num_return_sequences, sequence_length)
# For each input, you'll get num_return_sequences different outputs
"""
if __name__ == "__main__":
    from train_decoder import create_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = create_model()
    model.load_state_dict(torch.load("results_pubchem/checkpoint-15000/pytorch_model.bin"), strict=False)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Test cases
    test_smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    tokens = tokenizer(test_smiles, padding="max_length", truncation=True, max_length=75, return_tensors="pt")
    generated_ids = generate(
        model,
        tokens['input_ids'].to(device),
        attention_mask=tokens['attention_mask'].to(device),
        max_length=50,
        num_beams=5,
        num_return_sequences=5,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_smiles = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for i, smiles in enumerate(generated_smiles):
        is_correct = smiles == test_smiles[int(i//5)]
        print(f"SMILES {i + 1}: {smiles}, {'Correct' if is_correct else 'Incorrect'}")
