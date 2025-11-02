#!/usr/bin/env python3
"""
Compute CSM scores for ClinVar mutations using batched inference with ESM2.

Steps:
1. Load ClinVar–DepMap joined data
2. Filter for unique mutations
3. Load pretrained ESM2 model and adapter
4. Batch all mutations by protein sequence
5. Compute ESM and CSM scores efficiently
6. Save results

"""

import os
import re
import torch
import pandas as pd
from peft import PeftModel
from transformers import EsmForMaskedLM, EsmTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------
# Helper function: batched mutation scoring
# ---------------------------------------------------------------------
def compute_mut_scores_for_sequence(model, tokenizer, mutations, sequence, device,  batch_size=8):
    model.eval()

    # Tokenize sequence once
    encoded = tokenizer(sequence, return_tensors="pt").to(device)

    scores = {}

    for i in range(0, len(mutations), batch_size):
        batch_muts = mutations[i : i + batch_size]

        input_ids = encoded["input_ids"].repeat(len(batch_muts), 1)

        mask_positions = []
        aa_indices = []

        for j, mut in enumerate(batch_muts):
            try:
                wt, pos, mt = mut[0], int(mut[1:-1]), mut[-1]
                pos -= 1  # 0-based
                if sequence[pos] != wt:
                    raise ValueError(
                        f"Wildtype mismatch at {pos+1}: expected {wt}, found {sequence[pos]}"
                    )
                # Mask the position (+1 for BOS)
                input_ids[j, pos + 1] = tokenizer.mask_token_id
                mask_positions.append(pos + 1)
                aa_indices.append(tokenizer.convert_tokens_to_ids(mt))
            except Exception as e:
                mask_positions.append(None)
                aa_indices.append(None)
                print(f"Skipping mutation {mut}: {e}", flush=True)

        with torch.no_grad():
            logits = model(input_ids).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        for j, mut in enumerate(batch_muts):
            if mask_positions[j] is None:
                scores[mut] = None
            else:
                scores[mut] = log_probs[j, mask_positions[j], aa_indices[j]].item()

        del logits, log_probs, input_ids
        torch.cuda.empty_cache()
    return scores


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Starting CSM scoring...", flush=True)

    # Input and output paths
    input_csv = "/g/data/gi52/jaime/data/clinvar_depmap_joined.csv"
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    adapter_path = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run11/epoch0_batch10000"
    output_dir = "/g/data/gi52/jaime/clinvar/run11_ms/batched2_all_genes"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    cols = ["Name", "HGNC_ID", "GeneSymbol", "ClinicalSignificance", "ProteinChange", "wt_protein_seq"]
    df = pd.read_csv(input_csv, usecols=cols, low_memory=False)
    df = df.drop_duplicates(subset=["Name", "ClinicalSignificance"]).copy()
    print(f"Loaded {len(df)} unique mutations", flush=True)

    # Load model + adapter
    print("Loading ESM2 model and adapter...", flush=True)
    tokenizer = EsmTokenizer.from_pretrained(base_model_path)
    base_model = EsmForMaskedLM.from_pretrained(base_model_path)
    adapter_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    csm_model = adapter_model.merge_and_unload()

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csm_model = csm_model.to(device)
    print(f"Running on device: {device}", flush=True)

    # frozen baseline
    frozen_base_model = EsmForMaskedLM.from_pretrained(base_model_path)
    frozen_base_model.eval()
    for p in frozen_base_model.parameters():
        p.requires_grad = False
    frozen_base_model = frozen_base_model.to(device)

    # Group by sequence for batching
    grouped = df.groupby("wt_protein_seq")
    results = []
    for i, (seq, group) in enumerate(tqdm(grouped, desc="Processing sequences")):
        muts = [re.sub(r"^p\.", "", m) for m in group["ProteinChange"].dropna()]
        
        # compute both scores 
        esm_scores = compute_mut_scores_for_sequence(frozen_base_model, tokenizer, muts, seq, device)
        csm_scores = compute_mut_scores_for_sequence(csm_model, tokenizer, muts, seq, device)
        group["esm_score"] = group["ProteinChange"].map(lambda x: esm_scores.get(re.sub(r"^p\.", "", x), None))
        group["csm_score"] = group["ProteinChange"].map(lambda x: csm_scores.get(re.sub(r"^p\.", "", x), None))
        results.append(group)

        # save first group as example
        if i == 0:
            example_path = os.path.join(output_dir, "example_first_sequence_batch.csv")
            group.to_csv(example_path, index=False)
            print(f"Saved example batch to {example_path}", flush=True)

        # Save periodically
        if (i + 1) % 20 == 0:
            checkpoint = os.path.join(output_dir, f"checkpoint_seqbatch_{i+1}.csv")
            pd.concat(results).to_csv(checkpoint, index=False)
            print(f"Saved checkpoint: {checkpoint}", flush=True)
        

    final_df = pd.concat(results, ignore_index=True)
    save_path = os.path.join(output_dir, "epoch0_batch10000_all_clinvar_csm_scores.csv")
    final_df.to_csv(save_path, index=False)
    print(f"Saved final results to {save_path}", flush=True)


if __name__ == "__main__":
    main()
