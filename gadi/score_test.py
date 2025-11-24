#!/usr/bin/env python3
"""
Test scoring is working correctly.

"""

import os
import re
import torch
import pandas as pd
from peft import PeftModel
from transformers import EsmForMaskedLM, EsmTokenizer
from tqdm import tqdm


def compute_mut_score(model, tokenizer, mutation, sequence, device):
    model.eval()

    wt, pos, mt = mutation[0], int(mutation[1:-1]), mutation[-1]
    # Adjust for 0-based indexing
    pos -= 1
    if sequence[pos] != wt:
        raise ValueError(f"Wildtype mismatch at position {pos+1} for mutation {mutation}: expected {sequence[pos]}, found {wt}")
    
    # Tokenize the sequence
    encoded = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]

    # Mask the site 
    masked_input_ids = input_ids.clone()
    masked_input_ids[0, pos + 1] = tokenizer.mask_token_id  # +1 due to BOS token

    with torch.no_grad():
        logits = model(masked_input_ids).logits
        log_probs = torch.nn.functional.log_softmax(logits[0, pos + 1], dim=0)
    
    # Get the log probability of the mutated amino acid at the mutation position
    aa_index = tokenizer.convert_tokens_to_ids(mt)
    score = log_probs[aa_index].item()

    # free memory
    del logits, log_probs, masked_input_ids, encoded
    torch.cuda.empty_cache()

    return score

# ---------------------------------------------------------------------
# Helper function: batched mutation scoring
# ---------------------------------------------------------------------
def old_compute_mut_scores_for_sequence(model, tokenizer, mutations, sequence, device, batch_size=4): 
    model.eval() 
    # Tokenize sequence once 
    encoded = tokenizer(sequence, return_tensors="pt").to(device) 
    scores = [] 
    for i in range(0, len(mutations), batch_size): 
        batch_muts = mutations[i : i + batch_size] 
        input_ids = encoded["input_ids"].repeat(len(batch_muts), 1) 
        mask_positions = [] 
        aa_indices = [] 
        for j, mut in enumerate(batch_muts): 
            try: 
                wt, pos, mt = mut[0], int(mut[1:-1]), mut[-1] 
                pos -= 1 # 0-based 
                if sequence[pos] != wt: 
                    raise ValueError( 
                        f"Wildtype mismatch at {pos+1}: expected {wt}, found {sequence[pos]}" 
                    ) # Mask the position (+1 for BOS) 
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
                scores.append(None) 
            else: 
                scores.append(log_probs[j, mask_positions[j], aa_indices[j]].item()) 
        del logits, log_probs, input_ids 
        torch.cuda.empty_cache() 
    return scores



def fixed_compute_mut_scores_for_sequence(model, tokenizer, mutations, sequence, device,  batch_size=4):
    model.eval()

    # Tokenize sequence once
    encoded = tokenizer(sequence, return_tensors="pt").to(device)
    base_input_ids = encoded["input_ids"]

    scores = []

    for i in range(0, len(mutations), batch_size):
        batch_muts = mutations[i : i + batch_size]

        input_ids_batch = []
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
                masked_input = base_input_ids.clone()
                masked_input[0, pos + 1] = tokenizer.mask_token_id
                input_ids_batch.append(masked_input)

                mask_positions.append(pos + 1)
                aa_indices.append(tokenizer.convert_tokens_to_ids(mt))
            except Exception as e:
                mask_positions.append(None)
                aa_indices.append(None)
                print(f"Skipping mutation {mut}: {e}", flush=True)

        input_ids = torch.cat(input_ids_batch, dim=0).to(device)
        with torch.no_grad():
            logits = model(input_ids).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        for j, mut in enumerate(batch_muts):
            if mask_positions[j] is None:
                scores.append(None)
            else:
                scores.append(log_probs[j, mask_positions[j], aa_indices[j]].item())

        del logits, log_probs, input_ids, input_ids_batch
        torch.cuda.empty_cache()
    return scores

def main():
    print("Testing mutation scoring functions...")
    # Input and output paths
    input_csv = "/g/data/gi52/jaime/data/clinvar_depmap_joined.csv"
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    adapter_path = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run11/epoch0_batch10000"


    # Load data
    cols = ["Name", "HGNC_ID", "GeneSymbol", "ClinicalSignificance", "ProteinChange", "wt_protein_seq"]
    df = pd.read_csv(input_csv, usecols=cols, low_memory=False)
    print(f"Loaded {len(df)} unique mutations", flush=True)
    print(f" Example protein changes: {df['ProteinChange'].head().tolist()}", flush=True)

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
    grouped = df.groupby("wt_protein_seq", sort=False)
    num_groups = df["wt_protein_seq"].nunique()
    num_genes = df["GeneSymbol"].nunique()
    print(f"Total unique sequences to process: {num_groups}", flush=True)  
    print(f"Total unique genes to process: {num_genes}", flush=True)  
    print(f"Number of TP53 mutations: {len(df[df['GeneSymbol'] == 'TP53'])}", flush=True)

    # Test on a TP53 muatation set
    tp53_data = df[df["GeneSymbol"] == "TP53"].copy()
    seq = tp53_data["wt_protein_seq"].iloc[0]
    mutations = tp53_data["ProteinChange"].tolist()
    print(mutations[:10])


    # Find entries that are in tp53 csv file but not depmap joined file
    tp53_file = pd.read_csv("/g/data/gi52/jaime/clinvar/tp53_clinvar_csm_esm_scores.csv")
    tp53_mutations_file = set(tp53_file["ProteinChange"].tolist())
    tp53_mutations_joined = set(tp53_data["ProteinChange"].tolist())    
    missing_mutations = tp53_mutations_file - tp53_mutations_joined

    # test if certain mutatins are missing
    print("Checking for specific missing mutations...", flush=True)
    test_mutations = ["R248W", "E258K", "L252P"]
    for mut in test_mutations:
        if mut in missing_mutations:
            print(f"Mutation {mut} is missing from joined file.", flush=True)
        else:
            print(f"Mutation {mut} is present in joined file.", flush=True)
    exit()

    print(f"Mutations in tp53 file but not in joined file: {missing_mutations}", flush=True)    
    print(f"Total missing mutations: {len(missing_mutations)}", flush=True)

    # Count number of mutations that end in '.'
    dot_mutations = [m for m in missing_mutations if isinstance(m, str) and m.endswith('.')]
    print(f"Mutations ending with '.': {dot_mutations}", flush=True)
    print(f"Total mutations ending with '.': {len(dot_mutations)}", flush=True)


    mutations = [re.sub(r"^p\.", "", m) for m in mutations if isinstance(m, str)]
    print(mutations[:10])

    for mut in mutations:
        #s1 = compute_mut_score(csm_model, tokenizer, mut, seq, device)
        try:
            s2 = old_compute_mut_scores_for_sequence(csm_model, tokenizer, [mut], seq, device)[0]
            s3 = fixed_compute_mut_scores_for_sequence(csm_model, tokenizer, [mut], seq, device)[0]

        #print('csm model scores', mut, s2, s3)

            if (s2 != s3):
                print(f"Discrepancy in CSM model scores for {mut}: {s2} vs {s3}", flush=True )

            if (s2 < -15 or s2 < -15):
                print(f"low scores for {mut}: {s2} vs {s3}", flush=True )
        except Exception as e:
            print(f"Error processing mutation {mut}: {e}", flush=True)

    

if __name__ == "__main__":
    main()
