#!/usr/bin/env python3
"""
Compute CSM scores for ClinVar mutations using batched inference with ESM2.

Steps:
1. Load ClinVar data
2. DO NOT Filter for unique mutations
3. Load pretrained ESM2 model and adapter
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

# ==== Helper function: 3-letter → 1-letter amino acid codes ====
aa_map = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C",
    "Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
    "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P",
    "Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V"
}

def convert_protein_notation(s):
    if pd.isna(s):
        return s
    out = s
    for k, v in aa_map.items():
        out = re.sub(k, v, out)
    return out


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


def main():
    print("Starting CSM scoring...", flush=True)

    # Input and output paths
    input_csv = "/g/data/gi52/jaime/data/clinvar_variant_summary.csv"
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    adapter_path = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run11/epoch0_batch10000"
    output_dir = "/g/data/gi52/jaime/clinvar/run11_ms/rpl15"
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, "rpl15_clinvar_csm_scores.csv")

    # Load data with RPL15 as gene symbol
    cols = ["Name", "HGNC_ID", "GeneSymbol", "ClinicalSignificance"]
    df = pd.read_csv(input_csv, usecols=cols, low_memory=False)
    df = df[df["GeneSymbol"] == "RPL15"].copy()
    print(f"Loaded {len(df)} mutations", flush=True)
    print(f" Examples: {df.head()}", flush=True)

    df["ProteinChange"] = df["Name"].str.extract(r"(p\.[^ )]+)")
    df["ProteinChange"] = df["ProteinChange"].apply(convert_protein_notation)

    print(df.head(), flush=True)

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

    # Check if we’re appending to an existing file
    header_written = os.path.exists(final_path)
    if header_written:
        print(f"Appending to existing output file: {final_path}", flush=True)
    else:
        print(f"Creating new output file: {final_path}", flush=True)
        # add headings 
        with open(final_path, "w") as f:
            f.write("Name,HGNC_ID,GeneSymbol,ClinicalSignificance,ProteinChange,esm_score,csm_score\n")

    results = []

    for i, (idx, row) in enumerate(df.iterrows()):

        mut = str(row.ProteinChange)
        print(f"Processing row {i}: {row}", flush=True)

        mut = re.sub(r"^p\.", "", mut)

        print(mut)
        if mut is None or pd.isna(mut):
            print(f"Skipping invalid mutation format: {row.Name}", flush=True)
            df.loc[row.Index, "esm_score"] = None
            df.loc[row.Index, "csm_score"] = None
            continue

        seq = "MGAYKYIQELWRKKQSDVMRFLLRVRCWQYRQLSALHRAPRPTRPDKARRLGYKAKQGYVIYRIRVRRGGRKRPVPKGATYGKPVHHGVNQLKFARSLQSVAEERAGRHCGALRVLNSYWVGEDSTYKFFEVILIDPFHKAIRRNPDTQWITKPVHKHREMRGLTSAGRKSRGLGKGHKFHHTIGGSRRAAWRRRNTLQLHRYR"
    
        # compute both scores 
        try:
            esm_scores = compute_mut_score(frozen_base_model, tokenizer, mut, seq, device)
            csm_scores = compute_mut_score(csm_model, tokenizer, mut, seq, device)

            df.loc[idx, "esm_score"] = esm_scores 
            df.loc[idx, "csm_score"] = csm_scores 

            updated_row = df.loc[[idx]]

            # Append directly to the final output file
            updated_row.to_csv(final_path, mode="a", index=False, header=not header_written)
            header_written = True
            print(f"Appended results of group {i} to {final_path}", flush=True)

        except Exception as e:
            print(f"Error processing mutation {mut} at row {i}: {e}", flush=True)
            df.loc[idx, ["esm_score", "csm_score"]] = [None, None]

        
        
    print(f"Processing complete — results appended to {final_path}", flush=True)



if __name__ == "__main__":
    main()
