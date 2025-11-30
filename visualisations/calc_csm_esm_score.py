#!/usr/bin/env python3
"""
make heatmap first, then score all clinvar mutations 

Compute CSM scores for ClinVar mutations using batched inference with ESM2.

Steps:
1. Load ClinVar–DepMap joined data
2. Load pretrained ESM2 model and adapter
3. Batch all mutations by protein sequence
4. Make heatmap
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

    # handle case where p.Val25= is p.Val25Val
    m = re.match(r"^p\.([A-Z][a-z]{2})(\d+)=$", out)
    if m:
        three_letter = m.group(1)
        pos = m.group(2)
        if three_letter in aa_map:
            aa1 = aa_map[three_letter]
            # Build one-letter “no change” form: p.V25V
            out = f"p.{aa1}{pos}{aa1}"

        print(f"Converted no-change notation {s} to {out}", flush=True)

    for k, v in aa_map.items():
        out = re.sub(k, v, out)
    return out

# ---------------------------------------------------------------------
# Helper function: batched mutation scoring
# ---------------------------------------------------------------------
def compute_mut_scores_for_sequence(model, tokenizer, mutations, sequence, device,  batch_size=4):
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
                        f"Wildtype mismatch at {pos+1}: expected {sequence[pos]}, found {wt}"
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Starting CSM scoring...", flush=True)

    # Input and output paths
    input_csv = "/g/data/gi52/jaime/data/clinvar_variant_summary.csv"
    protein_seq_csv = "/g/data/gi52/jaime/data/gene_sequences_with_ids.csv"
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    adapter_path = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run11/epoch0_batch10000"
    output_dir = "/g/data/gi52/jaime/clinvar/run11_ms/pan_essential_genes"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    cols = ["Name", "GeneSymbol", "HGNC_ID", "ClinicalSignificance"]
    df = pd.read_csv(input_csv, usecols=cols, low_memory=False)

    print(f"Loaded {len(df)} mutations", flush=True)

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

    ten_genes = ["PLK1", "CDK1", "CDK7", "AURKA", "ATR", "AURKB", "DNMT1", "BRD4", "HDAC3"]
    
    # get only these genes
    results = []
    df = df[df["GeneSymbol"].isin(ten_genes)].copy()

    # load protein sequences
    protein_seqs = pd.read_csv(protein_seq_csv)

    # write heading to new file
    final_path = os.path.join(output_dir, "pan_essential_csm_esm_scores.csv")
    with open(final_path, "w") as f:
        f.write("Name,GeneSymbol,HGNC_ID,ClinicalSignificance,ProteinChange,esm_score,csm_score\n")

    # process each gene 
    for i, gene in enumerate(ten_genes):
        print(f"Processing gene {i+1}/{len(ten_genes)}: {gene}", flush=True)

        seq_row = protein_seqs.loc[protein_seqs["symbol"] == gene, "sequence"]
        if seq_row.empty:
            print(f"No protein sequence found for {gene}, skipping.", flush=True)
            continue
        seq = seq_row.iloc[0] 
        print(f"Protein sequence for {gene}: {seq}", flush=True)

        # Extract variants for this gene
        group = df[df["GeneSymbol"] == gene].copy()

        # Extract and normalize mutation notation
        group["ProteinChange"] = group["Name"].str.extract(r"(p\.[^ )]+)")
        group["ProteinChange"] = group["ProteinChange"].apply(convert_protein_notation)
        
        valid_mask = group["ProteinChange"].notna() & group["ProteinChange"].str.match(r"^p\.[A-Z]\d+[A-Z]$")
        muts = [re.sub(r"^p\.", "", m) for m in group.loc[valid_mask, "ProteinChange"]]     
        
        print(f"Found {len(muts)} missense mutations for {gene}", flush=True)
        print(f"Examples: {muts[:10]}", flush=True)

        if not muts:
            print(f"No valid mutations found for {gene}, skipping.", flush=True)
            continue
        
        # compute both scores 
        try: 
            esm_scores = compute_mut_scores_for_sequence(frozen_base_model, tokenizer, muts, seq, device)
            csm_scores = compute_mut_scores_for_sequence(csm_model, tokenizer, muts, seq, device)
            
            # assign the new column via loc
            group.loc[valid_mask, "esm_score"] = pd.Series(esm_scores, index=group.loc[valid_mask].index)
            group.loc[valid_mask, "csm_score"] = pd.Series(csm_scores, index=group.loc[valid_mask].index)

            # Debugging
            print(f"ESM scores for {gene}: {group['esm_score'].dropna().tolist()[:10]}", flush=True)
            print(f"CSM scores for {gene}: {group['csm_score'].dropna().tolist()[:10]}", flush=True)    

            # Print first ten rows 
            print(f"First ten scored mutations for {gene}:\n{group.head(10)}", flush=True)

            # Save intermediate results
            group.to_csv(final_path, mode="a", index=False, header=False)
            print(f"Appended {len(group)} results for {gene} to {final_path}", flush=True)

            # save first group as example
            if i == 0:
                example_path = os.path.join(output_dir, "example_first_gene.csv")
                group.to_csv(example_path, index=False)
                print(f"Saved example batch to {example_path}", flush=True)

            results.append(group)
        except Exception as e:
            print(f"Error processing gene {gene}: {e}", flush=True)
            continue 

    print(f"Processing complete — results appended to {final_path}", flush=True)

if __name__ == "__main__":
    main()
