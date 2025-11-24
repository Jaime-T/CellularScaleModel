import os
import re
import pandas as pd
from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import EsmForMaskedLM, EsmTokenizer

# Helper function: batched mutation scoring
# ---------------------------------------------------------------------
def compute_mut_scores_for_sequence(model, tokenizer, mutations, sequence, device,  batch_size=4):
    model.eval()

    # Tokenize sequence once
    encoded = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024).to(device)

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
    

def main():

    output_dir="/g/data/gi52/jaime/clinvar/run11_ms/batched5_all_genes"
    start_index=4760  # Index to resume from
    final_path = os.path.join(output_dir, "all_clinvar_csm_scores.csv")

    # Input and output paths
    input_csv = "/g/data/gi52/jaime/data/clinvar_depmap_joined.csv"
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    adapter_path = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run11/epoch0_batch10000"


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
    num_groups = df["wt_protein_seq"].nunique()
    print(f"Total unique sequences to process: {num_groups}", flush=True)  



    # Check if we’re appending to an existing file
    header_written = os.path.exists(final_path)
    if header_written:
        print(f"Appending to existing output file: {final_path}", flush=True)

    print(f"Starting from group index {start_index}", flush=True)

    for i, (seq, group) in enumerate(tqdm(grouped, desc="Processing sequences")):
        # ⏭ Skip until the desired starting index
        if i < start_index:
            continue

        if start_index > 0 and not header_written:
            raise RuntimeError("Starting from nonzero index, but output file missing! Possible overwrite.")


        print(f"Processing group {i} / {num_groups}", flush=True)
        print(f"Gene symbols in this group: {group['GeneSymbol'].unique()}", flush=True)

        muts = [re.sub(r"^p\.", "", m) for m in group["ProteinChange"].dropna()]

        # Compute both scores
        esm_scores = compute_mut_scores_for_sequence(frozen_base_model, tokenizer, muts, seq, device)
        csm_scores = compute_mut_scores_for_sequence(csm_model, tokenizer, muts, seq, device)

        group["esm_score"] = group["ProteinChange"].map(lambda x: esm_scores.get(re.sub(r"^p\.", "", x), None))
        group["csm_score"] = group["ProteinChange"].map(lambda x: csm_scores.get(re.sub(r"^p\.", "", x), None))

        # Save first processed group example (if starting fresh)
        if i == start_index:
            example_path = os.path.join(output_dir, f"cont_example_sequence_batch_{i}.csv")
            group.to_csv(example_path, index=False)
            print(f"Saved example batch to {example_path}", flush=True)

        # Append directly to the final output file
        group.to_csv(final_path, mode="a", index=False, header=not header_written)
        header_written = True
        print(f"Appended results of group {i} to {final_path}", flush=True)



        # Optional: release GPU memory
        torch.cuda.empty_cache()

    print(f"Processing complete — results appended to {final_path}", flush=True)
    
if __name__ == "__main__":
    main()
