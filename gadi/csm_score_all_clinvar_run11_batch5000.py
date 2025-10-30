"""
    Load in joined clinvar and depmap mutatins which have clinvar classifciation label
    Filter for unique mutations
    Load in ESM model
    Calculate the esm scores for each mutation
    
"""

import pandas as pd
from peft import PeftModel
from transformers import EsmForMaskedLM, EsmTokenizer
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ( ggplot, aes, geom_density, theme_minimal, scale_color_manual, 
    scale_fill_manual)
import os
from sklearn.preprocessing import StandardScaler
import re

def compute_mut_score(model, tokenizer, mutation, sequence):
    model.eval()

    wt, pos, mt = mutation[0], int(mutation[1:-1]), mutation[-1]
    # Adjust for 0-based indexing
    pos -= 1
    if sequence[pos] != wt:
        raise ValueError(f"Wildtype mismatch at position {pos+1} for mutation {mutation}: expected {sequence[pos]}, found {wt}")
    
    # Tokenize the sequence
    encoded = tokenizer(sequence, return_tensors="pt")
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
    return score


def main():

    print('hi')
    # Load in joined clinvar and depmap mutatins which have clinvar classifciation label
    df = pd.read_csv("/g/data/gi52/jaime/data/clinvar_depmap_joined.csv")
    #df = pd.read_csv("/home/cciamr.local/jtaitz/R_Drive/DDC/jaime/CellularScaleModel/data/clinvar_depmap_joined.csv")
    mutations = df.copy()
    print(f'Number of entries: {len(mutations)}')

    # Filter for unique mutations
    uni_mutations = mutations.drop_duplicates(subset=['Name', 'ClinicalSignificance'])
    print(f"Number of unique entries: {len(uni_mutations)}")

    # Filter for mutations that have label: Pathogenic, Benign, Uncertain Significance, and Conflicting classifications of pathogenicity
    filt_mutations = uni_mutations[uni_mutations['ClinicalSignificance'].isin(['Pathogenic', 'Benign', 'Uncertain significance','Conflicting classifications of pathogenicity'])]
    print(f"Number of filtered labeled entries: {len(filt_mutations)}")


    # Load ESM2 base model and tokenizer
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    tokenizer = EsmTokenizer.from_pretrained(base_model_path)
    base_model = EsmForMaskedLM.from_pretrained(base_model_path)
    base_model.eval()

    # load adapter
    iteration = "epoch0_batch5000"
    run = "run11_ms"
    adapter_path = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run11/epoch0_batch5000"
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False )

    # Merge the adapter weights into the base model
    csm_model = model.merge_and_unload()
    csm_model.eval()     

    # calculate the esm scores for each mutation 
    for row in filt_mutations.itertuples():
        mutation = row.ProteinChange
        mutation = re.sub(r"^p\.", "", mutation)

        sequence = row.wt_protein_seq
        idx = row.Index
        if pd.isna(mutation):
            continue
        try:
            csm_score = compute_mut_score(base_model, tokenizer, mutation, sequence)
            filt_mutations.loc[idx, 'csm_score'] = csm_score
        except ValueError as e:
            print(e)
            filt_mutations.loc[idx, 'csm_score'] = None

        # save intermediate results
        if idx % 10000 == 0:
            slct_mutations = filt_mutations[['Name','HGNC_ID', 'ClinicalSignificance', 'ProteinChange', 'wt_protein_seq', 'csm_score']]
            print(f"Processed {idx} mutations, saving intermediate results...")
            slct_mutations.to_csv(f"/g/data/gi52/jaime/clinvar/{run}/all_clinvar_esm_scores_intermediate{idx}.csv", index=False)


    # Save the final results
    save_path = os.path.join(f"/g/data/gi52/jaime/clinvar/{run}/{iteration}_all_clinvar_esm_scores.csv")
    filt_mutations.to_csv(save_path, index=False)
    print(f"Saved CSM results to {save_path}")



if __name__ == "__main__":
    main()   