"""
    Load in joined clinvar and depmap mutatins which have clinvar classifciation label
    Filter for unique mutations
    Load in ESM and CSM model
    Calculate the esm and csm scores for each mutation
    Create a distribution curve using classification labels

    Repeat this for different finetuned models 

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


def extremity_mut_distro_plot(csm_data, save_dir, run_num, xlabel="csm_score"):

    # Standardise the data
    scaler = StandardScaler()
    csm_data = csm_data.copy()  # avoid modifying original
    csm_data[xlabel] = scaler.fit_transform(csm_data[[xlabel]])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Custom color mapping
    color_map = {
        "Pathogenic": "red",
        "Benign": "green",
        "Uncertain significance": "yellow",
        "Conflicting classifications of pathogenicity": "grey"
    }

    # method 1: seaborn kdeplot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=csm_data,
        x=xlabel,
        hue="clinvar_label",
        fill=True,
        alpha=0.3,
        palette=color_map
    )

    plt.title(f"Density of {xlabel} by ClinVar Label")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    sns.despine()

    save_path = os.path.join(save_dir, f"run{run_num}_distro_curve_plot-1.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved to {save_path}")

    # method 2: plotnine (ggplot2-like)
    p = (
        ggplot(csm_data, aes(x=xlabel, color="clinvar_label", fill="clinvar_label"))
        + geom_density(alpha=0.3)
        + theme_minimal()
        + scale_color_manual(values=color_map)
        + scale_fill_manual(values=color_map)
        + theme_minimal()
    )
    save_path = os.path.join(save_dir, f"run{run_num}_distro_curve_plot-2.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")



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

    # Load in ESM and CSM model
    iterations = {
        "run11_batch5000": "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run11/epoch0_batch5000",
        "run11_batch10000": "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run11/epoch0_batch10000",
        "run10_batch5000": "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run10/epoch0_batch5000",
        "run10_batch10000": "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run10/epoch0_batch10000",
        "run9.1_batch5000": "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run9.1/epoch0_batch5000",
        "run8_batch5000": "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run8/epoch0_batch5000",
    }

    # Load ESM2 base model and tokenizer
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    tokenizer = EsmTokenizer.from_pretrained(base_model_path)
    base_model = EsmForMaskedLM.from_pretrained(base_model_path)
    base_model.eval()

 
    # Calculate the CSM scores for each mutation and append to dataframe
    for row in filt_mutations.itertuples():
        mutation = row.ProteinChange
        sequence = row.wt_protein_seq
        idx = row.Index
        if pd.isna(mutation):
            continue
        try:
            csm_score = compute_mut_score(base_model, tokenizer, mutation, sequence)
            df.loc[idx, 'csm_score'] = csm_score
        except ValueError as e:
            print(e)
            df.loc[idx, 'csm_score'] = None

    # Save the final results
    save_path = os.path.join("/g/data/gi52/jaime/clinvar/all_clinvar_esm_scores.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved ESM results to {save_path}")

    # For each iteration, calculate the csm scores for each mutation

    for key, value in iterations.items():
        print(f"{key}: {value}")



        #Create a distribution curve using classification labels




if __name__ == "__main__":
    main()   