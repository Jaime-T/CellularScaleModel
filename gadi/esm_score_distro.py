""" 
Compute and plot the distribution of CSM scores for TP53 mutations from ClinVar dataset.
Compare with base ESM2 model scores.

For missense run10 - just DepMap missense data (unique protein mutations)
"""


import pandas as pd
from peft import PeftModel, PeftConfig
from transformers import EsmForMaskedLM, EsmTokenizer
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ( ggplot, aes, geom_density, theme_minimal, scale_color_manual, 
    scale_fill_manual, xlim, ylim)
import re
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

def mut_distro_plot(csm_data, save_dir, xlabel="csm_score", num=1):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # method 1: seaborn kdeplot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=csm_data,
        x=xlabel,
        hue="clinvar_label",
        fill=True,
        alpha=0.3
    )

    plt.title(f"Density of {xlabel} by ClinVar Label")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    sns.despine()
    save_path = os.path.join(save_dir, f"tp53_{xlabel}_plot{num}-1.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved to {save_path}")

    # method 2: plotnine (ggplot2-like)
    p = (
        ggplot(csm_data, aes(x=xlabel, color="clinvar_label", fill="clinvar_label"))
        + geom_density(alpha=0.3)
        + theme_minimal()
    )
    save_path = os.path.join(save_dir, f"tp53_{xlabel}_plot{num}-2.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")

def extremity_mut_distro_plot(csm_data, save_dir, xlabel="csm_score", num=1):

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

    save_path = os.path.join(save_dir, f"tp53_{xlabel}_plot{num}-1.png")
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
    save_path = os.path.join(save_dir, f"tp53_{xlabel}_plot{num}-2.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")



def main():
    
    # Load ESM2 base model and tokenizer
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    tokenizer = EsmTokenizer.from_pretrained(base_model_path)
    base_model = EsmForMaskedLM.from_pretrained(base_model_path)
    base_model.eval()

    # Load dataset with ClinVar labels and mutations, and precomputed esm scores 

    #tp53_clinvar = pd.read_csv("/g/data/gi52/jaime/clinvar/clinvar_tp53_mutations_1letter.csv")
    tp53_clinvar = pd.read_csv("/g/data/gi52/jaime/clinvar/tp53_esm2_baseline_scores.csv")
    df = tp53_clinvar.copy()

    # rename columns for plotting
    csm_data = df.rename(
        columns={"Germline.classification": "clinvar_label"}
    )

        
    save_dir = f"/g/data/gi52/jaime/clinvar"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # filter out only Pathogenic, Benign, and Uncertain Significance
    filtered_data = csm_data[csm_data['clinvar_label'].isin(['Pathogenic', 'Benign', 'Uncertain significance', 'Conflicting classifications of pathogenicity'])]

    extremity_mut_distro_plot(filtered_data, save_dir, xlabel="esm_score", num=5)

    # delete models to free up memory
    del base_model
    torch.cuda.empty_cache()



if __name__ == "__main__":
    main()   