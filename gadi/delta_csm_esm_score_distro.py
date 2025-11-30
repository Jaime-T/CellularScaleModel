""" 
Compute and plot the distribution of CSM scores for mutations from ClinVar dataset.
Compare with base ESM2 model scores.

1. Cartesian dot plot of CSM vs ESM2 score 
2. Delta distribution plots (CSM - ESM2) score 

"""

import pandas as pd
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ( ggplot, aes, geom_density, theme_minimal, scale_color_manual, 
    scale_fill_manual, scale_y_continuous)
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

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

def scaled_delta_extremity_mut_distro_plot(csm_data, save_dir, batch_num, gene):

    csm_data = csm_data.copy()  # avoid modifying original

    csm_data['delta_score'] = csm_data['csm_score'] - csm_data['esm_score']

    # Standardize the delta values
    scaler = StandardScaler()
    csm_data["delta_score_scaled"] = scaler.fit_transform(
        csm_data["delta_score"].values.reshape(-1, 1)
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Custom color mapping
    color_map = {
        "Pathogenic": "blue",
        "Benign": "red",
        "Uncertain significance": "orange",
        "Conflicting classifications of pathogenicity": "grey"
    }

    p = (
        ggplot(csm_data, aes(x="delta_score_scaled", color="clinvar_label", fill="clinvar_label"))
        + geom_density(alpha=0.3)
        + theme_minimal()
        + scale_color_manual(values=color_map)
        + scale_fill_manual(values=color_map)
        + theme_minimal()
       # + scale_y_continuous(limits=(0, 5))
    )
    save_path = os.path.join(save_dir, f"scaled_delta_epoch0_batch{batch_num}_{gene}.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")



def cartesian_plot(filtered_data, save_dir, gene):
    "Plot esm score versus csm score scatter plot"

    csm_data = filtered_data.copy()  #
    # Convert columns to numeric, coercing invalid values to NaN
    csm_data["csm_score"] = pd.to_numeric(csm_data["csm_score"], errors="coerce")
    csm_data["esm_score"] = pd.to_numeric(csm_data["esm_score"], errors="coerce")


    # filter out rows with NaN scores
    csm_data = csm_data.dropna(subset=["esm_score", "csm_score"])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   

     # --- Custom colour map for ClinVar labels ---
    custom_palette = {
        'Pathogenic': '#00008B',  # dark blue
        'Pathogenic/Likely pathogenic': '#4169E1',  # medium blue (royal blue)
        'Likely pathogenic': '#87CEFA',  # light blue
        'Benign': '#FF0000',  # red
        'Benign/Likely benign': "#F27F52",  # light red (salmon)
        'Likely benign': "#ED5E76",  # pink
        'Uncertain significance': "#DEB052",  # yellow
        'Conflicting classifications of pathogenicity': '#808080'  # grey
    }

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=csm_data,
        y="esm_score",
        x="csm_score",
        hue="clinvar_label",
        palette=custom_palette,
        alpha=0.7
    )   
    
    # --- Set limits rounded to nearest 5 ---
    min_x = np.floor(csm_data["csm_score"].min() / 5) * 5
    print(f"min_x: {min_x}")
    min_y = np.floor(csm_data["esm_score"].min() / 5) * 5
    print(f"min_y: {min_y}")

    # --- Make axes equal and square ---
    overall_min = np.floor(min(csm_data["csm_score"].min(), csm_data["esm_score"].min()) / 5) * 5
    overall_max = np.ceil(max(csm_data["csm_score"].max(), csm_data["esm_score"].max()) / 5) * 5

    ax.set_xlim(overall_min, overall_max)
    ax.set_ylim(overall_min, overall_max)
    ax.set_aspect('equal', adjustable='box')

    # Invert both axes to make values decrease
    ax.invert_xaxis()
    

    # --- Set ticks every 5 units ---
    ax.set_xticks(np.arange(0, min_x - 5, -5))
    ax.set_yticks(np.arange(0, min_y - 5, -5))

    # --- Keep axes on left and top ---
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Move labels to left/top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    # --- Add grid and labels ---
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.title(f"CSM Score vs ESM Score for {gene.upper()} Missense Mutations")
    plt.ylabel("ESM Score")
    plt.xlabel("CSM Score")

    sns.despine(left=False, top=False)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"esm_vs_csm_scatter_{gene}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

def main():
    
    
    # Load dataset with ClinVar labels and mutations, and precomputed esm and csm scores 
    #tp53_clinvar = pd.read_csv("/g/data/gi52/jaime/clinvar/run11_ms/tp53/epoch0_batch10000_tp53_clinvar_csm_scores.csv")
    #all_genes_clinvar_intermediate = pd.read_csv("/g/data/gi52/jaime/clinvar/run11_ms/batched7_all_genes/all_clinvar_csm_scores.csv")
    #data = pd.read_csv("/g/data/gi52/jaime/clinvar/run11_ms/five_panlethal_genes/five_panlethal_genes_clinvar_csm_scores.csv", na_values=["NaN", "nan", "None", ""])
    #data = pd.read_csv("/g/data/gi52/jaime/clinvar/run11_ms/batched_ten_genes/ten_genes_clinvar_csm_scores.csv")
    data = pd.read_csv("/g/data/gi52/jaime/clinvar/run11_ms/rpl15/rpl15_esm_csm_scores_fixed.csv")
    
    gene = "rpl15"

    # save directory
    save_dir = f"/g/data/gi52/jaime/clinvar/run11_ms/{gene}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    df = data.copy()

    df = df[df['GeneSymbol'] == gene.upper()]
    print(f"Number of {gene} mutations in file: {len(df)}")

    # rename columns for plotting
    df = df.rename(
        columns={"ClinicalSignificance": "clinvar_label"}
    )

    # filter out only likely Pathogenic, Benign, Uncertain Significance, and Conflicting classifications of pathogenicity
    filtered_data = df[df['clinvar_label'].isin([
        'Pathogenic', 'Benign', 'Uncertain significance','Conflicting classifications of pathogenicity',
        'Pathogenic/Likely pathogenic', 'Likely pathogenic', 'Benign/Likely benign', 'Likely benign'])]
    
    cartesian_plot(filtered_data, save_dir, gene)
    
    # filter out only Pathogenic, Benign, Uncertain Significance, and Conflicting classifications of pathogenicity
    filtered_data = df[df['clinvar_label'].isin([
        'Pathogenic', 'Benign', 'Uncertain significance','Conflicting classifications of pathogenicity'])]

    scaled_delta_extremity_mut_distro_plot(filtered_data, save_dir, 10000, gene)


if __name__ == "__main__":
    main()   