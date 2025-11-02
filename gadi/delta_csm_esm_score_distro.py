""" 
Compute and plot the distribution of CSM scores for TP53 mutations from ClinVar dataset.
Compare with base ESM2 model scores.

For missense run11 - 7:1 MT to WT batches of 8
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

def extremity_mut_distro_plot(csm_data, save_dir, batch_num, xlabel="csm_score"):

    # Standardise the data
    scaler = StandardScaler()
    csm_data = csm_data.copy()  # avoid modifying original
    csm_data[xlabel] = scaler.fit_transform(csm_data[[xlabel]])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Custom color mapping
    color_map = {
        "Pathogenic": "blue",
        "Benign": "red",
        "Uncertain significance": "orange",
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

    save_path = os.path.join(save_dir, f"epoch0_batch{batch_num}_tp53_{xlabel}_plot-1.png")
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
    save_path = os.path.join(save_dir, f"epoch0_batch{batch_num}_tp53_{xlabel}_plot-2.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")


def scaled_delta_extremity_mut_distro_plot(csm_data, save_dir, batch_num):

    
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

    # method 1: seaborn kdeplot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=csm_data,
        x="delta_score_scaled",
        hue="clinvar_label",
        fill=True,
        alpha=0.3,
        palette=color_map
    )

    plt.title(f"Density of Scaled Δ(CSM - ESM) by ClinVar Label")
    plt.xlabel("Scaled CSM - ESM score")
    plt.ylabel("Density")
    sns.despine()

    save_path = os.path.join(save_dir, f"scaled_delta_epoch0_batch{batch_num}_tp53_plot-1.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved to {save_path}")

    # method 2: plotnine (ggplot2-like)
    p = (
        ggplot(csm_data, aes(x="delta_score_scaled", color="clinvar_label", fill="clinvar_label"))
        + geom_density(alpha=0.3)
        + theme_minimal()
        + scale_color_manual(values=color_map)
        + scale_fill_manual(values=color_map)
        + theme_minimal()
    )
    save_path = os.path.join(save_dir, f"scaled_delta_epoch0_batch{batch_num}_tp53_plot-2.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")


def delta_extremity_mut_distro_plot(csm_data, save_dir, batch_num):
    """
    Plot density distributions of (CSM Score - ESM Score) by ClinVar label,
    using both seaborn and plotnine.
    """

    csm_data = csm_data.copy()  # Avoid modifying original

    # Compute delta
    csm_data["delta_score"] = csm_data["csm_score"] - csm_data["esm_score"]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Custom color mapping
    color_map = {
        "Pathogenic": "blue",
        "Benign": "red",
        "Uncertain significance": "orange",
        "Conflicting classifications of pathogenicity": "grey"
    }

    # --- Method 1: seaborn kdeplot ---
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=csm_data,
        x="delta_score",
        hue="clinvar_label",
        fill=True,
        alpha=0.3,
        palette=color_map
    )

    plt.title("Density of (CSM Score - ESM Score) by ClinVar Label")
    plt.xlabel("CSM Score - ESM Score")
    plt.ylabel("Density")
    sns.despine()

    save_path = os.path.join(save_dir, f"delta_epoch0_batch{batch_num}_tp53_plot-1.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved seaborn plot to {save_path}")

    # --- Method 2: plotnine (ggplot2 style) ---
    p = (
        ggplot(csm_data, aes(x="delta_score", color="clinvar_label", fill="clinvar_label"))
        + geom_density(alpha=0.3)
        + theme_minimal()
        + scale_color_manual(values=color_map)
        + scale_fill_manual(values=color_map)
        + theme_minimal()
    )

    save_path = os.path.join(save_dir, f"delta_epoch0_batch{batch_num}_tp53_plot-2.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved plotnine plot to {save_path}")

def cartesian_plot(filtered_data, save_dir):
    "Plot esm score versus csm score scatter plot"

    csm_data = filtered_data.copy()  #

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=csm_data,
        y="esm_score",
        x="csm_score",
        hue="clinvar_label",
        alpha=0.7
    )   
    
    # --- Set limits rounded to nearest 5 ---
    min_x = np.floor(csm_data["csm_score"].min() / 5) * 5
    min_y = np.floor(csm_data["esm_score"].min() / 5) * 5

    # --- Set limits rounded to nearest 5 ---
    max_x = np.ceil(csm_data["csm_score"].max() / 5) * 5
    max_y = np.ceil(csm_data["esm_score"].max() / 5) * 5

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
    plt.title("TP53 Gene: CSM Score vs ESM Score by ClinVar Label")
    plt.ylabel("ESM Score")
    plt.xlabel("CSM Score")

    sns.despine(left=False, top=False)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"tp53_esm_vs_csm_scatter_grid_filt.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

def main():
    
    
    # Load dataset with ClinVar labels and mutations, and precomputed esm and csm scores 
    tp53_clinvar = pd.read_csv("/g/data/gi52/jaime/clinvar/run11_ms/tp53/epoch0_batch10000_tp53_clinvar_csm_scores.csv")

    # save directory
    save_dir = f"/g/data/gi52/jaime/clinvar/run11_ms/tp53"
    
    # rename columns for plotting
    df = tp53_clinvar.copy()
    df = df.rename(
        columns={"Germline.classification": "clinvar_label"}
    )

    # filter out only Pathogenic, Benign, Uncertain Significance, and Conflicting classifications of pathogenicity
    filtered_data = df[df['clinvar_label'].isin([
        'Pathogenic', 'Benign', 'Uncertain significance','Conflicting classifications of pathogenicity',
        'Pathogenic/Likely pathogenic', 'Likely pathogenic', 'Benign/Likely benign', 'Likely benign'])]
    
    cartesian_plot(filtered_data, save_dir)

    delta_extremity_mut_distro_plot(filtered_data, save_dir, 10000)
    scaled_delta_extremity_mut_distro_plot(filtered_data, save_dir, 10000)

    



if __name__ == "__main__":
    main()   