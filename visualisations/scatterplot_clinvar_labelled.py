""" 
Compute and plot the distribution of CSM scores for mutations from ClinVar dataset.
Compare with base ESM2 model scores. Includes silent mutations e.g. p.V25=

1. Cartesian dot plot of CSM vs ESM2 score 

"""

import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

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

    save_path = os.path.join(save_dir, f"{gene}_clinvar_esm_vs_csm_scatter.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

def main():
    
    # Load dataset with ClinVar labels and mutations, and precomputed esm and csm scores 
    data = pd.read_csv("/g/data/gi52/jaime/clinvar/run11_ms/batched_ten_genes_update/ten_genes_fixed.csv")

    # get genes in dataset
    genes = data['GeneSymbol'].unique().tolist()
    print(f"Genes in dataset: {genes}")

    # save directory
    save_dir = f"/g/data/gi52/jaime/clinvar/run11_ms/batched_ten_genes_update"
   
    for gene in genes:
        print(f"Generating plots for gene: {gene}")

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
    

if __name__ == "__main__":
    main()   