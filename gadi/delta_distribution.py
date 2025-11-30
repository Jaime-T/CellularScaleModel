""" 
Create delta distribution plots (CSM - ESM2) scores for mutations from ClinVar dataset.

"""


import pandas as pd
import pandas as pd
from plotnine import ( ggplot, aes, geom_density, theme_minimal, scale_color_manual, 
    scale_fill_manual, scale_y_continuous)
import os
from sklearn.preprocessing import StandardScaler
import numpy as np


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


def main():
    
    
    # Load dataset with ClinVar labels and mutations, and precomputed esm and csm scores 
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

    
    # filter out only Pathogenic, Benign, Uncertain Significance, and Conflicting classifications of pathogenicity
    filtered_data = df[df['clinvar_label'].isin([
        'Pathogenic', 'Benign', 'Uncertain significance','Conflicting classifications of pathogenicity'])]


    scaled_delta_extremity_mut_distro_plot(filtered_data, save_dir, 10000, gene)


if __name__ == "__main__":
    main()   