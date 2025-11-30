#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Generate scatterplots comparing ESM and CSM scores for any given 
    protein for all possible amino acid substitutions.
"""

import re 
import os
import tempfile
import pandas as pd
import numpy as np
tmpdir = os.getenv('TMPDIR', tempfile.gettempdir())
mpl_cache = os.path.join(tmpdir, 'matplotlib-cache')
os.makedirs(mpl_cache, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_cache
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset 
from transformers import EsmForMaskedLM, EsmTokenizer
from peft import PeftModel
import matplotlib.colors as mcolors
import matplotlib.image as mpimg

class TorchDataset(Dataset):
    def __init__(self, data):
        """
        data: can be a pandas DataFrame or a dictionary of lists
        """
        if hasattr(data, "to_dict"):  # Convert DataFrame to dict
            data = data.to_dict(orient="list")
        self.data = data

    def __len__(self):
        # Return number of samples
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        # Return one sample as a dictionary of tensors
        return {
            key: torch.tensor(self.data[key][idx]) for key in self.data
        }


def generate_heatmap(protein_sequence, model, tokenizer, start_pos=1, end_pos=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    decoded = tokenizer(protein_sequence, return_tensors="pt").to(device)
    input_ids = decoded['input_ids']
    sequence_length = input_ids.shape[1] - 2

    if end_pos is None:
        end_pos = sequence_length

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    heatmap = np.zeros((20, end_pos - start_pos + 1))

    for position in range(start_pos, end_pos + 1):
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, position] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(masked_input_ids).logits
            probabilities = torch.nn.functional.softmax(logits[0, position], dim=0)
            log_probabilities = torch.log(probabilities)

        wt_residue = input_ids[0, position].item()
        log_prob_wt = log_probabilities[wt_residue].item()

        for i, aa in enumerate(amino_acids):
            aa_id = tokenizer.convert_tokens_to_ids(aa)
            log_prob_mt = log_probabilities[aa_id].item()
            heatmap[i, position - start_pos] = log_prob_mt - log_prob_wt

    return heatmap, amino_acids

def custom_plot_heatmap(data, gene, title, sequence, base_dir, amino_acids):

    # Define the custom colormap
    colors = [
        (-20, "orange"),
        (-1, "red"),
        (0, "white"),
        (1, "blue"),
        (20, "cyan")
    ]

    # Create segmented colormap and normaliser
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_diverging", [c for _, c in colors])
    norm = mcolors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)

    plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    plt.yticks(range(20), amino_acids)
    plt.ylabel("Amino Acid Mutations")

    seq_len = len(sequence)
    xticks_positions = list(range(0, seq_len, 50)) # mark every 50th position
    # ensure last position is shown too
    if seq_len - 1 not in xticks_positions:
        xticks_positions.append(seq_len - 1)
    # set ticks and labels
    plt.xticks(xticks_positions, [str(pos) for pos in xticks_positions])
    plt.xlabel("Position in Protein Sequence")
    plt.title(f"{title} for {gene.upper()}")
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.tight_layout()
    
    # Define the path
    save_path = os.path.join(base_dir, f"{gene}/{title.replace(' ', '_')}.png")
    folder = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300)
    print(f"Saved {gene} heatmap to {save_path}")
    plt.close() 

def scaled_plot_heatmap(data, gene, title, sequence, base_dir, amino_acids):

    # scaling data to -1 to 1
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max != data_min:
        data = 2 * (data - data_min) / (data_max - data_min) - 1
    else:
        print("Warning: Heatmap has constant values. Skipping scaling.")

    plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis_r", aspect="auto")
    plt.yticks(range(20), amino_acids)
    plt.ylabel("Amino Acid Mutations")

    seq_len = len(sequence)
    xticks_positions = list(range(0, seq_len, 50)) # mark every 50th position
    # ensure last position is shown too
    if seq_len - 1 not in xticks_positions:
        xticks_positions.append(seq_len - 1)
    # set ticks and labels
    plt.xticks(xticks_positions, [str(pos) for pos in xticks_positions])
    plt.xlabel("Position in Protein Sequence")
    plt.title(f"{title} for {gene.upper()}")
    plt.colorbar(label="Log Likelihood Ratio (LLR) with Standardised Scale -1 to 1")
    plt.tight_layout()
    
    # Define the path
    save_path = os.path.join(base_dir, f"{gene}/{title.replace(' ', '_')}.png")
    folder = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    print(f"Saving heatmap to {save_path}")
    os.makedirs(folder, exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()


def cartesian_plot(base_heatmap, ms_heatmap, save_dir, gene):
    """
    Plot ESM (base model) score versus CSM score scatter plot.
    Each point represents a specific amino acid substitution at a given position.
    """
    # Flatten both matrices (20 x sequence_length)
    y = base_heatmap.flatten()  # ESM scores
    x = ms_heatmap.flatten()    # CSM scores

    # Filter out NaNs (if any)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    # --- Create plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.7, s=10)

    # --- Axis limits ---
    min_val = np.floor(min(x.min(), y.min()) / 5) * 5 # Round down to nearest 5
    max_val = np.ceil(max(x.max(), y.max()) / 5) * 5 # Round up to nearest 5
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')

     # Invert both axes to make values decrease
    ax.invert_xaxis()

    # --- Ticks every 5 units ---
    tick_step = 5
    ax.set_xticks(np.arange(min_val, max_val+tick_step, tick_step))
    ax.set_yticks(np.arange(min_val, max_val+tick_step, tick_step))

    # --- Axis styling ---
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- Labels and title ---
    plt.xlabel("CSM Score")
    plt.ylabel("ESM Score")
    plt.title(f"CSM Score vs ESM Score for {gene.upper()}")

    # Plot the y=x line
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-')
    plt.axvline(x=-5, color='red', linestyle='--')


    # --- Save figure ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cartesian_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {save_path}")

def main():

    # Model Parameters (in millions) of finetuned model
    params = 650

    # Sequence and gene of interest 
    rpl15_gene = "rpl15"  # Example gene for negative control
    rpl15_sequence = "MGAYKYIQELWRKKQSDVMRFLLRVRCWQYRQLSALHRAPRPTRPDKARRLGYKAKQGYVIYRIRVRRGGRKRPVPKGATYGKPVHHGVNQLKFARSLQSVAEERAGRHCGALRVLNSYWVGEDSTYKFFEVILIDPFHKAIRRNPDTQWITKPVHKHREMRGLTSAGRKSRGLGKGHKFHHTIGGSRRAAWRRRNTLQLHRYR"

    tp53_sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    tp53_gene = "tp53"

    # myc
    myc_gene = "myc"
    myc_sequence = "MDFFRVVENQQPPATMPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPSEDIWKKFELLPTPPLSPSRRSGLCSPSYVAVTPFSLRGDNDGGGGSFSTADQLEMVTELLGGDMVNQSFICDPDDETFIKNIIIQDCMWSGFSAAAKLVSEKLASYQAARKDSGSPNPARGHSVCSTSSLYLQDLSAAASECIDPSVVFPYPLNDSSSPKSCASQDSSAFSPSSDSLLSSTESSPQGSPEPLVLHEETPPTTSSDSEEEQEDEEEIDVVSVEKRQAPGKRSESGSPSAGGHSKPPHSPLVLKRCHVSTHQHNYAAPPSTRKDYPAAKRVKLDSVRVLRQISNNRKCTSPRSSDTEENVKRRTHNVLERQRRNELKRSFFALRDQIPELENNEKAPKVVILKKATAYILSVQAEEQKLISEEDLLRKRREQLKHKLEQLRNSCA"
   
    # rpl2 - gene for negative control
    rpl2_gene = "rpl2"  
    rpl2_sequence = "MILKKYKPTTPSLRGLVQIDRSLLWKGDPVKKLTVGMIESAGRNNTGRITVYHRGGGHKTKYRYIDFKRSNYNIPGIVERLEYDPNRTCFIALIKDNENNFSYILAPHDLKVGDTVITGNDIDIRIGNTLPLRNIPIGTMIHNIELNPGKGGKIVRSAGSSAQLISKDENGFCMLKLPSGEYRLFPNNSLATIGILSNIDNKNIKIGKAGRSRWMGRRPIVRGVAMNPVDHPHGGGEGKTSGGRPSVTPWSWPTKGQPTRSKRKYNKLIVQRAKKKI"

    # NEW PROTEIN TEST
    psma3_gene = "psma3"
    psma3_sequence = "MSSIGTGYDLSASTFSPDGRVFQVEYAMKAVENSSTAIGIRCKDGVVFGVEKLVLSKLYEEGSNKRLFNVDRHVGMAVAGLLADARSLADIAREEASNFRSNFGYNIPLKHLADRVAMYVHAYTLYSAVRPFGCSFMLGSYSVNDGAQLYMIDPSGVSYGYWGCAIGKARQAAKTEIEKLQMKEMTCRDIVKEVAKIIYIVHDEVKDKAFELELSWVGELTNGRHEIVPKDIREEAEKYAKESLKEEDESDDDNM"

    gene_dict = {
        rpl15_gene: rpl15_sequence,
        tp53_gene: tp53_sequence,
        myc_gene: myc_sequence,
        rpl2_gene: rpl2_sequence,
        psma3_gene: psma3_sequence,
    }
  

    # Load original ESM-2 model
    base_model_name = f"/g/data/gi52/jaime/esm2_{params}M_model"
    tokenizer = EsmTokenizer.from_pretrained(base_model_name)

    for batch_num in range(10000, 10001, 1000):

        print(f"Processing batch number: {batch_num}")
        
        base_model = EsmForMaskedLM.from_pretrained(base_model_name)

        # Load missense fine-tuned model
        ms_adapter_path = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/missense/run11/epoch0_batch{batch_num}"
        # Load the adapter into the model
        ms_model = PeftModel.from_pretrained(base_model, ms_adapter_path, is_trainable=False)

        # Merge the adapter weights into the base model
        csm_model = ms_model.merge_and_unload()
        csm_model.eval()

        # Load base model separately for comparison
        base_model_fresh = EsmForMaskedLM.from_pretrained(base_model_name)

        base_dir = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/missense/run11/scatterplots"
        os.makedirs(base_dir, exist_ok=True)

        for gene in gene_dict.keys():
            save_path = os.path.join(base_dir, gene)
            os.makedirs(save_path, exist_ok=True)

            sequence = gene_dict[gene]

            print(f"Generating scatterplots for {gene} using ESM2-{params}M models with sequence {sequence}...")

            # Generate heatmaps
            base_heatmap, amino_acids = generate_heatmap(sequence, base_model_fresh, tokenizer)
            ms_heatmap, _ = generate_heatmap(sequence, csm_model, tokenizer)

            cartesian_plot(base_heatmap, ms_heatmap, save_path, gene)

        print(f"[Batch {batch_num}] Heatmaps generated and saved to {base_dir}")

        # delete models to free up memory
        del csm_model, ms_model, base_model, base_model_fresh
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()