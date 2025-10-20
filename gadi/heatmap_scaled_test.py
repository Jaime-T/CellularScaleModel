#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# scaling LLRs from -1 to 1. 
# Plot heatmaps of log likelihood ratios (LLRs) for all possible amino acid substitutions
# across a given protein sequence using ESM-2 models (original and fine-tuned).
# This script generates heatmaps for the original ESM-2 model and a missense fine-tuned model.
# It also compares top-k amino acid predictions at a specific position in the sequence.
# It focuses on RPL15 protein as an example for a negative control. 

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
from sklearn.model_selection import train_test_split

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


def plot_heatmap(data, gene, title, sequence, base_dir, amino_acids):

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
    plt.title(title + ' ' + '650M')
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

def topk_predictions(model, tokenizer, protein_seq, masked_pos, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokens = list(protein_seq)
    tokens[masked_pos - 1] = tokenizer.mask_token  # Replace 1-based index with mask
    masked_seq = "".join(tokens)

    inputs = tokenizer(masked_seq, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        prediction_logits = logits[mask_index]

    probs = torch.softmax(prediction_logits, dim=-1)
    topk = torch.topk(probs, k=k, dim=-1)
    top_tokens = tokenizer.convert_ids_to_tokens(topk.indices[0].tolist())
    top_probs = topk.values[0].tolist()

    return list(zip(top_tokens, top_probs))

def main():

    # Model Parameters (in millions) of finetuned model
    params = 650

    # Sequence and gene of interest 
    #gene = "rpl15"  # Example gene for negative control
    #sequence = "MGAYKYIQELWRKKQSDVMRFLLRVRCWQYRQLSALHRAPRPTRPDKARRLGYKAKQGYVIYRIRVRRGGRKRPVPKGATYGKPVHHGVNQLKFARSLQSVAEERAGRHCGALRVLNSYWVGEDSTYKFFEVILIDPFHKAIRRNPDTQWITKPVHKHREMRGLTSAGRKSRGLGKGHKFHHTIGGSRRAAWRRRNTLQLHRYR"

    sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    gene = "tp53"

    # Load original ESM-2 model
    base_model_name = f"/g/data/gi52/jaime/esm2_{params}M_model"
    base_tokenizer = EsmTokenizer.from_pretrained(base_model_name)
    base_model = EsmForMaskedLM.from_pretrained(base_model_name)

    # Load missense fine-tuned model
    ms_model_path = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/missense/run10/epoch0_batch18000"
    ms_tokenizer = EsmTokenizer.from_pretrained(ms_model_path)
    ms_model = EsmForMaskedLM.from_pretrained(ms_model_path)

    # Load base model separately for comparison
    base_model_fresh = EsmForMaskedLM.from_pretrained(base_model_name)

    base_dir = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/missense/run10/heatmaps_scaled/epoch0_batch18000"
    os.makedirs(base_dir, exist_ok=True)

    print(f"Generating heatmaps for {gene} using ESM2-{params}M models...")

    # Generate heatmaps
    base_heatmap, amino_acids = generate_heatmap(sequence, base_model_fresh, base_tokenizer)
    ms_heatmap, _ = generate_heatmap(sequence, ms_model, ms_tokenizer)

    # Compute difference
    ms_diff_heatmap = ms_heatmap - base_heatmap

    plot_heatmap(params, gene, base_heatmap, "Original ESM2 Model (LLRs)", sequence, base_dir, amino_acids)
    plot_heatmap(params, gene, ms_heatmap, "Fine-tuned Missense Model (LLRs)", sequence, base_dir, amino_acids)
    plot_heatmap(params, gene, ms_diff_heatmap, "Difference (Fine-tuned Missense - Original)", sequence, base_dir, amino_acids)

   
    # Generate heatmap with mutations as dots in positions
    plot_heatmap(ms_diff_heatmap, gene, "Difference (Fine-tuned Missense - Original)", 
                           sequence, base_dir, amino_acids)
    plot_heatmap(ms_heatmap, gene, "Fine-tuned Missense Model", sequence, 
                           base_dir, amino_acids)
    

    # Compare amino acid predictions
    masked_pos = 100

    original_preds = topk_predictions(base_model, base_tokenizer, sequence, masked_pos)
    ms_preds = topk_predictions(ms_model, ms_tokenizer, sequence, masked_pos)
    #fs_preds = topk_predictions(fs_model, fs_tokenizer, sequence, masked_pos)

    print(f"Original model top predictions at position {masked_pos}:", original_preds)
    print(f"Fine-tuned missense model top predictions at position {masked_pos}:", ms_preds)
    #print(f"Fine-tuned frameshift model top predictions at position {masked_pos}:", fs_preds)


if __name__ == '__main__':
    main()