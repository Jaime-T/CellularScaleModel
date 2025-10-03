#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot heatmaps of log likelihood ratios (LLRs) for all possible amino acid substitutions
# across a given protein sequence using ESM-2 models (original and fine-tuned).
# This script generates heatmaps for the original ESM-2 model and a missense fine-tuned model,
# and overlays known mutations as dots on the heatmap. 
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

def load_dataset(path: str) -> TorchDataset:
    """
    Load a TorchDataset from a .pt file, handling PyTorch’s safe-unpickling behavior.
    """
    # whitelist TorchDataset via safe_globals
    torch.serialization.add_safe_globals([TorchDataset])
    ds = torch.load(path)  # uses weights_only=True by default, but TorchDataset is allowlisted

    return ds

def get_mutation_list(dataset: TorchDataset, gene: str) -> list:
    """
    From the dataset, reconstruct the DataFrame and filter for that gene’s protein changes.
    """
    # Convert dict-of-lists into a DataFrame
    df = pd.DataFrame(dataset.data)
    # Filter
    ms_filtered = df[df["HugoSymbol"] == gene.upper()][["ProteinChange"]]
    return ms_filtered["ProteinChange"].tolist()


def parse_hgvs(protein_change):
    """Extract WT AA, position, and mutant AA from HGVS string like 'p.A586V'."""
    pattern = r"^p\.([A-Z])(\d+)(?!Ter)([A-Z])"
    m = re.match(pattern, protein_change)
    if not m:
        return None
    wt, pos, mt = m.groups()
    return wt, int(pos), mt

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

def plot_heatmap_with_dots(data, gene, title, sequence, base_path, amino_acids, mutation_list, start_pos=1):

    plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis_r", aspect="auto", vmin=-20, vmax=20)
    plt.ylabel("Amino Acid Mutations")
    plt.yticks(range(len(amino_acids)), amino_acids)
    plt.xlabel("Position in Protein Sequence")
    seq_len = len(sequence)
    xticks_positions = list(range(0, seq_len, 50)) # mark every 50th position
    # ensure last position is shown too
    if seq_len - 1 not in xticks_positions:
        xticks_positions.append(seq_len - 1)
    # set ticks and labels
    plt.xticks(xticks_positions, [str(pos) for pos in xticks_positions])
    plt.title(f"{title} 650M")
    plt.colorbar(label="LLR Difference (Mutant − Base)")

    # Overlay mutation markers
    marks_x, marks_y = [], []
    for mut in mutation_list:
        parsed = parse_hgvs(mut)
        if parsed:
            _, pos, mt = parsed
            if mt in amino_acids:
                row = amino_acids.index(mt)
                col = pos - start_pos
                if 0 <= col < data.shape[1]:
                    marks_x.append(col)
                    marks_y.append(row)

    if marks_x:
        plt.scatter(marks_x, marks_y, marker='o', color='black', s=50, label='Mutations')
        plt.legend(loc='upper right')

    plt.tight_layout()

    # Define the path
    save_path = os.path.join(base_path, f"{gene}/{title.replace(' ', '_')}.png")
    folder = os.path.dirname(save_path)
    print(f"Saving heatmap with dots to {save_path}")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_heatmap(params, gene, data, title, sequence, base_dir, amino_acids):
    plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis_r", aspect="auto", vmin=-20, vmax=20)
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
    plt.title(title + ' ' + str(params) + 'M')
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
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
    gene = "rpl15"  # Example gene for negative control
    sequence = "MGAYKYIQELWRKKQSDVMRFLLRVRCWQYRQLSALHRAPRPTRPDKARRLGYKAKQGYVIYRIRVRRGGRKRPVPKGATYGKPVHHGVNQLKFARSLQSVAEERAGRHCGALRVLNSYWVGEDSTYKFFEVILIDPFHKAIRRNPDTQWITKPVHKHREMRGLTSAGRKSRGLGKGHKFHHTIGGSRRAAWRRRNTLQLHRYR"

    # Load original ESM-2 model
    base_model_name = f"/g/data/gi52/jaime/esm2_{params}M_model"
    base_tokenizer = EsmTokenizer.from_pretrained(base_model_name)
    base_model = EsmForMaskedLM.from_pretrained(base_model_name)

    # Load missense fine-tuned model
    ms_model_path = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/missense/run7/epoch0"
    ms_tokenizer = EsmTokenizer.from_pretrained(ms_model_path)
    ms_model = EsmForMaskedLM.from_pretrained(ms_model_path)


    base_dir = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/missense/run7/heatmaps2"

    ''' 
    # Load training dataset to see what was used
    base_dir = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/missense/run7"
    fname = "train_dataset.pt"
    full_path = os.path.join(base_dir, fname)
    train_dataset = load_dataset(full_path) 

    # get mutation list
    mut_list = get_mutation_list(train_dataset, gene)
    print(f"Number of missense training mutations for {gene}: {len(mut_list)}")
    print("Mutations for gene", gene, ":", mut_list)
    '''

    # Generate heatmaps
    base_heatmap, amino_acids = generate_heatmap(sequence, base_model, base_tokenizer)
    ms_heatmap, _ = generate_heatmap(sequence, ms_model, ms_tokenizer)

    # Compute difference
    ms_diff_heatmap = ms_heatmap - base_heatmap

    plot_heatmap(params, gene, base_heatmap, "Original ESM2 Model (LLRs)", sequence, base_dir, amino_acids)
    plot_heatmap(params, gene, ms_heatmap, "Fine-tuned Missense Model (LLRs)", sequence, base_dir, amino_acids)
    plot_heatmap(params, gene, ms_diff_heatmap, "Difference (Fine-tuned Missense - Original)", sequence, base_dir, amino_acids)

    # Load mutations and split as it was during training  

    data_path = Path("./data")
    test_size = 0.2
    valid_size = 0.0625

        # missense
    ms_df = pd.read_parquet(data_path / "update2_all_ms_samples.parquet")
    ms_train_df, ms_test_df = train_test_split(ms_df, test_size=test_size, random_state=0)
    ms_train_df, ms_valid_df = train_test_split(ms_train_df, test_size=valid_size, random_state=0)
    ms_filtered = (
        ms_train_df[ms_train_df['HugoSymbol'] == gene.upper()][['ProteinChange']] 
    )
    ms_mutation_list = ms_filtered['ProteinChange'].tolist()

    print(f"Number of missense training mutations for {gene}: {len(ms_mutation_list)}")
    print( "Mutations for gene", gene, ":", ms_mutation_list)
 

    # Generate heatmap with mutations as dots in positions
    plot_heatmap_with_dots(ms_diff_heatmap, gene, "Difference (Fine-tuned Missense - Original) with Mutations", 
                           sequence, base_dir, amino_acids, ms_mutation_list, start_pos=0)
    plot_heatmap_with_dots(ms_heatmap, gene, "Fine-tuned Missense Model with Mutations", sequence, 
                           base_dir, amino_acids, ms_mutation_list, start_pos=0)

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