#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from transformers import EsmForMaskedLM, EsmTokenizer
from sklearn.model_selection import train_test_split
from peft import PeftModel
import matplotlib.colors as mcolors

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

def plot_heatmap_with_dots(data, descr, gene, title, sequence, amino_acids, mutation_list, start_pos=1):
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
    plt.xticks(range(len(sequence)), list(sequence))
    plt.yticks(range(len(amino_acids)), amino_acids)
    plt.xlabel("Position in Protein Sequence")
    plt.ylabel("Amino Acid Mutations")
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
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/run4/heatmaps/{gene}/{title.replace(' ', '_')}.png"
    folder = os.path.dirname(save_path)

    os.makedirs(folder, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_heatmap(descr, params, gene, data, title, sequence, amino_acids):
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

    plt.title(title + ' ' + str(params) + 'M')
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.tight_layout()
    
    # Define the path
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/run4/heatmaps/{gene}/{title.replace(' ', '_')}.png"
    folder = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Create the directory if it doesn't exist
    print(f"Saving heatmap to {save_path}")

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()

def scaled_plot_heatmap(descr, params, gene, data, title, sequence, amino_acids):

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
    plt.title(title + ' ' + str(params) + 'M')
    plt.colorbar(label="Log Likelihood Ratio (LLR) with Standardised Scale")
    plt.tight_layout()
    
    # Define the path
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/run4/heatmaps/{gene}/{title.replace(' ', '_')}.png"
    folder = os.path.dirname(save_path)


    # Create the directory if it doesn't exist
    print(f"Saving heatmap to {save_path}")
    os.makedirs(folder, exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300)
    plt.close()

def custom_plot_heatmap(descr, params, gene, data, title, sequence, amino_acids):

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
    plt.title(title + ' 650M')
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.tight_layout()
    
    # Define the path
    # Define the path
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/run4/heatmaps/{gene}/{title.replace(' ', '_')}.png"
    folder = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300)
    print(f"Saved {gene} heatmap to {save_path}")
    plt.close() 

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
    gene = "tp53"
    sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"

    # Load original ESM-2 model
    base_model_name = f"/g/data/gi52/jaime/esm2_{params}M_model"
    base_tokenizer = EsmTokenizer.from_pretrained(base_model_name)
    base_model = EsmForMaskedLM.from_pretrained(base_model_name)

    # Load missense fine-tuned model
    ms_model_path = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/missense/run4/epoch2"
    ms_tokenizer = EsmTokenizer.from_pretrained(ms_model_path)
    ms_model = EsmForMaskedLM.from_pretrained(ms_model_path)

    # NEW!!: 
    ms_model_merge = PeftModel.from_pretrained(base_model, ms_model_path, is_trainable=False )
    # merge adapter into base
    ms_model = ms_model_merge.merge_and_unload()
    ms_model.eval()

    # Load base model separately for comparison
    base_model_fresh = EsmForMaskedLM.from_pretrained(base_model_name)


    # Load Frameshift fine-tuned model
    fs_model_path = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/frameshift/run4/final_merged"
    fs_tokenizer = EsmTokenizer.from_pretrained(fs_model_path)
    fs_model = EsmForMaskedLM.from_pretrained(fs_model_path)


    # Generate heatmaps
    base_heatmap, amino_acids = generate_heatmap(sequence, base_model_fresh, base_tokenizer)
    ms_heatmap, _ = generate_heatmap(sequence, ms_model, ms_tokenizer)
    fs_heatmap, _ = generate_heatmap(sequence, fs_model, fs_tokenizer)

    # Compute difference
    ms_diff_heatmap = ms_heatmap - base_heatmap
    fs_diff_heatmap = fs_heatmap - base_heatmap

    plot_heatmap("original", params, gene, base_heatmap, "Original ESM2 Model (LLRs)", sequence, amino_acids)

    plot_heatmap("missense", params, gene, ms_heatmap, "Fine-tuned Missense Model (LLRs)", sequence, amino_acids)
    plot_heatmap("missense", params, gene, ms_diff_heatmap, "Difference (Fine-tuned Missense - Original)", sequence, amino_acids)

    plot_heatmap("frameshift", params, gene, fs_heatmap, "Fine-tuned Frameshift Model (LLRs)", sequence, amino_acids)
    plot_heatmap("frameshift", params, gene, fs_diff_heatmap, "Difference (Fine-tuned Frameshift - Original)", sequence, amino_acids)

    # NEW!! plot scaled heatmaps
    scaled_plot_heatmap("original", params, gene, base_heatmap, "Scaled Original ESM2 Model (LLRs)", sequence, amino_acids)
    scaled_plot_heatmap("missense", params, gene, ms_heatmap, "Scaled Fine-tuned Missense Model (LLRs)", sequence, amino_acids)
    scaled_plot_heatmap("missense", params, gene, ms_diff_heatmap, "Scaled Difference (Fine-tuned Missense - Original)", sequence, amino_acids)

    custom_plot_heatmap("frameshift", params, gene, fs_heatmap, "Scaled Fine-tuned Frameshift Model (LLRs)", sequence, amino_acids)
    custom_plot_heatmap("frameshift", params, gene, fs_diff_heatmap, "Scaled Difference (Fine-tuned Frameshift - Original)", sequence, amino_acids)


    # Load mutations and split as it was during training  
    data_path = Path("./data")
    test_size = 0.2
    valid_size = 0.25

        # missense
    ms_df = pd.read_parquet(data_path / "update2_all_ms_samples.parquet")
    ms_train_df, ms_test_df = train_test_split(ms_df, test_size=test_size, random_state=0)
    ms_train_df, ms_valid_df = train_test_split(ms_train_df, test_size=valid_size, random_state=0)
    ms_filtered = (
        ms_train_df[ms_train_df['HugoSymbol'] == gene.upper()][['ProteinChange']] 
    )
    ms_mutation_list = ms_filtered['ProteinChange'].tolist()

        # frameshift 
    fs_df = pd.read_parquet(data_path / "update2_all_fs_samples.parquet")
    fs_train_df, fs_test_df = train_test_split(fs_df, test_size=test_size, random_state=0)
    fs_train_df, fs_valid_df = train_test_split(fs_train_df, test_size=valid_size, random_state=0)

    fs_filtered = (
        fs_train_df[fs_train_df['HugoSymbol'] == gene.upper()][['ProteinChange']] 
    )
    fs_mutation_list = fs_filtered['ProteinChange'].tolist()

    # Generate heatmap with mutations as dots in positions
    plot_heatmap_with_dots(ms_diff_heatmap, "missense", gene, "Difference (Fine-tuned Missense - Original) with Mutations", sequence, amino_acids, ms_mutation_list, start_pos=0)

    plot_heatmap_with_dots(fs_diff_heatmap, "frameshift", gene, "Difference (Fine-tuned Frameshift - Original) with Mutations", sequence, amino_acids, fs_mutation_list, start_pos=0)


    # Compare amino acid predictions
    masked_pos = 74

    original_preds = topk_predictions(base_model_fresh, base_tokenizer, sequence, masked_pos)
    ms_preds = topk_predictions(ms_model, ms_tokenizer, sequence, masked_pos)
    fs_preds = topk_predictions(fs_model, fs_tokenizer, sequence, masked_pos)

    print(f"Original model top predictions at position {masked_pos}:", original_preds)
    print(f"Fine-tuned missense model top predictions at position {masked_pos}:", ms_preds)
    print(f"Fine-tuned frameshift model top predictions at position {masked_pos}:", fs_preds)


if __name__ == '__main__':
    main()