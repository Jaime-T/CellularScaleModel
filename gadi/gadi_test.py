#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, EsmForMaskedLM

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

        #print(model(masked_input_ids))

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

def plot_heatmap(data, title, sequence, amino_acids, start_pos=1):
    fig = plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis_r", aspect="auto", vmin=None, vmax=None)
    plt.xticks(range(len(sequence)), list(sequence))
    plt.yticks(range(20), amino_acids)
    plt.xlabel("Position in Protein Sequence")
    plt.ylabel("Amino Acid Mutations")
    plt.title(title)
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.tight_layout()
    return fig
    
  
def main():
     # Load original ESM-2 model
    base_model_name = "facebook/esm2_t33_650M_UR50D"
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = EsmForMaskedLM.from_pretrained(base_model_name)
    sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"

    # Generate
    base_heatmap, amino_acids = generate_heatmap(sequence, base_model, base_tokenizer)
    fig = plot_heatmap(base_heatmap, "Original ESM2 Model (LLRs)", sequence, amino_acids)
    
    output_dir = "/g/data/gi52/jt4401/testfolder"
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)  # Creates it if it doesn't exist
    output_path = os.path.join(output_dir, "myplot.png")
    fig.savefig(output_path, dpi=300)
    print(f"✅ Saved figure to: {output_path}")

if __name__ == '__main__':
    main()