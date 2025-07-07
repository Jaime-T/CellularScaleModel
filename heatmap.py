#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, EsmModel, AutoModelForMaskedLM
from transformers import EsmForMaskedLM, EsmTokenizer 


def generate_heatmap(protein_sequence, model, tokenizer, start_pos=1, end_pos=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(protein_sequence, return_tensors="pt").to(device)
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

def plot_heatmap(data, title, sequence, amino_acids, start_pos=1):
    plt.figure(figsize=(15, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis", aspect="auto", vmin=None, vmax=None)
    plt.xticks(range(len(sequence)), list(sequence))
    plt.yticks(range(20), amino_acids)
    plt.xlabel("Position in Protein Sequence")
    plt.ylabel("Amino Acid Mutations")
    plt.title(title)
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.tight_layout()
    
    plt.savefig(f"heatmaps/missense_tst/{title.replace(' ', '_')}.png", dpi=300)

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

    # Load missense fine-tuned model
    ms_model_path = "./train/model_missense"
    ms_tokenizer = EsmTokenizer.from_pretrained(ms_model_path)
    ms_model = EsmForMaskedLM.from_pretrained(ms_model_path)

    # Load frameshift fine-tuned model
    fs_model_path = "./train/model_frameshift"
    fs_tokenizer = EsmTokenizer.from_pretrained(fs_model_path)
    fs_model = EsmForMaskedLM.from_pretrained(fs_model_path)

    # Load original ESM-2 model
    base_model_name = "facebook/esm2_t6_8M_UR50D"
    base_tokenizer = EsmTokenizer.from_pretrained(base_model_name)
    base_model = EsmForMaskedLM.from_pretrained(base_model_name)

    #sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ" random
    #sequence = "QQSELQPPAPSEDIWKKFELLPTPPLSPSRRSGLCSPSYVAVTPFSLRGD" 51-100 
    #sequence = "MDFFRVVENQQPPATMPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPSEDIWKKFELLPTPPLSPSRRSGLCSPSYVAVTPFSLRGDNDGGGGSFSTADQLEMVTELLGGDMVNQSFICDPDDETFIKNIIIQDCMWSGFSAAAKLVSEKLASYQAARKDSGSPNPARGHSVCSTSSLYLQDLSAAASECIDPSVVFPYPLNDSSSPKSCASQDSSAFSPSSDSLLSSTESSPQGSPEPLVLHEETPPTTSSDSEEEQEDEEEIDVVSVEKRQAPGKRSESGSPSAGGHSKPPHSPLVLKRCHVSTHQHNYAAPPSTRKDYPAAKRVKLDSVRVLRQISNNRKCTSPRSSDTEENVKRRTHNVLERQRRNELKRSFFALRDQIPELENNEKAPKVVILKKATAYILSVQAEEQKLISEEDLLRKRREQLKHKLEQLR" # myc
    sequence = "MPAVKKEFPGREDLALALATFHPTLAALPLPPLPGYLAPLPAAAALPPAASLPASAAGYEALLAPPLRPPRAYLSLHEAAPHLHLPRDPLALERFSATAAAAPDFQPLLDNGEPCIEVECGANRALLYVRKLCQGSKGPSIRHRGEWLTPNEFQFVSGRETAKDWKRSIRHKGKSLKTLMSKGILQVHPPICDCPGCRISSPVNRGRLADKRTVALPAARNLKKERTPSFSASDGDSDGSGPTCGRRPGLKQEDGPHIRIMKRRVHTHWDVNISFREASCSQDGNLPTLISSVHRSRHLVMPEHQSRCEFQRGSLEIGLRPAGDLLGKRLGRSPRISSDCFSEKRARSESPQEALLLPRELGPSMAPEDHYRRLVSALSEASTFEDPQRLYHLGLPSHDLLRVRQEVAAAALRGPSGLEAHLPSSTAGQRRKQGLAQHREGAAPAAAPSFSERELPQPPPLLSPQNAPHVALGPHLRPPFLGVPSALCQTPGYGFLPPAQAEMFAWQQELLRKQNLARLELPADLLRQKELESARPQLLAPETALRPNDGAEELQRRGALLVLNHGAAPLLALPPQGPPGSGPPTPSRDSARRAPRKGGPGPASARPSESKEMTGARLWAQDGSEDEPPKDSDGEDPETAAVGCRGPTPGQAPAGGAGAEGKGLFPGSTLPLGFPYAVSPYFHTGAVGGLSMDGEEAPAPEDVTKWTVDDVCSFVGGLSGCGEYTRVFREQGIDGETLPLLTEEHLLTNMGLKLGPALKIRAQVARRLGRVFYVASFPVALPLQPPTLRAPERELGTGEQPLSPTTATSPYGGGHALAGQTSPKQENGTLALLPGAPDPSQPLC"

    # Generate
    base_heatmap, amino_acids = generate_heatmap(sequence, base_model, base_tokenizer)
    ms_heatmap, _ = generate_heatmap(sequence, ms_model, ms_tokenizer)
    fs_heatmap, _ = generate_heatmap(sequence, fs_model, fs_tokenizer)

    # Compute difference
    ms_diff_heatmap = ms_heatmap - base_heatmap
    fs_diff_heatmap = fs_heatmap - base_heatmap

    plot_heatmap(base_heatmap, "Original ESM2 Model (LLRs)", sequence, amino_acids)
    plot_heatmap(ms_heatmap, "Fine-tuned Missense Model (LLRs)", sequence, amino_acids)
    plot_heatmap(fs_heatmap, "Fine-tuned Frameshift Model (LLRs)", sequence, amino_acids)
    plot_heatmap(ms_diff_heatmap, "Difference (Fine-tuned Missense - Original)", sequence, amino_acids)
    plot_heatmap(fs_diff_heatmap, "Difference (Fine-tuned Frameshift - Original)", sequence, amino_acids)

    # Comapre amino acid predictions
    masked_pos = 27

    original_preds = topk_predictions(base_model, base_tokenizer, sequence, masked_pos)
    ms_preds = topk_predictions(ms_model, ms_tokenizer, sequence, masked_pos)
    fs_preds = topk_predictions(fs_model, ms_tokenizer, sequence, masked_pos)

    print("Original model top predictions:", original_preds)
    print("Fine-tuned missense model top predictions:", ms_preds)
    print("Fine-tuned frameshift model top predictions:", fs_preds)

if __name__ == '__main__':
    main()