#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

def plot_heatmap(descr, params, gene, data, title, sequence, amino_acids):
    plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis_r", aspect="auto", vmin=None, vmax=None)
    plt.xticks(range(len(sequence)), list(sequence))
    plt.yticks(range(20), amino_acids)
    plt.xlabel("Position in Protein Sequence")
    plt.ylabel("Amino Acid Mutations")
    plt.title(title + ' ' + str(params) + 'M')
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.tight_layout()
    
    # Define the path
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/heatmaps/{gene}/{title.replace(' ', '_')}.png"
    folder = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
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
    ms_descr = "missense"
    fs_descr = "frameshift"

    # Sequence and gene of interest 
    ms_gene = "OBSCN"
    ms_sequence = "MDQPQFSGAPRFLTRPKAFVVSVGKDATLSCQIVGNPTPQVSWEKDQQPVAAGARFRLAQDGDLYRLTILDLALGDSGQYVCRARNAIGEAFAAVGLQVDAEAACAEQAPHFLLRPTSIRVREGSEATFRCRVGGSPRPAVSWSKDGRRLGEPDGPRVRVEELGEASALRIRAARPRDGGTYEVRAENPLGAASAAAALVVDSDAADTASRPGTSTAALLAHLQRRREAMRAEGAPASPPSTGTRTCTVTEGKHARLSCYVTGEPKPETVWKKDGQLVTEGRRHVVYEDAQENFVLKILFCKQSDRGLYTCTASNLVGQTYSSVLVVVREPAVPFKKRLQDLEVREKESATFLCEVPQPSTEAAWFKEETRLWASAKYGIEEEGTERRLTVRNVSADDDAVYICETPEGSRTVAELAVQGNLLRKLPRKTAVRVGDTAMFCVELAVPVGPVHWLRNQEEVVAGGRVAISAEGTRHTLTISQCCLEDVGQVAFMAGDCQTSTQFCVSAPRKPPLQPPVDPVVKARMESSVILSWSPPPHGERPVTIDGYLVEKKKLGTYTWIRCHEAEWVATPELTVADVAEEGNFQFRVSALNSFGQSPYLEFPGTVHLAPKLAVRTPLKAVQAVEGGEVTFSVDLTVASAGEWFLDGQALKASSVYEIHCDRTRHTLTIREVPASLHGAQLKFVANGIESSIRMEVRAAPGLTANKPPAAAAREVLARLHEEAQLLAELSDQAAAVTWLKDGRTLSPGPKYEVQASAGRRVLLVRDVARDDAGLYECVSRGGRIAYQLSVQGLARFLHKDMAGSCVDAVAGGPAQFECETSEAHVHVHWYKDGMELGHSGERFLQEDVGTRHRLVAATVTRQDEGTYSCRVGEDSVDFRLRVSEPKVVFAKEQLARRKLQAEAGASATLSCEVAQAQTEVTWYKDGKKLSSSSKVCMEATGCTRRLVVQQAGQADAGEYSCEAGGQRLSFHLDVKEPKVVFAKDQVAHSEVQAEAGASATLSCEVAQAQTEVMWYKDGKKL"

    fs_gene = "ARID1A"
    fs_sequence = "MAAQVAPAAASSLGNPPPPPPSELKKAEQQQREEAGGEAAAAAAAERGEMKAAAGQESEGPAVGPPQPLGKELQDGAESNGGGGGGGAGSGGGPGAEPDLKNSNGNAGPRPALNNNLTEPPGGGGGGSSDGVGAPPHSAAAALPPPAYGFGQPYGRSPSAVAAAAAAVFHQQHGGQQSPGLAALQSGGGGGLEPYAGPQQNSHDHGFPNHQYNSYYPNRSAYPPPAPAYALSSPRGGTPGSGAAAAAGSKPPPSSSASASSSSSSFAQQRFGAMGGGGPSAAGGGTPQPTATPTLNQLLTSPSSARGYQGYPGGDYSGGPQDGGAGKGPADMASQCWGAAAAAAAAAAASGGAQQRSHHAPMSPGSSGGGGQPLARTPQPSSPMDQMGKMRPQPYGGTNPYSQQQGPPSGPQQGHGYPGQPYGSQTPQRYPMTMQGRAQSAMGGLSYTQQIPPYGQQGPSGYGQQGQTPYYNQQSPHPQQQQPPYSQQPPSQTPHAQPSYQQQPQSQPPQLQSSQPPYSQQPSQPPHQQSPAPYPSQQSTTQQHPQSQPPYSQPQAQSPYQQQQPQQPAPSTLSQQAAYPQPQSQQSQQTAYSQQRFPPPQELSQDSFGSQASSAPSMTSSKGGQEDMNLSLQSRPSSLPDLSGSIDDLPMGTEGALSPGVSTSGISSSQGEQSNPAQSPFSPHTSPHLPGIRGPSPSPVGSPASVAQSRSGPLSPAAVPGNQMPPRPPSGQSDSIMHPSMNQSSIAQDRGYMQRNPQMPQYSSPQPGSALSPRQPSGGQIHTGMGSYQQNSMGSYGPQGGQYGPQGGYPRQPNYNALPNANYPSAGMAGGINPMGAGGQMHGQPGIPPYGTLPPGRMSHASMGNRPYGPNMANMPPQVGSGMCPPPGGMNRKTQETAVAMHVAANSIQNRPPGYPNMNQGGMMGTGPPYGQGINSMAGMINPQGPPYSMGGTMANNSAGMAASPEMMGLGDVKLTPATKMNNKADGTPKTESKSKKSSSSTTTNEKITKLYELGGEPERKM"

    # Load original ESM-2 model
    base_model_name = f"/g/data/gi52/jaime/esm2_{params}M_model"
    base_tokenizer = EsmTokenizer.from_pretrained(base_model_name)
    base_model = EsmForMaskedLM.from_pretrained(base_model_name)

    # Load missense fine-tuned model
    ms_model_path = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/{ms_descr}/run2/final_merged"
    ms_tokenizer = EsmTokenizer.from_pretrained(ms_model_path)
    ms_model = EsmForMaskedLM.from_pretrained(ms_model_path)

    # Load Frameshift fine-tuned model
    fs_model_path = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/{fs_descr}/run2/final_merged"
    fs_tokenizer = EsmTokenizer.from_pretrained(fs_model_path)
    fs_model = EsmForMaskedLM.from_pretrained(fs_model_path)


    # Generate heatmaps
    ms_base_heatmap, amino_acids = generate_heatmap(ms_sequence, base_model, base_tokenizer)
    ms_heatmap, _ = generate_heatmap(ms_sequence, ms_model, ms_tokenizer)

    fs_base_heatmap, amino_acids = generate_heatmap(fs_sequence, base_model, base_tokenizer)
    fs_heatmap, _ = generate_heatmap(fs_sequence, fs_model, fs_tokenizer)

    # Compute difference
    ms_diff_heatmap = ms_heatmap - ms_base_heatmap
    fs_diff_heatmap = fs_heatmap - fs_base_heatmap

    plot_heatmap("original", params, ms_gene, ms_base_heatmap, "Original ESM2 Model (LLRs)", ms_sequence, amino_acids)
    plot_heatmap("missense", params, ms_gene, ms_heatmap, "Fine-tuned Missense Model (LLRs)", ms_sequence, amino_acids)
    plot_heatmap("missense", params, ms_gene, ms_diff_heatmap, "Difference (Fine-tuned Missense - Original)", ms_sequence, amino_acids)

    plot_heatmap("original", params, fs_gene, fs_base_heatmap, "Original ESM2 Model (LLRs)", fs_sequence, amino_acids)
    plot_heatmap("frameshift", params, fs_gene, fs_heatmap, "Fine-tuned Frameshift Model (LLRs)", fs_sequence, amino_acids)
    plot_heatmap("frameshift", params, fs_gene, fs_diff_heatmap, "Difference (Fine-tuned Frameshift - Original)", fs_sequence, amino_acids)

    # Compare amino acid predictions
    ms_masked_pos = 81
    fs_masked_pos = 122

    ms_original_preds = topk_predictions(base_model, base_tokenizer, ms_sequence, ms_masked_pos)
    ms_preds = topk_predictions(ms_model, ms_tokenizer, ms_sequence, ms_masked_pos)

    fs_original_preds = topk_predictions(base_model, base_tokenizer, fs_sequence, fs_masked_pos)
    fs_preds = topk_predictions(fs_model, fs_tokenizer, fs_sequence, fs_masked_pos)

    print(f"Original model top predictions for gene {ms_gene} at position {ms_masked_pos}:", ms_original_preds)
    print(f"Fine-tuned missense model top predictions for gene {ms_gene} at position {ms_masked_pos}:", ms_preds)
    print('\n')
    print(f"Original model top predictions for gene {fs_gene} at position {fs_masked_pos}:", fs_original_preds)
    print(f"Fine-tuned frameshift model top predictions for gene {fs_gene} at position {fs_masked_pos}:", fs_preds)


if __name__ == '__main__':
    main()