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
from peft import PeftModel, PeftConfig

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

def plot_heatmap_with_dots(data, gene, title, sequence, amino_acids, mutation_list, start_pos=1):

    plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis_r", aspect="auto", vmin=-20, vmax=10)
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
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/frameshift/run6/heatmaps/{gene}/{title.replace(' ', '_')}.png"
    folder = os.path.dirname(save_path)

    os.makedirs(folder, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_heatmap(gene, data, title, sequence, amino_acids):
    plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis_r", aspect="auto", vmin=-20, vmax=10)
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
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/frameshift/run6/heatmaps/{gene}/{title.replace(' ', '_')}.png"
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

    # Sequence and gene of interest 
    #sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    #gene = "tp53"

    gene = "myc"
    sequence = "MDFFRVVENQQPPATMPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPSEDIWKKFELLPTPPLSPSRRSGLCSPSYVAVTPFSLRGDNDGGGGSFSTADQLEMVTELLGGDMVNQSFICDPDDETFIKNIIIQDCMWSGFSAAAKLVSEKLASYQAARKDSGSPNPARGHSVCSTSSLYLQDLSAAASECIDPSVVFPYPLNDSSSPKSCASQDSSAFSPSSDSLLSSTESSPQGSPEPLVLHEETPPTTSSDSEEEQEDEEEIDVVSVEKRQAPGKRSESGSPSAGGHSKPPHSPLVLKRCHVSTHQHNYAAPPSTRKDYPAAKRVKLDSVRVLRQISNNRKCTSPRSSDTEENVKRRTHNVLERQRRNELKRSFFALRDQIPELENNEKAPKVVILKKATAYILSVQAEEQKLISEEDLLRKRREQLKHKLEQLRNSCA"


    # Load original ESM-2 model
    base_model_name = f"/g/data/gi52/jaime/esm2_650M_model"
    base_tokenizer = EsmTokenizer.from_pretrained(base_model_name)
    base_model = EsmForMaskedLM.from_pretrained(base_model_name)
    base_heatmap, amino_acids = generate_heatmap(sequence, base_model, base_tokenizer)
    #plot_heatmap(gene, base_heatmap, "Original ESM2 Model (LLRs)", sequence, amino_acids)
    #print("Plotted original model heatmap")

 
    # Load mutations and split as it was during training  
    data_path = Path("./data")
    test_size = 0.2
    valid_size = 0.25

    # frameshift 
    fs_df = pd.read_parquet(data_path / "update2_all_fs_samples.parquet")
    fs_train_df, fs_test_df = train_test_split(fs_df, test_size=test_size, random_state=0)
    fs_train_df, fs_valid_df = train_test_split(fs_train_df, test_size=valid_size, random_state=0)

    fs_filtered = (
        fs_train_df[fs_train_df['HugoSymbol'] == gene.upper()][['ProteinChange']] 
    )
    fs_mutation_list = fs_filtered['ProteinChange'].tolist()


    for epoch in range(5):
        print(f"\nstarting plotting for epoch{epoch}")
        
        base_model_name = f"/g/data/gi52/jaime/esm2_650M_model"
        base_model = EsmForMaskedLM.from_pretrained(base_model_name)

        # Load fine-tuned model 
        epoch_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/frameshift/run6/epoch{epoch}"

        ''' 
        # Inspect config
        cfg = PeftConfig.from_pretrained(epoch_path)
        print("PEFT config:", cfg)
        print(PeftConfig.from_pretrained(epoch_path).to_dict())

        peft_config = PeftConfig.from_pretrained(epoch_path)
        peft_model = PeftModel.from_pretrained(base_model, epoch_path, config=peft_config)
        peft_model.print_trainable_parameters()
        merged_model = peft_model.merge_and_unload()
        print('merged')
        merged_model.print_trainable_parameters()
        merged_model.eval()
        '''

        #merged_model = PeftModel.from_pretrained(epoch_path)
        merged_model = EsmForMaskedLM.from_pretrained(epoch_path)
        # ms_tokenizer = EsmTokenizer.from_pretrained(epoch_path)

        # Generate heatmaps
        fs_heatmap, _ = generate_heatmap(sequence, merged_model, base_tokenizer)

        # Compute difference
        fs_diff_heatmap = fs_heatmap - base_heatmap

        plot_heatmap(gene, fs_heatmap, f"Epoch{epoch}: Fine-tuned Frameshift Model (LLRs)", sequence, amino_acids)
        plot_heatmap(gene, fs_diff_heatmap, f"Epoch{epoch}: Difference (Fine-tuned Frameshift - Original) (LLRs)", sequence, amino_acids)

        # Generate heatmap with mutations as dots in positions
        plot_heatmap_with_dots(fs_diff_heatmap, gene, f"Epoch{epoch}: Difference (Fine-tuned Frameshift - Original) with Mutations", sequence, amino_acids, fs_mutation_list, start_pos=0)
        
        # Compare amino acid predictions
        masked_pos = 74

        original_preds = topk_predictions(base_model, base_tokenizer, sequence, masked_pos)
        fs_preds = topk_predictions(merged_model, base_tokenizer, sequence, masked_pos)

        print(f"Epoch {epoch}: Original model top predictions at position {masked_pos}:", original_preds)
        print(f"Epoch {epoch}: Fine-tuned frameshift model top predictions at position {masked_pos}:", fs_preds)

        for batch in range(2000, 6001, 2000):
            # Load fine-tuned model 
            batch_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/frameshift/run6/epoch{epoch}_batch{batch}"
            #batch_model = PeftModel.from_pretrained(base_model, batch_path)
            #merged_batch_model = batch_model.merge_and_unload()
            batch_model = EsmForMaskedLM.from_pretrained(batch_path)

            # Generate heatmaps
            fs_heatmap, _ = generate_heatmap(sequence, batch_model, base_tokenizer)

            # Compute difference
            fs_diff_heatmap = fs_heatmap - base_heatmap

            plot_heatmap(gene, fs_heatmap, f"Epoch{epoch}, batch{batch} checkpoint: Fine-tuned Frameshift Model (LLRs)", sequence, amino_acids)
            plot_heatmap(gene, fs_diff_heatmap, f"Epoch{epoch}, batch{batch} checkpoint: Difference (Fine-tuned Frameshift - Original) (LLRs)", sequence, amino_acids)

            # Compare amino acid predictions

            original_preds = topk_predictions(base_model, base_tokenizer, sequence, masked_pos)
            fs_preds = topk_predictions(batch_model, base_tokenizer, sequence, masked_pos)

            print(f"\nEpoch {epoch}, Batch {batch}: Original model top predictions at position {masked_pos}:", original_preds)
            print(f"Epoch {epoch}, Batch {batch}: Fine-tuned frameshift model top predictions at position {masked_pos}:", fs_preds)


if __name__ == '__main__':
    main()