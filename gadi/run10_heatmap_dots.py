"""
make heatmap from run10 showing the mutations that were seen during training
"""
import torch
import pandas as pd
import os, re
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from peft import PeftModel
from transformers import EsmForMaskedLM, EsmTokenizer
import matplotlib.colors as mcolors
import numpy as np

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
    plt.title(title + ' 650M')
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

def parse_hgvs(protein_change):
    """Extract WT AA, position, and mutant AA from HGVS string like 'p.A586V'."""
    pattern = r"^p\.([A-Z])(\d+)(?!Ter)([A-Z])"
    m = re.match(pattern, protein_change)
    if not m:
        return None
    wt, pos, mt = m.groups()
    return wt, int(pos), mt

def custom_plot_heatmap_with_dots(mutation_list, data, gene, title, sequence, base_dir, amino_acids, start_pos=1):

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
    save_path = os.path.join(base_dir, f"{gene}/{title.replace(' ', '_')}.png")
    folder = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300)
    print(f"Saved {gene} heatmap with dots to {save_path}")
    plt.close() 


def main():

    # Path to your saved indices file
    base_dir = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run10"
    indices = torch.load(os.path.join(base_dir, "mut_train_indices.pt"))
    
    # Convert to Python integers
    indices = indices.tolist()
    
    # Load missense data
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "update4_all_ms_samples.parquet")

    # Split data into 75% train, 5% validate, 20% test 
    test_size = 0.20
    ms_train_df, ms_test_df = train_test_split(ms_df, test_size=test_size, random_state=0)
    valid_size = 0.0625 
    ms_train_df, ms_valid_df = train_test_split(ms_train_df, test_size=valid_size, random_state=0)
    print("Train, Valid, Test split is:", len(ms_train_df), len(ms_valid_df), len(ms_test_df))


    # Get first 10,000 x 8 (batch size)rows
    first_80k_indices = indices[:80000]
    # Access the first 10,000 samples
    first_80k_subset = ms_train_df.iloc[first_80k_indices]

    print(first_80k_subset.head())


    # mutations for the specific gene
    gene = "TP53"
    sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    
    ms_filtered = (
        first_80k_subset[first_80k_subset['HugoSymbol'] == gene.upper()][['ProteinChange']] 
    )
    ms_mutation_list = ms_filtered['ProteinChange'].tolist()

    print(f"Number of missense training mutations for {gene}: {len(ms_mutation_list)}")
    print(f"Mutations: {ms_mutation_list}")


    # Load in the missense run10 model 
    adapter_path = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run10/epoch0_batch10000"
    base_model_path = f"/g/data/gi52/jaime/esm2_650M_model"
    base_model = EsmForMaskedLM.from_pretrained(base_model_path)

    # Load model + adapter
    print("Loading ESM2 model and adapter...", flush=True)
    tokenizer = EsmTokenizer.from_pretrained(base_model_path)
    adapter_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    csm_model = adapter_model.merge_and_unload()

    # frozen baseline
    frozen_base_model = EsmForMaskedLM.from_pretrained(base_model_path)
    frozen_base_model.eval()
    for p in frozen_base_model.parameters():
        p.requires_grad = False


    # Generate heatmaps
    base_heatmap, amino_acids = generate_heatmap(sequence, frozen_base_model, tokenizer)
    csm_heatmap, _ = generate_heatmap(sequence, csm_model, tokenizer)

    # Compute difference
    ms_diff_heatmap = csm_heatmap - base_heatmap


    # save dir
    save_dir = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run10"


    # Create a heatmap of the first 10,000 samples
    # Generate heatmap with mutations as dots in positions

    custom_plot_heatmap_with_dots(ms_mutation_list, ms_diff_heatmap, gene, "Difference (Fine-tuned Missense - Original) with Mutations", 
                           sequence, save_dir, amino_acids, start_pos=0)



if __name__ == "__main__":
    main()