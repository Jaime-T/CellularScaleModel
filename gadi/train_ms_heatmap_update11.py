# -*- coding: utf-8 -*-

"""
Iteration 5
Train on all missense mutations including replicates (mask mutation position)
Batches of 8 have 7 MT and 1 WT sequence to reduce catastrophic forgetting

Compute 2 loss values: on new mutation data, and on old wildtype data
Make heatmap every 1000 batches: TP53, MYC, RPL2 (negative control)
Train ESM-2 650M model on missense variants from updated DepMap data

Progress tracking, plus model, tokeniser and optimiser saving 
Split data into 75% train, 5% validate, 20% test 

"""
import os
import re
import tempfile
import pandas as pd
import numpy as np
from timeit import default_timer as timer

tmpdir = os.getenv('TMPDIR', tempfile.gettempdir())
mpl_cache = os.path.join(tmpdir, 'matplotlib-cache')
os.makedirs(mpl_cache, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_cache

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import EsmForMaskedLM, EsmTokenizer
import random
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import torch.nn.functional as F
import csv
from itertools import zip_longest
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
    
def random_tokenize_and_mask_seqs(batch, tokenizer, window_size: int = 1022, mlm_probability: float = 0.15):
    # Tokenize the batch
    encoded_seqs = tokenizer(
        batch['wt_windowed_seq'].tolist(),
        padding="max_length",
        truncation=True,
        max_length=min(window_size, tokenizer.model_max_length),
        return_tensors="pt"  # use PyTorch for masking logic
    )

    input_ids = encoded_seqs["input_ids"]
    attention_mask = encoded_seqs["attention_mask"]

    # Clone to create targets
    targets = input_ids.clone()

    # Create probability mask (randomly choose tokens to mask)
    probability_matrix = torch.full(targets.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in targets.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Sample masked indices
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Replace selected input_ids with [MASK] token
    input_ids[masked_indices] = tokenizer.mask_token_id

    # Only keep targets for masked tokens
    targets[~masked_indices] = -100

    df = pd.DataFrame({
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": targets.tolist()
    })

    return df

def mut_tokenize_and_mask_seqs(batch, tokenizer, col='mt_windowed_seq', window_size: int = 1022):
    # Tokenize the batch
    encoded_seqs = tokenizer(
        batch[col].tolist(),
        padding="max_length",
        truncation=True,
        max_length=min(window_size, tokenizer.model_max_length),
        return_tensors="pt"  # use PyTorch for masking logic
    )

    input_ids = encoded_seqs["input_ids"]
    attention_mask = encoded_seqs["attention_mask"]

    # Clone to create targets
    targets = input_ids.clone()

    # Mask the mutation site position: 
    for i, (prot_change, start_index) in enumerate(zip(batch['ProteinChange'], batch['start_index'])):

        # Extract mutation position, e.g. p.A27K → 27
        m = re.match(r"p\.\D+(\d+)", prot_change)
        if m:
            mut_pos = int(m.group(1))

            # Convert to window-relative position
            window_pos = mut_pos - start_index  

            token_index = window_pos 

            if 0 <= token_index < input_ids.shape[1]:
                # Keep only mutation site for loss
                targets[i, :] = -100
                targets[i, token_index] = encoded_seqs["input_ids"][i, token_index]

                # Mask the mutation position
                input_ids[i, token_index] = tokenizer.mask_token_id

    df = pd.DataFrame({
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": targets.tolist()
    })

    return df

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_loss(model, data_loader, device):
    model.eval()  # disable dropout & batchnorm
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    return total_loss / len(data_loader)

def plot_loss(loss_per_epoch, descr, base_dir):
    # Unpack lists
    epochs = [e[0] for e in loss_per_epoch]
    train_losses = [e[1] for e in loss_per_epoch]
    val_losses = [e[2] for e in loss_per_epoch]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='s', label='Validation Loss')
    plt.title(f'Loss vs Epochs for {descr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(base_dir, "loss_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved loss plot to {plot_path}")

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

def custom_plot_heatmap(gene, data, title, sequence, amino_acids, base_dir):

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
    save_path = os.path.join(base_dir, f"heatmaps/{gene}/{title.replace(' ', '_')}.png")
    folder = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300)
    print(f"Saved {gene} heatmap to {save_path}")
    plt.close() 

def scaled_plot_heatmap(gene, data, title, sequence, amino_acids, base_dir):

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
    plt.title(title + ' 650M')
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.tight_layout()
    
    # Define the path
    save_path = os.path.join(base_dir, f"heatmaps/{gene}/{title.replace(' ', '_')}.png")
    folder = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300)
    print(f"Saved {gene} heatmap to {save_path}")
    plt.close() 


#### Testing new function - creates batches with 7 MT and 1 WT sample
# Has debug option to print out batch details to confirm correctness
def combined_batch_generator(
        mut_dataset, wt_dataset, device,
        ratio=(7, 1), tokenizer=None,
        debug=False, max_debug_batches=3
    ):
    mut_len = len(mut_dataset)
    wt_len = len(wt_dataset)
    n_batches = min(mut_len, wt_len)

    # Create a single shared shuffled index order
    indices = list(range(n_batches))
    random.shuffle(indices)

    debug_batches_printed = 0

    for idx in indices:
        mut_item = mut_dataset[idx]  # dict
        wt_item  = wt_dataset[idx]   # dict

        batch = {}
        for key in mut_item.keys():
            mut_tensor = mut_item[key]
            wt_tensor  = wt_item[key]

            # Ensure both are at least 2D (batch dimension first)
            if mut_tensor.dim() == 1:
                mut_tensor = mut_tensor.unsqueeze(0)
            if wt_tensor.dim() == 1:
                wt_tensor = wt_tensor.unsqueeze(0)

            # Replicate the single mutant sample 7 times
            mut_rep = mut_tensor.repeat(
                (ratio[0],) + (1,) * (mut_tensor.dim() - 1)
            )
            # Add single wildtype
            wt_rep = wt_tensor
            batch[key] = torch.cat([mut_rep, wt_rep], dim=0).to(device, non_blocking=True)

        # ✅ Optional Debugging
        if debug and debug_batches_printed < max_debug_batches:
            print("\n================ DEBUG BATCH =================")
            print(f"Index used: {idx}")
            print(f"Batch keys: {list(batch.keys())}")
            print(f"Batch shapes:")
            for k, v in batch.items():
                print(f"  {k}: {v.shape}")

            if tokenizer is not None and "input_ids" in batch:
                input_ids = batch["input_ids"].cpu()

                # Decode first mutant example
                decoded_mut = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                # Decode wildtype example (last)
                decoded_wt = tokenizer.decode(input_ids[-1], skip_special_tokens=False)

                print("\n--- Mutant[0] Decoded ---")
                print(decoded_mut)
                print("\n--- Wildtype[-1] Decoded ---")
                print(decoded_wt)

                # Locate mask token position(s)
                if tokenizer.mask_token_id in input_ids:
                    mask_pos_mut = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                    mask_pos_wt  = (input_ids[-1] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                    print(f"\nMask positions in mutant sample: {mask_pos_mut.tolist()}")
                    print(f"Mask positions in wildtype sample: {mask_pos_wt.tolist()}")
                else:
                    print("No [MASK] token found in batch[\"input_ids\"]")

            debug_batches_printed += 1

        yield batch


def train_model(tokenizer, base_model, frozen_base_model, descr, mut_train_data, wt_train_data,
                mut_valid_data, wt_valid_data,
                lora_config, batch_size=6, max_epochs=20, lr=5e-5):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # LoRA-wrapped trainable model
    model = get_peft_model(base_model, lora_config)
    optimizer = AdamW(model.parameters(), lr=lr)
    model.print_trainable_parameters()
    model.to(device)

    # directories
    base_dir = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/run11"
    os.makedirs(base_dir, exist_ok=True)

    loss_file_csv = os.path.join(base_dir, "loss_per_epoch.csv")
    progress_file = os.path.join(base_dir, "progress.pt")

    # track loss
    loss_per_epoch = []

    n_batches = min(len(mut_train_data), len(wt_train_data))

    # New!!
    combined_batches = combined_batch_generator(
        mut_train_data,
        wt_train_data,
        device,
        ratio=(7,1),
        tokenizer=tokenizer,  # pass tokenizer for decoding
        debug=True,           # turn ON debug
        max_debug_batches=2   # print first 2 batches only
    )

    ''' 
    # Training set data loaders
    mut_train_loader = torch.utils.data.DataLoader(
        mut_train_data, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    wt_train_loader = torch.utils.data.DataLoader(
        wt_train_data, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    '''

    # Validation set data loaders
    mut_valid_loader = DataLoader(
        mut_valid_data, batch_size=batch_size,shuffle=False, pin_memory=True
    )
    wt_valid_loader = DataLoader(
        wt_valid_data, batch_size=batch_size,shuffle=False, pin_memory=True
    )

    # make heatmaps for frozen model:

        # myc
    myc_gene = "myc"
    myc_sequence = "MDFFRVVENQQPPATMPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPSEDIWKKFELLPTPPLSPSRRSGLCSPSYVAVTPFSLRGDNDGGGGSFSTADQLEMVTELLGGDMVNQSFICDPDDETFIKNIIIQDCMWSGFSAAAKLVSEKLASYQAARKDSGSPNPARGHSVCSTSSLYLQDLSAAASECIDPSVVFPYPLNDSSSPKSCASQDSSAFSPSSDSLLSSTESSPQGSPEPLVLHEETPPTTSSDSEEEQEDEEEIDVVSVEKRQAPGKRSESGSPSAGGHSKPPHSPLVLKRCHVSTHQHNYAAPPSTRKDYPAAKRVKLDSVRVLRQISNNRKCTSPRSSDTEENVKRRTHNVLERQRRNELKRSFFALRDQIPELENNEKAPKVVILKKATAYILSVQAEEQKLISEEDLLRKRREQLKHKLEQLRNSCA"
    myc_base_heatmap, amino_acids = generate_heatmap(myc_sequence, frozen_base_model, tokenizer)
    scaled_plot_heatmap(myc_gene, myc_base_heatmap, f"Original ESM2 Model for {myc_gene} gene (LLRs)", myc_sequence, amino_acids, base_dir)

        # tp53
    tp53_gene = "tp53"
    tp53_sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    tp53_base_heatmap, amino_acids = generate_heatmap(tp53_sequence, frozen_base_model, tokenizer)
    scaled_plot_heatmap(tp53_gene, tp53_base_heatmap, f"Original ESM2 Model for {tp53_gene} gene (LLRs)", tp53_sequence, amino_acids, base_dir)

        # rpl2 - gene for negative control
    rpl_gene = "rpl2"  
    rpl_sequence = "MILKKYKPTTPSLRGLVQIDRSLLWKGDPVKKLTVGMIESAGRNNTGRITVYHRGGGHKTKYRYIDFKRSNYNIPGIVERLEYDPNRTCFIALIKDNENNFSYILAPHDLKVGDTVITGNDIDIRIGNTLPLRNIPIGTMIHNIELNPGKGGKIVRSAGSSAQLISKDENGFCMLKLPSGEYRLFPNNSLATIGILSNIDNKNIKIGKAGRSRWMGRRPIVRGVAMNPVDHPHGGGEGKTSGGRPSVTPWSWPTKGQPTRSKRKYNKLIVQRAKKKI"
    rpl_base_heatmap, amino_acids = generate_heatmap(rpl_sequence, frozen_base_model, tokenizer)
    scaled_plot_heatmap(rpl_gene, rpl_base_heatmap, f"Original ESM2 Model for {rpl_gene} gene (LLRs)", rpl_sequence, amino_acids, base_dir)

    # -- Training loop --
    for epoch in range(max_epochs):

        print(f"\nStarting training for epoch {epoch}...")
        epoch_start = timer()
        total_loss = 0

        #for batch_idx, batch in enumerate(mut_train_loader):
        for batch_idx, batch in enumerate(combined_batches):

            model.train()
            # Move to device
            #batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # save progress every batch
            torch.save({"epoch": epoch, "batch_idx": batch_idx}, progress_file)

            # make heatmap every 1000 batches
            if (batch_idx + 1) % 1000 == 0:  
                batch_num = batch_idx + 1 
                batch_loss = total_loss / batch_num
                print(f"Epoch {epoch}, Batch {batch_num}/{n_batches} | Avg Loss during training: {batch_loss:.4f}")

                # plot heatmap for myc gene 
                myc_fs_heatmap, _ = generate_heatmap(myc_sequence, model, tokenizer)
                myc_fs_diff_heatmap = myc_fs_heatmap - myc_base_heatmap
                scaled_plot_heatmap(myc_gene, myc_fs_heatmap, f"Epoch {epoch}, Batch {batch_num}: Fine-tuned Missense Model (LLRs)", myc_sequence, amino_acids, base_dir)
                custom_plot_heatmap(myc_gene, myc_fs_diff_heatmap, f"Epoch {epoch}, Batch {batch_num}: Difference (Fine-tuned Missense - Original)", myc_sequence, amino_acids, base_dir)
                
                # plot heatmap for tp53 gene 
                tp53_fs_heatmap, _ = generate_heatmap(tp53_sequence, model, tokenizer)
                tp53_fs_diff_heatmap = tp53_fs_heatmap - tp53_base_heatmap
                scaled_plot_heatmap(tp53_gene, tp53_fs_heatmap, f"Epoch {epoch}, Batch {batch_num}: Fine-tuned Missense Model (LLRs)", tp53_sequence, amino_acids, base_dir)
                custom_plot_heatmap(tp53_gene, tp53_fs_diff_heatmap, f"Epoch {epoch}, Batch {batch_num}: Difference (Fine-tuned Missense - Original)", tp53_sequence, amino_acids, base_dir)

                # plot heatmap for rpl2 gene 
                rpl_fs_heatmap, _ = generate_heatmap(rpl_sequence, model, tokenizer)
                rpl_fs_diff_heatmap = rpl_fs_heatmap - rpl_base_heatmap
                scaled_plot_heatmap(rpl_gene, rpl_fs_heatmap, f"Epoch {epoch}, Batch {batch_num}: Fine-tuned Missense Model (LLRs)", rpl_sequence, amino_acids, base_dir)
                custom_plot_heatmap(rpl_gene, rpl_fs_diff_heatmap, f"Epoch {epoch}, Batch {batch_num}: Difference (Fine-tuned Missense - Original)", rpl_sequence, amino_acids, base_dir)

                # Compute loss on validation set so far
                # MT val set 
                mut_val_loss = compute_loss(model, mut_valid_loader, device)

                # WT val set 
                wt_val_loss = compute_loss(model, wt_valid_loader, device)
                
                print(f"Epoch {epoch}, Batch {batch_num} | Validation Loss - Mutant: {mut_val_loss:.4f}, Wildtype: {wt_val_loss:.4f}\n")

                # Save loss values so far in a file
                temp_loss_file = os.path.join(base_dir, "temp_loss_per_batch.csv")
                with open(temp_loss_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:  # Write header if file is empty
                        writer.writerow(["epoch", "batch", "during_train_loss", "mut_val_loss", "wt_val_loss"])
                    writer.writerow([epoch, batch_num, batch_loss, mut_val_loss, wt_val_loss])
                print(f"Saved temporary loss values to {temp_loss_file}\n")


            # save every 1000 batches
            if (batch_idx + 1) % 1000 == 0:  
                batch_num = batch_idx + 1 
                batch_dir = os.path.join(base_dir, f"epoch{epoch}_batch{batch_num}")
                os.makedirs(batch_dir, exist_ok=True)
                model.save_pretrained(batch_dir)
                tokenizer.save_pretrained(batch_dir)
                torch.save(optimizer.state_dict(), os.path.join(batch_dir, "optimizer.pt"))
                print(f"Saved checkpoint for epoch {epoch}, batch {batch_num} in {batch_dir}\n")


        

        # save every epoch - just the Lora adapter weights
        epoch_dir = os.path.join(base_dir, f"epoch{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        torch.save(optimizer.state_dict(), os.path.join(epoch_dir, "optimizer.pt"))
        print(f"Saved checkpoint for epoch {epoch} in {epoch_dir}\n")


        epoch_time = timer() - epoch_start

        # Compute loss on validation set after each epoch
        # MT val set 
        mut_val_loss = compute_loss(model, mut_valid_loader, device)

        # WT val set 
        wt_val_loss = compute_loss(model, wt_valid_loader, device)
        
        print(f"Epoch {epoch}: Validation Loss - Mutant: {mut_val_loss:.4f}, Wildtype: {wt_val_loss:.4f}\n")

        
        loss_per_epoch.append((epoch, mut_val_loss, wt_val_loss))

        # plot heatmap for myc gene 
        myc_fs_heatmap, _ = generate_heatmap(myc_sequence, model, tokenizer)
        myc_fs_diff_heatmap = myc_fs_heatmap - myc_base_heatmap
        scaled_plot_heatmap(myc_gene, myc_fs_heatmap, f"Epoch {epoch}: Fine-tuned Missense Model (LLRs)", myc_sequence, amino_acids, base_dir)
        custom_plot_heatmap(myc_gene, myc_fs_diff_heatmap, f"Epoch {epoch}: Difference (Fine-tuned Missense - Original)", myc_sequence, amino_acids, base_dir)
        print(f"Plotted heatmap for Epoch{epoch} for gene MYC")

        # plot heatmap for tp53 gene 
        tp53_fs_heatmap, _ = generate_heatmap(tp53_sequence, model, tokenizer)
        tp53_fs_diff_heatmap = tp53_fs_heatmap - tp53_base_heatmap
        scaled_plot_heatmap(tp53_gene, tp53_fs_heatmap, f"Epoch {epoch}: Fine-tuned Missense Model (LLRs)", tp53_sequence, amino_acids, base_dir)
        custom_plot_heatmap(tp53_gene, tp53_fs_diff_heatmap, f"Epoch {epoch}: Difference (Fine-tuned Missense - Original)", tp53_sequence, amino_acids, base_dir)
        print(f"Plotted heatmap for Epoch{epoch} for gene TP53")

        # plot heatmap for rpl gene 
        rpl_fs_heatmap, _ = generate_heatmap(rpl_sequence, model, tokenizer)
        rpl_fs_diff_heatmap = rpl_fs_heatmap - rpl_base_heatmap
        scaled_plot_heatmap(rpl_gene, rpl_fs_heatmap, f"Epoch {epoch}: Fine-tuned Missense Model (LLRs)", rpl_sequence, amino_acids, base_dir)
        custom_plot_heatmap(rpl_gene, rpl_fs_diff_heatmap, f"Epoch {epoch}: Difference (Fine-tuned Missense - Original)", rpl_sequence, amino_acids, base_dir)
        print(f"Plotted heatmap for Epoch{epoch} for gene RPL2")

        # save loss after each epoch 
        with open(loss_file_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "during_train_loss", "after_train_loss", "valid_loss"])
            writer.writerows(loss_per_epoch)
        print(f"Saved loss per epoch to {loss_file_csv}")

        
    #  after loop finishes 
    print("\nTraining complete.")


    # Plot and save loss plot
    plot_loss(loss_per_epoch, descr="Missense Model Finetuning", base_dir=base_dir)
    return

    #return f"{descr} training complete. Best merged model saved to {final_dir}"


def main():
    # Set reproducibility
    set_seeds(0)

    # Configuration
    batch_size = 8  
    window_size = 1022
    model_params_millions = 650
    descr = "missense"
    max_epochs = 5

    # Load missense data
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "update4_all_ms_samples.parquet")

    # Split data into 75% train, 5% validate, 20% test 
    test_size = 0.20
    ms_train_df, ms_test_df = train_test_split(ms_df, test_size=test_size, random_state=0)
    valid_size = 0.0625 
    ms_train_df, ms_valid_df = train_test_split(ms_train_df, test_size=valid_size, random_state=0)
    print("Train, Valid, Test split is:", len(ms_train_df), len(ms_valid_df), len(ms_test_df))

    # Load original ESM-2 model
    model_path = f"/g/data/gi52/jaime/esm2_{model_params_millions}M_model"
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    base_model = EsmForMaskedLM.from_pretrained(model_path)

    # frozen baseline
    frozen_base_model = EsmForMaskedLM.from_pretrained(model_path)
    frozen_base_model.eval()
    for p in frozen_base_model.parameters():
        p.requires_grad = False

    '''
    # Load finetuned model from last saved checkpoint
    ckpt_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/missense/run8/epoch0_batch35000"

    # Load PEFT adapter onto base model
    model = PeftModel.from_pretrained(base_model, ckpt_path, is_trainable=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    '''
       
    # Tokenize and mask mutant sequences (mask the position where mutation occurs)
    mut_train_tokenized = mut_tokenize_and_mask_seqs(ms_train_df, tokenizer, 'mt_windowed_seq', window_size)
    mut_train_data = TorchDataset(mut_train_tokenized)

    # Tokenize and mask wildtype sequences (mask the position where mutation occurs)
    wt_train_tokenized = mut_tokenize_and_mask_seqs(ms_train_df, tokenizer, 'wt_windowed_seq', window_size)
    wt_train_data = TorchDataset(wt_train_tokenized)

    # Same for Validation data
    mut_valid_tokenized = mut_tokenize_and_mask_seqs(ms_valid_df, tokenizer, 'mt_windowed_seq', window_size)
    mut_valid_data = TorchDataset(mut_valid_tokenized)

    wt_valid_tokenized = mut_tokenize_and_mask_seqs(ms_valid_df, tokenizer, 'wt_windowed_seq', window_size)
    wt_valid_data = TorchDataset(wt_valid_tokenized)

    # Set up LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["query"],  # PEFT will insert LoRA into matching linear layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.TOKEN_CLS,  # Best fit for masked token modelling
    )

    t6 = timer()
    print('\n\nStarting training!')
    train_model(tokenizer, base_model, frozen_base_model, descr, mut_train_data, wt_train_data, mut_valid_data, wt_valid_data, lora_config, batch_size, max_epochs)

    t7 = timer()
    print(f"Total Time for training model: {t7 - t6:.4f} seconds")


if __name__ == '__main__':
    main()