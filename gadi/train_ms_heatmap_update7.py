# -*- coding: utf-8 -*-

"""
Make heatmap after each epoch
5 epochs 
Rearranged train function
Progress tracking, plus model, tokeniser and optimiser saving 
Split data into 75% train, 5% validate, 20% test 

Changed masking function to only mask the mutation positions 
Using updated DepMap data, with 81446 frameshift sequences 

Train test set and tokenize inputs

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
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import torch.nn.functional as F
import csv

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

def tokenize_and_mask_seqs(batch, tokenizer, window_size: int = 1022):
    # Tokenize the batch
    encoded_seqs = tokenizer(
        batch['windowed_seq'].tolist(),
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

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Validation loss: {avg_loss:.4f}")

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

def plot_heatmap(gene, data, title, sequence, amino_acids, base_dir):
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

def train_model(tokenizer, base_model, frozen_base_model, descr, train_dataset, valid_dataset, 
                lora_config, batch_size=6, max_epochs=20, lr=5e-5, 
                patience=3,min_delta=1e-4):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # LoRA-wrapped trainable model
    model = get_peft_model(base_model, lora_config)
    optimizer = AdamW(model.parameters(), lr=lr)
    model.print_trainable_parameters()
    model.to(device)

    # directories
    base_dir = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/run7"
    os.makedirs(base_dir, exist_ok=True)

    best_model_dir = os.path.join(base_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    loss_file_csv = os.path.join(base_dir, "loss_per_epoch.csv")
    progress_file = os.path.join(base_dir, "progress.pt")

    # track loss
    loss_per_epoch = []
    best_val_loss = float("inf")
    no_improve = 0

    # save the raw datasets
    torch.save(train_dataset, os.path.join(base_dir, "train_dataset.pt"))
    torch.save(valid_dataset, os.path.join(base_dir, "valid_dataset.pt"))

    # save the current shuffle order so can reproduce
    train_indices = torch.randperm(len(train_dataset)) 
    torch.save(train_indices, os.path.join(base_dir, "train_indices.pt"))
    train_dataset_shuffled = Subset(train_dataset, train_indices)

    # loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset_shuffled, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size,shuffle=False, pin_memory=True
    )

    print(f"Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")

    # make heatmaps for frozen model:

        # myc
    myc_gene = "myc"
    myc_sequence = "MDFFRVVENQQPPATMPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPSEDIWKKFELLPTPPLSPSRRSGLCSPSYVAVTPFSLRGDNDGGGGSFSTADQLEMVTELLGGDMVNQSFICDPDDETFIKNIIIQDCMWSGFSAAAKLVSEKLASYQAARKDSGSPNPARGHSVCSTSSLYLQDLSAAASECIDPSVVFPYPLNDSSSPKSCASQDSSAFSPSSDSLLSSTESSPQGSPEPLVLHEETPPTTSSDSEEEQEDEEEIDVVSVEKRQAPGKRSESGSPSAGGHSKPPHSPLVLKRCHVSTHQHNYAAPPSTRKDYPAAKRVKLDSVRVLRQISNNRKCTSPRSSDTEENVKRRTHNVLERQRRNELKRSFFALRDQIPELENNEKAPKVVILKKATAYILSVQAEEQKLISEEDLLRKRREQLKHKLEQLRNSCA"
    myc_base_heatmap, amino_acids = generate_heatmap(myc_sequence, frozen_base_model, tokenizer)
    plot_heatmap(myc_gene, myc_base_heatmap, "Original ESM2 Model (LLRs)", myc_sequence, amino_acids, base_dir)

        # tp53
    tp53_gene = "tp53"
    tp53_sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    tp53_base_heatmap, amino_acids = generate_heatmap(tp53_sequence, frozen_base_model, tokenizer)
    plot_heatmap(tp53_gene, tp53_base_heatmap, "Original ESM2 Model (LLRs)", tp53_sequence, amino_acids, base_dir)
                
    # -- Training loop --
    for epoch in range(max_epochs):

        print(f"\nStarting training for epoch {epoch}...")
        epoch_start = timer()
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # save progress every batch
            torch.save({"epoch": epoch, "batch_idx": batch_idx}, progress_file)

            # make heatmap every 5000 batches
            if (batch_idx + 1) % 5000 == 0:  
                batch_num = batch_idx + 1 
                batch_loss = total_loss / batch_num
                print(f"[Epoch {epoch}] Batch {batch_num}/{len(train_loader)} | Avg Loss: {batch_loss:.4f}")

                # plot heatmap for myc gene 
                myc_fs_heatmap, _ = generate_heatmap(myc_sequence, model, tokenizer)
                myc_fs_diff_heatmap = myc_fs_heatmap - myc_base_heatmap
                plot_heatmap(myc_gene, myc_fs_heatmap, f"Epoch {epoch}, Batch {batch_num}: Fine-tuned Frameshift Model (LLRs)", myc_sequence, amino_acids, base_dir)
                plot_heatmap(myc_gene, myc_fs_diff_heatmap, f"Epoch {epoch}, Batch {batch_num}: Difference (Fine-tuned Frameshift - Original)", myc_sequence, amino_acids, base_dir)
                
                # plot heatmap for tp53 gene 
                tp53_fs_heatmap, _ = generate_heatmap(tp53_sequence, model, tokenizer)
                tp53_fs_diff_heatmap = tp53_fs_heatmap - tp53_base_heatmap
                plot_heatmap(tp53_gene, tp53_fs_heatmap, f"Epoch {epoch}, Batch {batch_num}: Fine-tuned Frameshift Model (LLRs)", tp53_sequence, amino_acids, base_dir)
                plot_heatmap(tp53_gene, tp53_fs_diff_heatmap, f"Epoch {epoch}, Batch {batch_num}: Difference (Fine-tuned Frameshift - Original)", tp53_sequence, amino_acids, base_dir)

        avg_train_loss = total_loss / len(train_loader)

        # save every epoch 
        epoch_dir = os.path.join(base_dir, f"epoch{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        torch.save(optimizer.state_dict(), os.path.join(epoch_dir, "optimizer.pt"))
        print(f"Saved checkpoint for epoch {epoch} in {epoch_dir}\n")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                val_loss += loss.item()

                if (i + 1) % 5000 == 0:  
                    batch_num = i + 1 
                    batch_val_loss = val_loss / batch_num
                    print(f"[Epoch {epoch}] Batch {batch_num}/{len(valid_loader)} | Avg Loss: {batch_val_loss:.4f}")
        avg_val_loss = val_loss / len(valid_loader)

        epoch_time = timer() - epoch_start
        print(f"Epoch {epoch}: train loss ={avg_train_loss:.4f}, valid loss ={avg_val_loss:.4f} | time: {epoch_time:.2f}s")
        
        loss_per_epoch.append((epoch, avg_train_loss, avg_val_loss))

        # plot heatmap for myc gene 
        myc_fs_heatmap, _ = generate_heatmap(myc_sequence, model, tokenizer)
        myc_fs_diff_heatmap = myc_fs_heatmap - myc_base_heatmap

        plot_heatmap(myc_gene, myc_fs_heatmap, f"Epoch {epoch}: Fine-tuned Frameshift Model (LLRs)", myc_sequence, amino_acids, base_dir)
        plot_heatmap(myc_gene, myc_fs_diff_heatmap, f"Epoch {epoch}: Difference (Fine-tuned Frameshift - Original)", myc_sequence, amino_acids, base_dir)
        print(f"Plotted heatmap for Epoch{epoch} for gene MYC")

        # plot heatmap for tp53 gene 
        tp53_fs_heatmap, _ = generate_heatmap(tp53_sequence, model, tokenizer)
        tp53_fs_diff_heatmap = tp53_fs_heatmap - tp53_base_heatmap
        plot_heatmap(tp53_gene, tp53_fs_heatmap, f"Epoch {epoch}: Fine-tuned Frameshift Model (LLRs)", tp53_sequence, amino_acids, base_dir)
        plot_heatmap(tp53_gene, tp53_fs_diff_heatmap, f"Epoch {epoch}: Difference (Fine-tuned Frameshift - Original)", tp53_sequence, amino_acids, base_dir)
        print(f"Plotted heatmap for Epoch{epoch} for gene TP53")

        # save loss after each epoch 
        with open(loss_file_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "valid_loss"])
            writer.writerows(loss_per_epoch)

        # ---- early stopping check + save best model ----
        if avg_val_loss < (best_val_loss - min_delta):
            best_val_loss = avg_val_loss
            no_improve = 0
            # Save best model 
            print(f"New best model at epoch {epoch} (val_loss={avg_val_loss:.4f})")
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            torch.save(optimizer.state_dict(), os.path.join(best_model_dir, "optimizer.pt"))
        else:
            no_improve += 1
            print(f"[EarlyStop] No improvement #{no_improve}/{patience}")
            if no_improve >= patience:
                print(f"Stopping early at epoch {epoch}")
                break

        
    #  after loop finishes 
    print("\nTraining complete.")
    '''
    # reload best model (with adapters) and merge 
    print("Loading best model for merging...")
    best_model = EsmForMaskedLM.from_pretrained(best_model_dir)
    best_model = get_peft_model(best_model, lora_config)  # reattach adapter config
    best_model.load_adapter(best_model_dir, "default")   # load LoRA adapter
    best_model = best_model.merge_and_unload()           # merge into base

    # Save final merged best model
    final_dir = os.path.join(base_dir, "final_merged")
    os.makedirs(final_dir, exist_ok=True)
    best_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f" Saved final merged BEST {descr} model to {final_dir}")
    '''

    # Plot and save loss plot
    plot_loss(loss_per_epoch, descr="Frameshift Model Finetuning", base_dir=base_dir)
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
    ms_test_df_df = pd.read_parquet(data_path / "update2_all_ms_samples.parquet")

    # Split data into 75% train, 5% validate, 20% test 
    test_size = 0.20
    ms_test_df_train_df, ms_test_df_test_df = train_test_split(ms_test_df_df, test_size=test_size, random_state=0)
    valid_size = 0.0625 
    ms_test_df_train_df, ms_test_df_valid_df = train_test_split(ms_test_df_train_df, test_size=valid_size, random_state=0)
    print("Train, Valid, Test split is:", len(ms_test_df_train_df), len(ms_test_df_valid_df), len(ms_test_df_test_df))

    # Load original ESM-2 model
    model_path = f"/g/data/gi52/jaime/esm2_{model_params_millions}M_model"
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    base_model = EsmForMaskedLM.from_pretrained(model_path)

    # frozen baseline
    frozen_base_model = EsmForMaskedLM.from_pretrained(model_path)
    frozen_base_model.eval()
    for p in frozen_base_model.parameters():
        p.requires_grad = False


    train_tokenized_df = tokenize_and_mask_seqs(ms_test_df_train_df, tokenizer, window_size)
    ms_test_df_train_dataset = TorchDataset(train_tokenized_df)

    valid_tokenized_df = tokenize_and_mask_seqs(ms_test_df_valid_df, tokenizer, window_size)
    ms_test_df_valid_dataset = TorchDataset(valid_tokenized_df)

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
    train_model(tokenizer, base_model, frozen_base_model, descr, ms_test_df_train_dataset, ms_test_df_valid_dataset, lora_config, batch_size, max_epochs)

    t7 = timer()
    print(f"Total Time for training model: {t7 - t6:.4f} seconds")


if __name__ == '__main__':
    main()