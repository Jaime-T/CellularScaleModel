# -*- coding: utf-8 -*-

"""
Progress tracking, plus model, tokeniser and optimiser saving 

Split data into 75% train, 5% validate, 20% test 

Changed masking function to only mask the mutation positions 
Using updated DepMap data, with 550281 total missense sequences 

Train test set and tokenize inputs

"""
import os
import tempfile
import pandas as pd
import re
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

def plot_loss(loss_values, descr, base_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', label='LOSS')
    plt.title(f'Loss vs Epochs for {descr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_dir}/loss_plot.png", dpi=300)
    plt.close()

def train_model(tokenizer, model, descr, train_dataset, lora_config, batch_size=6, epochs=3, lr=5e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # set up files to save progress to  
    base_dir = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/run6"
    os.makedirs(base_dir, exist_ok=True)

    progress_file = os.path.join(base_dir, "progress.pt")
    loss_file_csv = os.path.join(base_dir, "loss_per_epoch.csv")


    # Shuffle dataset once
    indices = np.random.permutation(len(train_dataset))
    shuffled_dataset = Subset(train_dataset, indices)
    train_loader = DataLoader(
        shuffled_dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True
    )

    print(f"Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")

    loss_per_epoch = []
    for epoch in range(epochs):
        print(f"\nStarting training for epoch {epoch+1}...")
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
            if (batch_idx+1) % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")


        avg_loss = total_loss / len(train_loader)
        epoch_time = timer() - epoch_start

        print(f"[{descr}] Epoch {epoch+1} avg training loss: {avg_loss:.4f} | time: {epoch_time:.2f}s")
        loss_per_epoch.append(avg_loss)

        # save after each epoch
        print(f"Saving checkpoint epoch{epoch+1} ...")
        checkpoint_dir = os.path.join(base_dir, f"epoch{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        print(f"Saved epoch{epoch+1} to {checkpoint_dir}")

        # save loss 
        with open(loss_file_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss"])
            for i, l in enumerate(loss_per_epoch, start=1):
                writer.writerow([i, l])
    

    # Save final merged model
    final_dir = os.path.join(base_dir, "final_merged")
    os.makedirs(final_dir, exist_ok=True)
    print("Merging and saving final model...")
    model = model.merge_and_unload()
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final {descr} model and tokenizer to {final_dir}")

    return f"{descr} training complete"


def main():
    t0 = timer()
    # Set reproducibility
    set_seeds(0)

    # Configuration
    batch_size = 8
    window_size = 1022
    model_params_millions = 650
    descr = "missense"

    # Load missense data
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "update2_all_ms_samples.parquet")
    t1 = timer()
    print(f"Time for loading missense data: {t1 - t0:.4f} seconds")

    # Split data into 75% train, 5% validate, 20% test 
    test_size = 0.20
    ms_train_df, ms_test_df = train_test_split(ms_df, test_size=test_size, random_state=0)
    valid_size = 0.0625
    ms_train_df, ms_valid_df = train_test_split(ms_train_df, test_size=valid_size, random_state=0)
    print("Train, Valid, Test split is:", len(ms_train_df), len(ms_valid_df), len(ms_test_df))
    t2 = timer()
    print(f"Time for splitting data: {t2 - t1:.4f} seconds")

    
    # Load original ESM-2 model
    model_path = f"/g/data/gi52/jaime/esm2_{model_params_millions}M_model"
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)

    t4 = timer()
    ms_tokenized_df = tokenize_and_mask_seqs(ms_train_df, tokenizer, window_size)
    ms_train_dataset = TorchDataset(ms_tokenized_df)
    t5 = timer()
    print(f"Time for tokenising and masking: {t5 - t4:.4f} seconds")
    print("Number of ms training samples:", len(ms_train_dataset))

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
    print('\nStarting training!')
    train_model(tokenizer, model, descr, ms_train_dataset, lora_config, batch_size)

    t7 = timer()
    print(f"Total Time for training model: {t7 - t6:.4f} seconds")
    # Plot and save loss plot


if __name__ == '__main__':
    main()