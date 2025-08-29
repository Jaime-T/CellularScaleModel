# -*- coding: utf-8 -*-

"""
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
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F

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

def tokenize_and_mask_seqs(batch, tokenizer, window_size: int = 1022, mlm_probability: float = 0.15):
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

        print(f'sample is {batch}')
        # Extract mutation position, e.g. p.A27K → 27
        m = re.match(r"p\.\D+(\d+)", prot_change)
        if m:
            mut_pos = int(m.group(1))

            # Convert to window-relative position
            window_pos = mut_pos - start_index  

            token_index = window_pos 

            print(f'Protein change is {prot_change}, and the mutation position is {mut_pos}, token is at {token_index}')
            print(f'start index: {batch['start_index']}')

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

###===###
def testing_pretrained_model(lora_config, model, train_dataset, batch_size=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print(f"Batch size: {batch_size}, Batches per epoch: {len(test_loader)}")

    for epoch in range(1):
        print(f"Starting testing for epoch {epoch+1}...")
        model.eval()
        total_loss = 0
        for b_itr, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            if np.mod(b_itr+1, int(round(len(test_loader)/20))) == 0:
                print(f"Progress: {b_itr+1}/{len(test_loader) } -- {loss}")

        avg_loss = total_loss / len(test_loader)
        print("#---")
        print(f"Total loss: {total_loss}")
        print(f"Avg loss: {avg_loss}")

    return total_loss
###===###

def train_model(tokenizer, model, descr, train_dataset, lora_config, batch_size=6, epochs=3, lr=5e-5):
    t0 = timer()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    t1 = timer()
    print(f"Setup time: {t1 - t0:.3f}s")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    t2 = timer()
    print(f"DataLoader init time: {t2 - t1:.3f}s")
    print(f"Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")

    loss_per_epoch = []
    base_dir = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/run5"
    os.makedirs(base_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nStarting training for epoch {epoch+1}...")
        epoch_start = timer()
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            batch_start = timer()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            batch_time = timer() - batch_start
            if i % 1000 == 0:
                print(f"Batch {i} time: {batch_time:.3f}s")

        avg_loss = total_loss / len(train_loader)
        print(f"[{descr}] Epoch {epoch+1} training loss: {avg_loss:.4f}")
        loss_per_epoch.append(avg_loss)

        epoch_time = timer() - epoch_start
        print(f"Epoch {epoch+1} time: {epoch_time:.3f}s")

        # save after each epoch
        save_start = timer()
        print(f"Saving checkpoint epoch{epoch+1} ...")
        checkpoint_dir = os.path.join(base_dir, f"epoch{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved epoch{epoch+1} to {checkpoint_dir}")
        print(f"Save time: {timer() - save_start:.3f}s")


    # Save model
    save_final = timer()
    final_dir = os.path.join(base_dir, "final_merged")
    os.makedirs(final_dir, exist_ok=True)
    print("Merging and saving final model...")
    model = model.merge_and_unload()
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final {descr} model and tokenizer to {final_dir}")
    print(f"Save final model time: {timer() - save_final:.3f}s")

    # Plot and save loss plot
    plot_loss(loss_per_epoch, descr, base_dir)

    return f"{descr} training complete"



def main():
    t0 = timer()
    # Set reproducibility
    set_seeds(0)

    # Configuration
    batch_size = 8
    window_size = 1022
    model_params_millions = 650
    descr = "frameshift"

    # Load missense data
    data_path = Path("./data")
    fs_df = pd.read_parquet(data_path / "update2_all_fs_samples.parquet")
    t1 = timer()
    print(f"Time for loading frameshift data: {t1 - t0:.4f} seconds")

    # Split data into 60% train, 20% validate, 20% test 
    test_size = 0.2
    fs_train_df, fs_test_df = train_test_split(fs_df, test_size=test_size, random_state=0)
    valid_size = 0.25  
    fs_train_df, fs_valid_df = train_test_split(fs_train_df, test_size=valid_size, random_state=0)
    t2 = timer()
    print(f"Time for splitting data: {t2 - t1:.4f} seconds")

    # Analyse training samples
    # Count HugoSymbol frequencies and select top N
    hugo_counts = fs_train_df['HugoSymbol'].value_counts()
    top_genes = hugo_counts.head(10).index.tolist()

    # Filter rows and print relevant columns
    filtered = (
        fs_train_df[fs_train_df['HugoSymbol'].isin(top_genes)]
        [['HugoSymbol', 'ProteinChange', 'windowed_seq']]
    )
    print(f"Showing rows for top 10 most frequent HugoSymbols:\n{filtered}")

    t3 = timer()
    print(f"Time for analysing training samples: {t3 - t2:.4f} seconds")

    # Load original ESM-2 model
    model_path = f"/g/data/gi52/jaime/esm2_{model_params_millions}M_model"
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)
    t4 = timer()
    print(f"Time for loading esm model: {t4 - t3:.4f} seconds")

    fs_tokenized_df = tokenize_and_mask_seqs(fs_train_df, tokenizer, window_size)
    fs_train_dataset = TorchDataset(fs_tokenized_df)
    t5 = timer()
    print(f"Time for tokenising and masking: {t5 - t4:.4f} seconds")
    print("Number of ms training samples:", len(fs_train_dataset))

    # Set up LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["query"],  # PEFT will insert LoRA into matching linear layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.TOKEN_CLS,  # Best fit for masked token modelling
    )

    print('\nTesting pretrained model')
    #testing_pretrained_model(lora_config, model, fs_train_dataset, batch_size)
    t6 = timer()
    print(f"Time for testing pretrained model: {t6 - t5:.4f} seconds")

    print('\n\nStarting training!')
    train_model(tokenizer, model, descr, fs_train_dataset, lora_config, batch_size)
    t7 = timer()
    print(f"Total Time for training model: {t7 - t6:.4f} seconds")



if __name__ == '__main__':
    main()