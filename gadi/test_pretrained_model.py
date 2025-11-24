# -*- coding: utf-8 -*-

"""
Validating pretrained model learning 

Progress tracking, plus model, tokeniser and optimiser saving 

Split data into 75% train, 5% validate, 20% test 

Changed masking function to only mask the mutation positions 
Using updated DepMap data, with 550281 total missense sequences 

Train test set and tokenize inputs

"""
import math
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
from torch.nn.utils import clip_grad_norm_
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

def train_model_test(model, train_dataset, test_dataset, lora_config, batch_size=6, lr=5e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    epochs = 1                              # set as needed
    max_grad_norm = 1.0                     # optional grad clipping

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)
    
    best_eval = float("inf")
    no_improve = 0
    stop_training = False
    min_delta = 1e-4
    patience_eval = 3

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        steps_per_epoch = len(train_loader)
        eval_every = max(1, math.ceil(0.05 * steps_per_epoch))   # <-- evaluate every 5%
        print(f"\n=== Epoch {epoch} ===")
        print(f"Steps per epoch: {steps_per_epoch} | Eval every: {eval_every} steps (~5%)")

        for step, batch in enumerate(train_loader, start=1):
            # ---------------------------
            # Training step
            # ---------------------------
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
            total_train_loss += float(loss)

            # ---------------------------
            # Periodic evaluation @ 5%
            # ---------------------------
            if (step % eval_every) == 0 or step == steps_per_epoch:
                avg_train_so_far = total_train_loss / step
                print(f"[Train] step {step}/{steps_per_epoch} | avg_loss_so_far={avg_train_so_far:.4f}")

                # Switch to eval and run a full pass on test_loader
                model.eval()
                eval_loss = 0.0
                n_eval_batches = 0

                with torch.inference_mode():
                    for ebatch in test_loader:
                        ebatch = {k: v.to(device, non_blocking=True) for k, v in ebatch.items()}
                        eout = model(**ebatch)
                        eloss = eout.loss
                        eval_loss += float(eloss)
                        n_eval_batches += 1
                avg_eval = eval_loss / max(1, n_eval_batches)
                print(f"[Eval ] after step {step}: avg_eval_loss={avg_eval:.4f}")

                # ---- early stopping check (within epoch) ----
                if avg_eval < (best_eval - min_delta):
                    best_eval = avg_eval
                    no_improve = 0
                else:
                    no_improve += 1
                    print(f"[EarlyStop] no improvement #{no_improve}/{patience_eval} "
                        f"(best={best_eval:.4f}, min_delta={min_delta})")
                    if no_improve >= patience_eval:
                        print("[EarlyStop] Stopping early due to plateau on eval loss.")
                        stop_training = True
                        break  # exit training loop for this epoch

                # Go back to training mode
                model.train()

        if stop_training:
            break  # exit outer epoch loop if triggered

        epoch_avg_train = total_train_loss / steps_per_epoch
        print(f"=== End Epoch {epoch} | avg_train_loss={epoch_avg_train:.4f} ===")



def main():
    # Set reproducibility
    set_seeds(0)

    # Configuration
    batch_size = 8
    window_size = 1022
    model_params_millions = 650

    # Load missense data
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "update2_all_ms_samples.parquet")

    # Split data into 75% train, 5% validate, 20% test 
    test_size = 0.20
    ms_train_df, ms_test_df = train_test_split(ms_df, test_size=test_size, random_state=0)
    valid_size = 0.0625
    ms_train_df, ms_valid_df = train_test_split(ms_train_df, test_size=valid_size, random_state=0)
    print("Missense - Train, Valid, Test split is:", len(ms_train_df), len(ms_valid_df), len(ms_test_df))
    
    # Load original ESM-2 model
    model_path = f"/g/data/gi52/jaime/esm2_{model_params_millions}M_model"
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)

    train_tokenized_df = tokenize_and_mask_seqs(ms_train_df, tokenizer, window_size)
    ms_train_dataset = TorchDataset(train_tokenized_df)

    valid_tokenized_df = tokenize_and_mask_seqs(ms_valid_df, tokenizer, window_size)
    ms_valid_dataset = TorchDataset(valid_tokenized_df)

    # Set up LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["query"],  # PEFT will insert LoRA into matching linear layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.TOKEN_CLS,  # Best fit for masked token modelling
    )

    print('\nStarting training and validating!')
    train_model_test(model, ms_train_dataset, ms_valid_dataset, lora_config, batch_size)


if __name__ == '__main__':
    main()