# -*- coding: utf-8 -*-

"""
Test initial loss of pretrained ESM-2 model on missense mutation data.
Compute the loss on mutant (MT) and wildtype (WT) sequences in the validation set.

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

def mut_tokenize_and_mask_seqs(batch, tokenizer, window_size: int = 1022):
    # Tokenize the batch
    encoded_seqs = tokenizer(
        batch['mt_windowed_seq'].tolist(),
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


def main():
    # Set reproducibility
    set_seeds(0)

    # Configuration
    batch_size = 4  # per mutation and wildtype, so effective batch size is x2
    window_size = 1022
    model_params_millions = 650
    descr = "missense"
    max_epochs = 5

    # Load missense data
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "update3_all_ms_samples.parquet")

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)

    print("Loaded model, starting tokenisation and masking...")
    # Tokenise and mask Validation data
    mut_valid_tokenized = mut_tokenize_and_mask_seqs(ms_valid_df, tokenizer, window_size)
    mut_valid_data = TorchDataset(mut_valid_tokenized)

    wt_valid_tokenized = random_tokenize_and_mask_seqs(ms_valid_df, tokenizer, window_size)
    wt_valid_data = TorchDataset(wt_valid_tokenized)

    # Validation set data loaders
    mut_valid_loader = DataLoader(
        mut_valid_data, batch_size=batch_size,shuffle=False, pin_memory=True
    )
    wt_valid_loader = DataLoader(
        wt_valid_data, batch_size=batch_size,shuffle=False, pin_memory=True
    )
    print("Tokenisation and masking done.")
    # Compute loss on validation set 
    # MT val set 
    mut_val_loss = compute_loss(base_model, mut_valid_loader, device)
    print(f"Mutant Validation Loss: {mut_val_loss:.4f}")

    # WT val set 
    wt_val_loss = compute_loss(base_model, wt_valid_loader, device)
    print(f"Wildtype Validation Loss: {wt_val_loss:.4f}")

    # combined MT+WT val set
    combine_val_loader = DataLoader(torch.utils.data.ConcatDataset([mut_valid_data, wt_valid_data]), 
                                    batch_size=batch_size*2, shuffle=False, pin_memory=True)
    combined_loss = compute_loss(base_model, combine_val_loader, device)
    print(f"Combined Validation Loss: {combined_loss:.4f}")

    print(f"Initial Validation Loss - Mutant: {mut_val_loss:.4f}, Wildtype: {wt_val_loss:.4f}, Combined: {combined_loss:.4f}\n")

if __name__ == '__main__':
    main()