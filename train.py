#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train test set and tokenize inputs

Code inspiration: https://github.com/naity/finetune-esm/blob/main/notebooks/cafa5_train.ipynb

"""
import os
import ray
from ray.data.dataset import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, EsmModel, AutoModelForMaskedLM
from transformers import EsmForMaskedLM, EsmTokenizer 
import random
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class RayTorchDataset(Dataset):
    def __init__(self, ray_dataset):
        self.data = ray_dataset.to_pandas()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
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

def get_base_model():
    model_name = "facebook/esm2_t6_8M_UR50D"
    model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    return model, tokenizer

@ray.remote(num_cpus=8, num_gpus=1)
def train_model(model_name, train_dataset, lora_config, batch_size, epochs=3, lr=5e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Re-initialize model and tokenizer inside the remote function
    model, tokenizer = get_base_model()
    model = get_peft_model(model, lora_config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[{model_name}] Epoch {epoch+1} training loss: {avg_loss:.4f}")

    # Save model
    model = model.merge_and_unload()
    model.save_pretrained(f"./train/{model_name}")
    tokenizer.save_pretrained(f"./train/{model_name}")

    return f"{model_name} training complete"


def main():
    # Set reproducibility 
    set_seeds(0)

    # Configuration
    num_workers = 1
    num_devices = 1
    resources_per_worker = {"CPU": 8, "GPU": 1 } 
    batch_size = 8
    window_size = 1022

    # Initialise Ray, a distributed computing framework
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        num_cpus=num_workers * resources_per_worker["CPU"],
        num_gpus=num_workers * resources_per_worker["GPU"]
    )
    print(ray.cluster_resources())

    # Set ray to be deterministic 
    ray.data.DatasetContext.get_current().execution_options.preserve_order = (
        True  
    )

    # Load data
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "ms_train_split.parquet")
    fs_df = pd.read_parquet(data_path / "fs_train_split.parquet")

    # Split Data (80% train, 20% validate)
    valid_size = 0.25  
    ms_train_df, ms_valid_df = train_test_split(ms_df, test_size=valid_size, random_state=0)
    fs_train_df, fs_valid_df = train_test_split(fs_df, test_size=valid_size, random_state=0)

    # Convert DataFrames to Ray Datasets for batch processing
    ms_ray_ds = ray.data.from_pandas(ms_train_df)
    fs_ray_ds = ray.data.from_pandas(fs_train_df)

    # Tokenize and mask 
    model, tokenizer = get_base_model()
    ms_ray_ds = ms_ray_ds.map_batches(
        lambda batch: tokenize_and_mask_seqs(batch, tokenizer, window_size),
        batch_format="pandas",
    )

    fs_ray_ds = fs_ray_ds.map_batches(
        lambda batch: tokenize_and_mask_seqs(batch, tokenizer, window_size),
        batch_format="pandas",
    )
    
    # Convert Ray Datasets to PyTorch Datasets
    ms_train_dataset = RayTorchDataset(ms_ray_ds)
    fs_train_dataset = RayTorchDataset(fs_ray_ds)

    # Set up LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["query"],  # PEFT will insert LoRA into matching linear layers
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,   # ! Changed from TOKEN_CLS !
    )

    print('\n\nstarting training!')
    # Launch Ray training jobs
    ms_future = train_model.remote("model_missense", ms_train_dataset, lora_config, batch_size)
    fs_future = train_model.remote("model_frameshift", fs_train_dataset, lora_config, batch_size)
    print('\n\nTraining done!')
    # Wait for both jobs to finish
    ray.get([ms_future, fs_future])



if __name__ == '__main__':
    main()
   
