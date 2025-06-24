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


def tokenize_seqs(batch, tokenizer, window_size: int = 1022):
    encoded_seqs = tokenizer(
        batch['windowed_seq'].tolist(),        
        padding="max_length",                 
        truncation=True,
        max_length=min(window_size, tokenizer.model_max_length),
        return_tensors="np",  # or "pt" for PyTorch, "tf" for TensorFlow
    )
    return dict(
        input_ids=encoded_seqs["input_ids"],
        attention_mask=encoded_seqs["attention_mask"],
    )

def mask_input_ids(input_ids, tokenizer, mlm_probability=0.15):
    labels = input_ids.copy()

    # Create a mask of which tokens to mask
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Replace selected tokens with the mask token
    input_ids[masked_indices] = tokenizer.mask_token_id

    # Only compute loss on masked tokens
    labels[~masked_indices] = -100

    return input_ids, labels

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

    # Return everything as numpy (optional, depending on downstream)
    return dict(
        input_ids=input_ids.numpy(),
        attention_mask=attention_mask.numpy(),
        labels=targets.numpy()
    )

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def train(model, train_loader, val_loader, epochs=3, lr=5e-5, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

        # Optional: Add validation loss
        if val_loader:
            evaluate(model, val_loader, device)

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

def main():
    # Set reproducibility 
    set_seeds(0)

    # Initialise Ray, a distributed computing framework
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    #print(ray.cluster_resources())

    # Set ray to be deterministic 
    ray.data.DatasetContext.get_current().execution_options.preserve_order = (
        True  
    )

    num_workers = 1
    num_devices = 1
    resources_per_worker = {"CPU": 8} # add later: , "GPU": 1 

    # Load data
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "ms_train_split.parquet")
    fs_df = pd.read_parquet(data_path / "fs_train_split.parquet")

    # Split Data (80% train, 20% validate)
    valid_size = 0.25  
    ms_train_df, ms_valid_df = train_test_split(ms_df, test_size=valid_size, random_state=0)
    fs_train_df, fs_valid_df = train_test_split(fs_df, test_size=valid_size, random_state=0)

    # Load Tokeniser and Model
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()  

    window_size = 1022

    # Convert DataFrames to Ray for batch processing
    ms_ray_ds = ray.data.from_pandas(ms_train_df)
    fs_ray_ds = ray.data.from_pandas(fs_train_df)

    # Tokenize and mask 
    ms_ray_ds = ms_ray_ds.map_batches(
        lambda batch: tokenize_and_mask_seqs(batch, tokenizer, window_size),
        batch_format="pandas"
    )

    fs_ray_ds = fs_ray_ds.map_batches(
        lambda batch: tokenize_and_mask_seqs(batch, tokenizer, window_size),
        batch_format="pandas"
    )

    
    # Convert Ray Datasets to PyTorch Datasets
    ms_train_dataset = RayTorchDataset(ms_ray_ds)
    fs_train_dataset = RayTorchDataset(fs_ray_ds)

    # Create DataLoaders
    batch_size = 8
    ms_loader = DataLoader(ms_train_dataset, batch_size=batch_size, shuffle=True)
    fs_loader = DataLoader(fs_train_dataset, batch_size=batch_size, shuffle=True)

    # Train on one dataset for now (e.g. ms)
    train(model, ms_loader, val_loader=None, epochs=3)


    exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  

    # Set up LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["query"],  # PEFT will insert LoRA into matching linear layers
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.TOKEN_CLS,  # Best fit for masked token modelling
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    wildtype = ["MKTFFVAGV<mask>AGK", "GV<mask>AGK"]  # Random dummy sequence with one mask
    inputs = tokenizer(wildtype, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # optimiser  setup
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = optim.Adam(lora_params, lr=1e-3)

    # Train model on masked tokens, predict original tokens
    # Training loop
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        prediction = logits[mask_token_index]
        target_id = tokenizer.convert_tokens_to_ids("G")  # Assume we want 'G' at <mask>
        loss = nn.CrossEntropyLoss()(prediction, torch.tensor([target_id], device=device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        prediction = logits[mask_token_index]
        probs = torch.softmax(prediction, dim=-1)
        topk = torch.topk(probs, k=5, dim=-1)
        top_tokens = tokenizer.convert_ids_to_tokens(topk.indices[0].tolist())
        print("Top predictions at <mask>:", top_tokens)


    

    



if __name__ == '__main__':
    main()
   