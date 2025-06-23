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
from transformers import AutoTokenizer, EsmModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import random

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
    # 1. Tokenize the batch
    encoded_seqs = tokenizer(
        batch['windowed_seq'].tolist(),
        padding="max_length",
        truncation=True,
        max_length=min(window_size, tokenizer.model_max_length),
        return_tensors="pt"  # use PyTorch for masking logic
    )

    input_ids = encoded_seqs["input_ids"]
    attention_mask = encoded_seqs["attention_mask"]

    # 2. Clone to create labels
    labels = input_ids.clone()

    # 3. Create probability mask (randomly choose tokens to mask)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # 4. Sample masked indices
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 5. Replace selected input_ids with [MASK] token
    input_ids[masked_indices] = tokenizer.mask_token_id

    # 6. Only keep labels for masked tokens
    labels[~masked_indices] = -100

    # 7. Return everything as numpy (optional, depending on downstream)
    return dict(
        input_ids=input_ids.numpy(),
        attention_mask=attention_mask.numpy(),
        labels=labels.numpy()
    )

def main():

    # set up Ray, a distributed computing framework
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    print(ray.cluster_resources())

    num_workers = 1
    num_devices = 1
    resources_per_worker = {"CPU": 8} # add later: , "GPU": 1 

    # Load data
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "ms_train_split.parquet")
    fs_df = pd.read_parquet(data_path / "fs_train_split.parquet")

    # Split Data 
        # create a validation set which is 0.8*0.25 = 0.2 of whole dataset
    valid_size = 0.25  
    ms_train_df, ms_valid_df = train_test_split(ms_df, test_size=valid_size, random_state=0)
    fs_train_df, fs_valid_df = train_test_split(fs_df, test_size=valid_size, random_state=0)

    # Data Processing - Load Tokeniser and Model
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Tokenize all sequences in the "windowed_seq" column
    window_size = 1022

    # Use new function combine tokenise and mask:
        # Use Ray DataFrame for batch processing
    ms_ray_ds = ray.data.from_pandas(ms_train_df)
    fs_ray_ds = ray.data.from_pandas(fs_train_df)

    #    Apply your combined tokenization + masking function
    ms_ray_ds = ms_ray_ds.map_batches(
        lambda batch: tokenize_and_mask_seqs(batch, tokenizer, window_size),
        batch_format="pandas"
    )

    fs_ray_ds = fs_ray_ds.map_batches(
        lambda batch: tokenize_and_mask_seqs(batch, tokenizer, window_size),
        batch_format="pandas"
    )
        


    exit()
    ms_tokenized = tokenize_seqs(ms_train_df, tokenizer, window_size)
    fs_tokenized = tokenize_seqs(fs_train_df, tokenizer, window_size)

    # Mask amino acid tokens for masked language modelling (MLM)
    ms_input_ids = ms_tokenized["input_ids"]
    ms_masked_input_ids, ms_labels = mask_input_ids(ms_input_ids, tokenizer)
    
    print("masked input ids:\n", ms_masked_input_ids[0])
    print("labels:\n", ms_labels[0])
    #print("decoded masked seqs:\n")
    #print(tokenizer.batch_decode(ms_masked_input_ids, skip_special_tokens=False))

    

    # Distributed Processing, ray dataset wraps batches 
    ray.data.DatasetContext.get_current().execution_options.preserve_order = (
        True  # deterministic
    )

    ms_ds = ray.data.read_parquet(data_path / "ms_train_split.parquet")
    ms_ds = ms_ds.random_shuffle(seed=0)
    
    fs_ds = ray.data.read_parquet(data_path / "fs_train_split.parquet")
    fs_ds = fs_ds.random_shuffle(seed=0)

    test_size = 0.25
    train_ds, val_ds = ds.train_test_split(test_size=test_size)


    # Train model on masked tokens, predict original tokens





    #ms_input_ids = ms_tokenized["input_ids"]
    #ms_attention_masks = ms_tokenized["attention_mask"]


    #df['input_ids'] = list(input_ids)
    #df['attention_mask'] = list(attention_mask)
    #print(df.head(10))

if __name__ == '__main__':
    main()
   