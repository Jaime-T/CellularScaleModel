#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

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
    
def main():
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "ms_train_split.parquet")
    fs_df = pd.read_parquet(data_path / "fs_train_split.parquet")

    # Split Data 
        # create a validation set which is 0.8*0.25 = 0.2 of whole dataset
    valid_size = 0.25  
    ms_train_df, ms_valid_df = train_test_split(ms_df, test_size=valid_size, random_state=0)
    fs_train_df, fs_valid_df = train_test_split(fs_df, test_size=valid_size, random_state=0)

    # Data Processing
        # Load Tokeniser
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize all sequences in the "windowed_seq" column
    ms_tokenized = tokenize_seqs(ms_train_df, tokenizer, window_size)
    fs_tokenized = tokenize_seqs(fs_train_df, tokenizer, window_size)

    
    ms_input_ids = ms_tokenized["input_ids"]
    ms_attention_masks = ms_tokenized["attention_mask"]


    #df['input_ids'] = list(input_ids)
    #df['attention_mask'] = list(attention_mask)
    #print(df.head(10))

if __name__ == '__main__':
    main()
   