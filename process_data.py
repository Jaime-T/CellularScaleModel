#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the distribution of types of variants of the mutated proteins. 
Check the distribution of sequence length of mutated proteins and plot graph.

If sequences are longer than 1022 amino acids, randomly select a 
1022 aa window to include the mutation.

Categorise data into missense and frameshift variants and split into different
dataframes. Split data into test and train set and save to files in ./data folder.

Note: Change csv path in main() to where mutated data file is saved 

Code inspiration: https://github.com/naity/finetune-esm/blob/main/notebooks/cafa5_data_processing.ipynb
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import re
from pathlib import Path

def analyse(df: object):

    # VariantType counts
    print("\nVariantType frequencies:")
    counts = df['VariantType'].value_counts()
    print(counts.to_string())

    # VariantInfo counts
    print("\nVariantInfo frequencies:")
    counts = df['VariantInfo'].value_counts()
    print(counts.to_string())

    # Compute protein lengths 
    df['mt_protein_len'] = df['mt_protein_seq'].map(lambda x: len(str(x)))
    stats = df['mt_protein_len'].agg(['min', 'max', 'mean', 'median']).rename({
        'min': 'Minimum', 'max': 'Maximum', 'mean': 'Average', 'median': 'Median'
    })
    
    stats = df['mt_protein_len'].agg(['min', 'max', 'mean', 'median'])
    stats.index = ['Minimum', 'Maximum', 'Average', 'Median']

    print("\nProtein length stats:")
    print(stats.to_string(
        float_format='{:,.1f}'.format,
        header=True,
        index=True
    ))

    # Plot length distribution 
    bin_width = 10
    bins = np.arange(stats['Minimum'], stats['Maximum'] + bin_width, bin_width)

    plt.figure(figsize=(10, 6))
    plt.hist(df['mt_protein_len'], bins=bins,
             color='skyblue', edgecolor='black')
    plt.title('Distribution of Mutated Protein Sequence Lengths')
    plt.xlabel('Sequence Length (aa)')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, stats['Maximum'] + bin_width)
    plt.tight_layout()
    plt.savefig('./data/seq_len_distr.png')

    # Unique proteins 
    print(f"\nNumber of unique proteins by Hugo Symbol: {len(df['HugoSymbol'].unique())}")
    hugo_count = df['HugoSymbol'].value_counts()
    print(hugo_count)
    top = 10
    print(f"\nTop {top}: {hugo_count.head(top)}\n")

def filter_empty(df: object):
    # Filter out null/empty mutant protein sequenves
    df = df[df['mt_protein_seq'].notnull()].copy()
    return df

def slide_window(df: pd.DataFrame, window_size: int = 1022, seed: int = 42) -> pd.DataFrame:
    """
    If protein sequence length exceeds 1022 amino acids, select a random
    1022-aa window that includes the mutation site. Supports HGVS formats
    like p.A27K and p.A32VfsTer5.
    
    Args:
        df (pd.DataFrame): DataFrame with columns:
            - 'mt_protein_seq': the mutant protein sequence (string)
            - 'ProteinChange': HGVS notation (e.g., 'p.A27K', 'p.A32VfsTer5')
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Copy of df with a new column 'windowed_seq'
                     containing the 1022-aa windowed sequences.
    """
    random.seed(seed)
    windowed_seqs = []

    for idx, row in df.iterrows():
        seq = row['mt_protein_seq']
        hgvs = row['ProteinChange']
        gene = row['HugoSymbol']
        seq_len = len(seq)
        windowed_seq = seq  # default is full sequence

        # Extract mutation position using regex (e.g. p.A27K or p.A32VfsTer5)
        match = re.search(r'p\.\D+(\d+)', hgvs)
        if match:
            try:
                mut_pos = int(match.group(1)) - 1  # Convert to 0-based index
                

                if seq_len > window_size:
                    # Ensure the mutation is within the 1022-aa window
                    min_start = max(0, mut_pos - window_size + 1)
                    max_start = min(mut_pos, seq_len - window_size)

                    # Adjust in case mutation is too close to start or end
                    if max_start < min_start:
                        start = max(0, seq_len - window_size)
                    else:
                        start = random.randint(min_start, max_start)
                    windowed_seq = seq[start:start + window_size]
            except ValueError:
                pass  # fallback to full sequence on parsing error
        windowed_seqs.append(windowed_seq)

    df = df.copy()
    df['windowed_seq'] = windowed_seqs
    return df

def filter_missense(df: pd.DataFrame):
    return df[df['VariantInfo'].str.contains('missense_variant', na=False)]

def filter_frameshift(df: pd.DataFrame):
    return df[df['VariantInfo'].str.contains(r'frameshift_variant|stop_gained|stop_lost', na=False)]

def main():

    # Load data 
    csv_path = './data/update3_unique_mutant_proteins.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"\nLoaded {len(df):,} records from {csv_path}")

    # Data Filtering
    df = filter_empty(df)
    print(f'After filtering, there are {len(df)} sequences.')

    # Data Analysis 
    analyse(df)

    # Slide window 
    window_size = 1022
    seed = 42
    df = slide_window(df, window_size, seed)

    # Save as a new file to keep the original safe
    df.to_csv("./data/update_unique_mutant_proteins_with_window.csv", index=False)

    # Filter df by the variant type: missense and frameshift 
    ms_df = filter_missense(df) 
    fs_df = filter_frameshift(df) 

    # analyse ms and fs data
    print(f'\nMs data analysis: {len(ms_df)} sequences')
    analyse(ms_df)
    print(f'\nFs data analysis: {len(fs_df)} sequences')
    analyse(fs_df)

    # Save to file
    data_path = Path.cwd() / "data"

    # Create the folder if it doesn't exist
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Save
    df.to_parquet(data_path / "update_processed_data.parquet")
    ms_df.to_parquet(data_path / "update_all_ms_samples.parquet")
    fs_df.to_parquet(data_path / "update_all_fs_samples.parquet")

if __name__ == '__main__':
    main()

