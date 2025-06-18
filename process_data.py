#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the distribution of types of variants of the mutated proteins. 
Check the distribution of sequence length of mutated proteins and plot graph.

Note: Change csv path in main() to where mutated data file is saved 
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyse(df):
    
    # Filter to non-null mt_protein_seq
    df = df[df['mt_protein_seq'].notnull()].copy()
    print(f"{len(df):,} records remain after filtering null mt_protein_seq")

    # VariantType counts
    print("\nVariantType frequencies:")
    counts = df['VariantType'].value_counts(dropna=False)
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
    plt.show()



def main():
    csv_path = '/Users/jaimetaitz/Downloads/OmicsSomaticMutations_with_protein_seqs.csv'
    # Load data 
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"\nLoaded {len(df):,} records from {csv_path}")

    analyse(df)




if __name__ == '__main__':
    main()

