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

def analyse(df: object):
    
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
    #plt.show()

    # Unique proteins 
    print(f"\nNumber of unique proteins by Hugo Symbol: {len(df['HugoSymbol'].unique())}")
    hugo_count = df['HugoSymbol'].value_counts()
    top = 100
    print(f"\nTop {top}: {hugo_count.head(top)}\n")

def filter(df: object, num: int):
    """ Filter out sequences that no protein sequence,  
        and filter out sequences in genes that have 
        less than a specified number of variants 
    """
    # Filter out null mt_protein_seq
    df = df[df['mt_protein_seq'].notnull()].copy()

    # Filter out low variant genes
    counts = df['HugoSymbol'].value_counts()
    valid = counts[counts > num].index
    df = df[df['HugoSymbol'].isin(valid)]
    return df


def filter_var(df: object, var: str):
    """Variant type can be: SNV, substitution, insertion, deletion
    """

    # Filter those with variant type
    snv = df[df['VariantType'] == str(var)]
    return snv

def main():

    # Load data 
    csv_path = '/Users/jaimetaitz/Downloads/OmicsSomaticMutations_with_protein_seqs.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"\nLoaded {len(df):,} records from {csv_path}")

    # Data Analysis 
    analyse(df)

    # Data Filtering
    df = filter(df, 10)
    print(df.shape)

    # Filter df by the variant type  
    snv_df = filter_var(df, 'SNV')
    del_df = filter_var(df, 'deletion')
    sub_df = filter_var(df, 'substitution')
    ins_df = filter_var(df, 'insertion')




if __name__ == '__main__':
    main()

