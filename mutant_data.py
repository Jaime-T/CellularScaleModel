#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
 
 
def main():
 
    depmap_df = pd.read_csv('/Users/jtaitz/Documents/Honours/datasets/depmap/OmicsSomaticMutations.csv', sep=',', low_memory=False)
    print(depmap_df.head(10))
    print(depmap_df.shape)

    cosmic_df = pd.read_csv('/Users/jtaitz/Documents/Honours/datasets/cosmic/Cosmic_MutantCensus_Tsv_v101_GRCh38/Cosmic_MutantCensus_v101_GRCh38.tsv', sep='\t',low_memory=False)
    print(cosmic_df.head(10))
    print(cosmic_df.shape)
    print('hello')
if __name__ == '__main__':
  main()