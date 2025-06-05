#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
from getSequence import getseq

def get_wildtype_sequence(uniprot_id):
    return getseq(str(uniprot_id), uniprot_id=True, just_sequence=True)
 
def main():
 
    depmap_df = pd.read_csv('/Users/jtaitz/Documents/Honours/datasets/depmap/OmicsSomaticMutations.csv', sep=',', low_memory=False)
    
    # Identify which samples have an associated UniProtID in the UniprotID column (many are empty, with no UniprotID)
    depmap_with_uniprotid = depmap_df[depmap_df['UniprotID'].notna()] # 407669 out of 718369 have uniprotID not null (uniprot_ids.count())
    
    # Extract the UniprotID from those samples, and print them
    uniprot_ids = depmap_with_uniprotid['UniprotID']

    # Turn into a list of strings
    uniprot_string_list = uniprot_ids.tolist()

    # Get the protein sequence from the sample's UniprotID using the getseq function from the getSequence package
    # Example is: seq = getseq(['Q9BY66-1', 'Q9BY66-1'], uniprot_id=True, just_sequence=True)

    # Make a new column for wildtype protein sequence, and mutant type sequence
    depmap_with_uniprotid['wildtype_seq'] = np.nan
    depmap_with_uniprotid['mutant_seq'] = np.nan

    # Add wildtype sequence to each row .. need to fix 
    depmap_with_uniprotid['wildtype_seq'] = depmap_with_uniprotid['UniprotID'].apply(get_wildtype_sequence)
    
    #For each row, add entries to the wildtype and mutant columns (this is currently very slow as searching same uniprotid multiple times)
    '''for row in depmap_with_uniprotid.itertuples():
        uniprot_id = row.UniprotID
        print('uniprot_id:', uniprot_id)
        wildtype_seq = getseq(str(uniprot_id), uniprot_id=True, just_sequence=True)
        row['wildtype_seq'] = wildtype_seq'''

    print(depmap_with_uniprotid.head(5))

    #print(depmap_df.head(10))
    #print(depmap_df.shape)


    # COSMIC DB:
    #cosmic_df = pd.read_csv('/Users/jtaitz/Documents/Honours/datasets/cosmic/Cosmic_MutantCensus_Tsv_v101_GRCh38/Cosmic_MutantCensus_v101_GRCh38.tsv', sep='\t',low_memory=False)
    #print(cosmic_df.head(10))
    #print(cosmic_df.shape)
    


if __name__ == '__main__':
  main()