#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 15:58:48 2025

@author: paceramateos
"""


import pandas as pd
import numpy as np
import json
import requests
import os
import urllib
import sys
import re
import ast
import sqlite3

def getResponse(url, outtype='dict'):
    operUrl = urllib.request.urlopen(url)
    if(operUrl.getcode()==200):
        data = operUrl.read()
        jsonData = json.loads(data)
    else:
        print("Error receiving data", operUrl.getcode())

    if outtype.lower() == 'dataframe':
        jsonData = pd.DataFrame.from_dict(jsonData)

    return jsonData

def get_ensembl_gene_info(goi='KRAS'):
    if goi[:4].lower() == 'ensg':
        url = f'http://rest.ensembl.org/lookup/id/{goi}'
        r = requests.get(url, headers={ "Content-Type" : "application/json"})
    else:
        url = f'http://rest.ensembl.org/lookup/symbol/homo_sapiens/{goi}'
        r = requests.get(url, headers={ "Content-Type" : "application/json"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()
    output = r.json()

    return output

## get gene stats and wildtype sequence (from ensembl)
def ensembl_geneseq(ensg = 'ENSG00000133703'):
    # ENSG = get_ensembl_id(goi)
    # ENSG = get_ensembl_gene_info(goi)['id']
    if ensg is not None:
        CDNA = requests.get('http://rest.ensembl.org/sequence/id/'+ensg+'?', headers={ "Content-Type" : "application/json"})
        data = CDNA.json()

        gene_stat = dict()
        _, _, chromosome, start, end, strand = data['desc'].split(':')
        gene_stat['chr'] = f'chr{chromosome}'
        gene_stat['start'] = int(start)
        gene_stat['end'] = int(end)
        gene_stat['strand'] = int(strand)
        gene_stat['length'] = len(data['seq'])

        WT_sequence = data['seq']

        return gene_stat, WT_sequence
    else:
        return None, None

# get transcript's intron positions
def get_enst_exon_info(enst='ENST00000311936'):

    url = f'http://rest.ensembl.org/lookup/id/{enst}?expand=1'
    r = requests.get(url, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    enst_annot = r.json()

    translation_start = enst_annot['Translation']['start']
    translation_end = enst_annot['Translation']['end']

    exon_start_list = [k['start'] for k in enst_annot['Exon']]
    exon_end_list = [k['end'] for k in enst_annot['Exon']]

    return translation_start,translation_end, sorted(exon_start_list), sorted(exon_end_list)

def extract_exon_sequences(dna_sequence, exon_starts, exon_ends):
    exon_sequences = []
    for start, end in zip(exon_starts, exon_ends):
        exon_sequences.append(dna_sequence[start:end+1])
    return exon_sequences

def complement_dna(dna_sequence):
    """
    Generates the complementary DNA sequence for a given DNA sequence.

    Parameters:
    dna_sequence (str): The original DNA sequence.

    Returns:
    str: The complementary DNA sequence.
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in dna_sequence)

def reverse_complement_dna(dna_sequence):
    """
    Generates the reverse complement of a DNA sequence.

    Parameters:
    dna_sequence (str): The original DNA sequence.

    Returns:
    str: The reverse complement of the DNA sequence.
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    # Generate the complement and then reverse it
    return ''.join(complement[base] for base in reversed(dna_sequence))


def dna_to_protein(dna_sequence):
    """
    Translate a DNA sequence into a protein sequence, using the standard genetic code.

    Args:
    dna_sequence (str): A string representing the DNA sequence (must be divisible by 3).

    Returns:
    str: The translated protein sequence.
    """
    # Genetic code dictionary
    genetic_code = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }

    protein_sequence = ""

    # Ensure the DNA sequence length is divisible by 3
    # Translate each codon to an amino acid
    for i in range(0, len(dna_sequence), 3):
        codon = dna_sequence[i:i+3]
        protein_sequence += genetic_code.get(codon, "?")  # Use '?' for unknown codons

    return protein_sequence



##### Add the exception to avoid the error and run it checking the previous output

geneinfo = get_ensembl_gene_info('MYC')

# get intron positions of canonical transcript
enst = re.sub('[.].*$','',geneinfo['canonical_transcript'])
print(enst)

gene_dic = ensembl_geneseq(enst)
translation_start, translation_end, exon_start_list, exon_end_list = get_enst_exon_info(enst)

exon_start_list_translated = sorted([translation_start]+[i for i in exon_start_list if i>translation_start and i<=translation_end])
exon_end_list_translated = sorted([i for i in  exon_end_list if i<translation_end and i>=translation_start]+[translation_end])

if geneinfo['strand'] == -1:
    DNA = reverse_complement_dna(gene_dic[-1])
else:
    DNA = gene_dic[-1]

gene_start_coor = gene_dic[0]['start']
CDS = []
for pos_exons in enumerate(exon_start_list_translated):
    CDS.append(DNA[exon_start_list_translated[pos_exons[0]]-gene_start_coor:exon_end_list_translated[pos_exons[0]]+1-gene_start_coor])

CDS = ''.join(CDS)

if geneinfo['strand'] == -1:
    CDS = reverse_complement_dna(CDS)

print(CDS)
protein = dna_to_protein(CDS)
#print(protein)
print(len(protein))

 