#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
import pickle
import os, re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from test_variant import compare_proteins # 

""" Function: Apply a mutation to a coding nucleotide sequence 
  and return the mutated sequence """
def make_mutation(seq: str, hgvs: str) -> str:

  # case 1) substitution/missense mutation e.g. c.563C>T
  m = re.match(r".+:c\.(\d+)([ACGT])>([ACGT])$", hgvs)
  if m:
    pos = int(m.group(1)) - 1
    return seq[:pos] + m.group(3) + seq[pos+1:]

  # case 2) insertion e.g. c.21_22insT
  m = re.match(r".+:c\.(\d+)_(\d+)ins([ACGT]+)$", hgvs)
  if m:
    a = int(m.group(1))
    ins = m.group(3)
    return seq[:a] + ins + seq[a:]

  # case 3) deletion e.g. c.80_84del or c.80del
  m = re.match(r".+:c\.(\d+)_(\d+)del$", hgvs)
  if m:
    a, b = int(m.group(1)) - 1, int(m.group(2))
    return seq[:a] + seq[b:]
  m = re.match(r".+:c\.(\d+)del$", hgvs)
  if m:
    p = int(m.group(1)) - 1
    return seq[:p] + seq[p+1:]

  # case 4) inversion: change to reverse complement e.g. c.18_19inv
  m = re.match(r".+:c\.(\d+)_(\d+)inv$", hgvs)
  if m:
    a, b = int(m.group(1)) - 1, int(m.group(2))
    frag = seq[a:b]
    inv  = str(Seq(frag).reverse_complement())
    return seq[:a] + inv + seq[b:]

  # case 5) duplication e.g. c.1_2dup or c.52dup
  m = re.match(r".+:c\.(\d+)_(\d+)dup$", hgvs)
  if m:
    a, b = int(m.group(1)) - 1, int(m.group(2))
    return seq[:b] + seq[a:b] + seq[b:]
  m = re.match(r".+:c\.(\d+)dup$", hgvs)
  if m:
    p = int(m.group(1))
    return seq[:p] + seq[p-1:p] + seq[p:]

  #case 6) deletion-insertion (delins) e.g. c.1_3delinsG or c.56delinsAC 
  m = re.match(r".+:c\.(\d+)_(\d+)delins([ACGT]+)$", hgvs)
  if m:
    a, b, ins = int(m.group(1)) - 1, int(m.group(2)), m.group(3)
    return seq[:a] + ins + seq[b:]
  
  m = re.match(r".+:c\.(\d+)delins([ACGT]+)$", hgvs)
  if m:
    a, ins = int(m.group(1)) - 1, m.group(2)
    return seq[:a] + ins + seq[a+1:]

  # case 7) affects splicing site
  m = re.match(r".+:c\.(\d+)_(\d+)\+1ins([ACGT]+)$", hgvs)
  if m:
    a, ins = int(m.group(1)), m.group(3)
    return seq[:a] + ins + seq[a:]

  m = re.match(r".+:c\.(\d+)-1_(\d+)ins([ACGT]+)$", hgvs)
  if m:
    a, ins = int(m.group(1)) - 1, m.group(3)
    return seq[:a] + ins + seq[a:]

  raise ValueError(f"Unsupported HGVS pattern: {hgvs}")


""" Function: Translate a coding DNA sequence and return the protein
    up to (but not including) the first stop codon. """
def translate_until_stop(dna_seq: str) -> str:
  seq_obj = Seq(dna_seq)
  protein = seq_obj.translate(to_stop=False)
  
  if '*' not in protein:
      return ''
  # Truncate the protein sequence at the first stop codon
  truncated_protein = protein.split('*')[0]
  return str(truncated_protein)
  #return str(Seq(dna_seq).translate(to_stop=True))


def main():

  # Step 1: load coding sequence data from Ensembl biomart in fasta format
  #fasta_path = '/Users/jaimetaitz/Downloads/mart_export_utr.txt'
  fasta_path = 'cds_with_utr3.fasta'
  if not os.path.isfile(fasta_path):
    raise FileNotFoundError(f"Could not find {fasta_path}")

  # parse into SeqRecord objects
  records = SeqIO.parse(fasta_path, "fasta")

  # store in a dict: { transcript_id: SeqRecord }
  transcript_dict = {}
  for rec in records:
    # header looks like:
    #   ENSG00000001461|ENSG00000001461.19|ENST00000003912|ENST00000003912.9
    # split on '|' and take the last field as the transcript versioned ID
    tid = rec.id.split("|")[-1]
    
    # reset the record's .id and clear description if you like
    rec.id = tid
    rec.description = ""
    transcript_dict[tid] = rec

  # Step 2: load depmap mutation data from local file
  OmicsSomaticMutations = pd.read_csv('/Users/jaimetaitz/Downloads/OmicsSomaticMutations.csv', sep=',', low_memory=False)

  # Step 3: make container for mutated seqRecords
  mutated_records = []

  # Step 4: initialise empty new columns for protein wt and mut type seqs
  OmicsSomaticMutations['wt_protein_seq'] = None
  OmicsSomaticMutations['mt_protein_seq'] = None

  # Step 5: Iterate through each row of mutations data table
  for idx, row in OmicsSomaticMutations.iterrows():

    # we only care about the mutations that result in a protein change e.g. p.* annotation
    protein_chg = row["ProteinChange"]
    if not (isinstance(protein_chg, str) and protein_chg.startswith("p.")):
      continue
    
    hgvs = row["DNAChange"] # where hgvs is the format of the dna change 

    # check hgvs is not empty 
    if isinstance(hgvs, str):
      transcript_id = hgvs.split(":", 1)[0]
    else:
      continue 

    # check transcript seq is in fasta file and dict
    if transcript_id in transcript_dict:
      wt_seq = str(transcript_dict[transcript_id].seq)
      
    else: 
      continue 

    # make mutant sequence 
    try:
      mut_seq = make_mutation(wt_seq, hgvs)
    except ValueError as e:
      print(f"Error: Could not apply {hgvs}: {e}")
      continue
    
    # translate up to first stop
    prot_seq_mut = translate_until_stop(mut_seq)
    prot_seq_wt = translate_until_stop(wt_seq)

    # check if sequence exists i.e. there is a stop codon
    if (len(prot_seq_mut) == 0 or len(prot_seq_wt) == 0):
      continue

    # build a SeqRecord for the mutated CDS
    rec = SeqRecord(
      Seq(mut_seq),
      id=f"{tid}|{hgvs}|{protein_chg}|{row['HugoSymbol']}|{row['VariantType']}",
      description=f"{hgvs}; wt protein up to stop: {prot_seq_wt}; mutant protein up to stop: {prot_seq_mut}"
      )
    
    # attach the protein as an annotation if you like
    rec.annotations["mt_protein"] = prot_seq_mut
    rec.annotations["wt_protein"] = prot_seq_wt

    mutated_records.append(rec)

    # Assign the protein sequences to the DataFrame
    OmicsSomaticMutations.at[idx, 'wt_protein_seq'] = prot_seq_wt
    OmicsSomaticMutations.at[idx, 'mt_protein_seq'] = prot_seq_mut

    """if row["DNAChange"] == "ENST00000486637.2:c.276-1_276insGCT":
      mutant_pos = compare_proteins(prot_seq_wt, prot_seq_mut)
      ter = len(prot_seq_mut) - int(mutant_pos)
      print('mt length is', len(prot_seq_mut), 'length wt is', len(prot_seq_wt))
      print('terminates after ', ter, 'amino acids')
      print('mutant seq', mut_seq)
      print('wt seq', wt_seq)
      print("\n")
      print('wt prot', prot_seq_wt)
      print('mt prot', prot_seq_mut)
      break"""

  # mutated_records now holds one SeqRecord per applied variant
  print(f"Generated {len(mutated_records)} mutated sequences")

  # After processing and updating the DataFrame
  output_path = '/Users/jaimetaitz/Downloads/OmicsSomaticMutations_with_protein_seqs.csv'
  OmicsSomaticMutations.to_csv(output_path, index=False)

  ''' COSMIC DB:
  #cosmic_df = pd.read_csv('/Users/jtaitz/Documents/Honours/datasets/cosmic/Cosmic_MutantCensus_Tsv_v101_GRCh38/Cosmic_MutantCensus_v101_GRCh38.tsv', sep='\t',low_memory=False)
  #print(cosmic_df.head(10))
  #print(cosmic_df.shape)'''
  
if __name__ == '__main__':
  main()