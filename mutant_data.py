#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
import pickle
import os, re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

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

  raise ValueError(f"Unsupported HGVS pattern: {hgvs}")


""" Function: Translate a coding DNA sequence and return the protein
    up to (but not including) the first stop codon. """
def translate_until_stop(dna_seq: str) -> str:
  return str(Seq(dna_seq).translate(to_stop=True))


def main():

  # Step 1: load coding sequence data from Ensembl biomart in fasta format
  #fasta_path = '/Users/jaimetaitz/Downloads/mart_export_utr.txt'
  fasta_path = 'cds_with_3utr.fasta'
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

  # Step 4: Iterate through each row of mutations data table
  for _, row in OmicsSomaticMutations.iterrows():

    # we only care about the mutations that result in a protein change e.g. p.* annotation
    protein_chg = row["ProteinChange"]
    if not (isinstance(protein_chg, str) and protein_chg.startswith("p.")):
      continue
    
    hgvs = row["DNAChange"] # where hgvs is the format of the dna change 
    print('hgvs is:', hgvs)
    transcript_id = hgvs.split(":", 1)[0]

    wt_seq = str(transcript_dict[transcript_id].seq)
    try:
      mut_seq = make_mutation(wt_seq, hgvs)
    except ValueError as e:
      print(f"Error: Could not apply {hgvs}: {e}")
      continue
    
    # translate up to first stop
    prot_seq_mut = translate_until_stop(mut_seq)
    prot_seq_wt = translate_until_stop(wt_seq)

    # build a SeqRecord for the mutated CDS
    rec = SeqRecord(
      Seq(mut_seq),
      id=f"{tid}|{row['HugoSymbol']}|{protein_chg}",
      description=f"{hgvs}; wt protein up to stop: {prot_seq_wt}; mutant protein up to stop: {prot_seq_mut}"
      )
    print(rec)
    
    # attach the protein as an annotation if you like
    rec.annotations["protein"] = prot_seq_mut

    mutated_records.append(rec)

    #if row["DNAChange"] == "ENST00000433179.4:c.2258del":
    #  break
    

  # mutated_records now holds one SeqRecord per applied variant
  print(f"Generated {len(mutated_records)} mutated sequences")


  ''' COSMIC DB:
  #cosmic_df = pd.read_csv('/Users/jtaitz/Documents/Honours/datasets/cosmic/Cosmic_MutantCensus_Tsv_v101_GRCh38/Cosmic_MutantCensus_v101_GRCh38.tsv', sep='\t',low_memory=False)
  #print(cosmic_df.head(10))
  #print(cosmic_df.shape)'''
  
if __name__ == '__main__':
  main()