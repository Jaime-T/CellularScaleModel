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

    # Trim sequence to multiple of 3
    dna_seq = dna_seq[:len(dna_seq) - (len(dna_seq) % 3)]

    # Translate full sequence (stop codons appear as '*')
    protein = Seq(dna_seq).translate(to_stop=False)
    truncated_protein = protein.split('*')[0]  # stops at first stop if there is one
    return str(truncated_protein)


def get_transcript_seq(transcript_dict, transcript_id):
    # Try exact match first
    if transcript_id in transcript_dict:
        return str(transcript_dict[transcript_id].seq)
    
    # Fall back: check base IDs stored in rec.name
    for rec in transcript_dict.values():
      if rec.name == transcript_id.split(".")[0]:
        return str(rec.seq)
    
    # Not found
    return None

def main():

  # Step 1: load coding sequence data from Ensembl biomart in fasta format
  fasta_path = './data/update_cds_with_utr3.fasta'
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
    # Take the second last field as base, non versioned transript ID
    tid = rec.id.split("|")[-1]
    base_tid = rec.id.split("|")[-2]

    # reset the record's .id and clear description if you like
    rec.id = tid
    rec.name = base_tid
    rec.description = ""
    transcript_dict[tid] = rec

  # Step 2: load depmap mutation data from local file
  # OmicsSomaticMutations file at https://depmap.org/portal/data_page/?tab=allData
  OmicsSomaticMutations = pd.read_csv('./data/OmicsSomaticMutations.csv', sep=',', low_memory=False)

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
      base_tid = hgvs.split(".", 1)[0]
    else:
      continue 

    # check transcript seq is in fasta file and dict
    wt_seq = get_transcript_seq(transcript_dict, transcript_id)

    if (wt_seq is None) or (len(wt_seq) == 0):
      print(f"{transcript_id} has no seq ")
      continue

    # make mutant sequence 
    try:
      mut_seq = make_mutation(wt_seq, hgvs)
    except ValueError as e:
      continue
    
    # translate up to first stop
    prot_seq_mut = translate_until_stop(mut_seq)
    prot_seq_wt = translate_until_stop(wt_seq)

    # check if sequence exists 
    if (len(prot_seq_mut) == 0 or len(prot_seq_wt) == 0):
      continue

    # build a SeqRecord for the mutated CDS
    rec = SeqRecord(
      Seq(mut_seq),
      id=f"{transcript_id}|{hgvs}|{protein_chg}|{row['HugoSymbol']}|{row['VariantType']}",
      description=f"{hgvs}; wt protein up to stop: {prot_seq_wt}; mutant protein up to stop: {prot_seq_mut}"
      )
    
    # attach the protein as an annotation if you like
    rec.annotations["mt_protein"] = prot_seq_mut
    rec.annotations["wt_protein"] = prot_seq_wt

    mutated_records.append(rec)

    # Assign the protein sequences to the DataFrame
    OmicsSomaticMutations.at[idx, 'wt_protein_seq'] = prot_seq_wt
    OmicsSomaticMutations.at[idx, 'mt_protein_seq'] = prot_seq_mut

  # mutated_records now holds one SeqRecord per applied variant
  print(f"Generated {len(mutated_records)} mutated sequences")

  # After processing and updating the DataFrame
  output_path = './data/update_OmicsSomaticMutations_with_protein_seqs.csv'
  OmicsSomaticMutations.to_csv(output_path, index=False)

  # Save just the mutated sequences in a separate file 
  # Extract unique mutant protein sequences
  unique_proteins = {}
  for rec in mutated_records:
    prot_seq_mut = rec.annotations["mt_protein"]
    if prot_seq_mut not in unique_proteins:
      # Use protein seq as key, keep the first record ID as reference
      unique_proteins[prot_seq_mut] = rec.id
  
  print(f'number of unique protein seqs are {len(unique_proteins)}')

  # Convert to SeqRecords in FASTA format
  unique_records = [
    SeqRecord(Seq(prot), id=uid, description="unique mutant protein")
    for prot, uid in unique_proteins.items()
  ]

  # Write to fasta file
  with open("unique_mutant_proteins.fasta", "w") as output_handle:
    SeqIO.write(unique_records, output_handle, "fasta")

    
if __name__ == '__main__':
  main()