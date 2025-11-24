import pandas as pd
import re
import requests

def fetch_ensembl_sequence(transcript_id):
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{transcript_id}?type=utr"  # use type=cds, type=cdna, type=utr3, etc.
    
    r = requests.get(server + ext, headers={ "Content-Type" : "text/plain"})
    if not r.ok:
        return None
    return r.text

seq = fetch_ensembl_sequence("ENST00000646461")
print(seq)

seq = fetch_ensembl_sequence("ENST00000269305")
print(seq)
exit()

cds_fasta = "./data/update_mart_export_coding.txt"
utr3_fasta = "./data/update_mart_export_utr3.txt"
output_fasta = "./data/update_cds_with_utr3.fasta"
mutant = "./data/OmicsSomaticMutations_with_protein_seqs.csv"

# --- Count number of fasta entries (lines starting with '>') ---
def count_fasta_entries(fasta_file):
    count = 0
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                count += 1
    return count

cds_count = count_fasta_entries(cds_fasta)
utr3_count = count_fasta_entries(utr3_fasta)

combine_seq_count = count_fasta_entries(output_fasta)

print(f"Number of CDS fasta entries: {cds_count}")
print(f"Number of 3'UTR fasta entries: {utr3_count}")
print(f"Number of combined entries: {combine_seq_count}")
# --- Check number of rows in mutant file ---
mutant_df = pd.read_csv(mutant)
mutant_df = mutant_df[mutant_df["ProteinChange"].notna() & (mutant_df["ProteinChange"] != "")]
mutant_rows = len(mutant_df)

print(f"Number of rows in mutant file: {mutant_rows}")
print(f"Mutant file columns: {list(mutant_df.columns)}")


# --- Extract transcript IDs from DNAChange ---
mutant_df["TranscriptID"] = mutant_df["DNAChange"].str.extract(r'^(ENST\d+\.\d+)')

# --- Count unique transcript IDs ---
unique_ids = mutant_df["TranscriptID"].nunique()

print(f"Number of rows in mutant file: {len(mutant_df)}")
print(f"Number of unique transcript IDs: {unique_ids}")
print(f"Example transcript IDs: {mutant_df['TranscriptID'].dropna().unique()[:10]}")

mutant_ids = set(mutant_df["TranscriptID"].dropna())
print(f"Number of unique transcript IDs in mutant file: {len(mutant_ids)}")

# --- Extract transcript IDs from fasta headers ---
fasta_ids = set()
fasta_base_ids = set()
with open(output_fasta) as f:
    for line in f:
        if line.startswith(">"):
            # Look for ENST... with version inside header
            match = re.search(r'(ENST\d+\.\d+)', line)
            if match:
                fasta_ids.add(match.group(1))

            match = re.search(r'(ENST\d+)', line)
            if match:
                fasta_base_ids.add(match.group(1))


# --- Check coverage ---
missing_ids = mutant_ids - fasta_ids
print(f"Transcript IDs in mutant file but missing in fasta: {len(missing_ids)}")

if missing_ids:
    print("Example missing IDs:", list(missing_ids)[:10])

for tid in missing_ids:
    base_tid = tid.split('.')[0]
    if base_tid not in fasta_base_ids:
        print(f'{base_tid} {tid} is not in either list\n')



