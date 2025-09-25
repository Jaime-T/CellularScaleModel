from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import re

# Define input and output file paths
cds_fasta = "./data/update_mart_export_coding.txt"
utr3_fasta = "./data/update_mart_export_utr3.txt"
output_fasta = "./data/update_cds_with_utr3.fasta"

# Read sequences from both FASTA files into dictionaries
cds_dict = SeqIO.to_dict(SeqIO.parse(cds_fasta, "fasta"))
utr3_dict = SeqIO.to_dict(SeqIO.parse(utr3_fasta, "fasta"))

# Open the output file for writing
with open(output_fasta, "w") as output_handle:

    pattern = re.compile(r'^[NACTG]+$')
    for seq_id, cds_record in cds_dict.items():
        print(seq_id)
        cds_seq = cds_record.seq

        if pattern.match(str(cds_seq)):
            if seq_id in utr3_dict:
                utr3_seq = utr3_dict[seq_id].seq

                if pattern.match(str(utr3_seq)):
                    # Concatenate the CDS and 3' UTR sequences
                    combined_seq = cds_seq + utr3_seq
                    description = "CDS + 3'UTR"
                else:
                    combined_seq = cds_seq
                    description = "CDS only (no 3' UTR transcript available)"
                    
            else:
                combined_seq = cds_seq
                description = "CDS only (no full transcript available)"
        else:
            combined_seq = ""
            description = "No seq available"
        # Create a new SeqRecord with the combined sequence
        new_record = SeqRecord(
            combined_seq,
            id=seq_id,
            description=description
        )

        # Write the new record to the output file
        SeqIO.write(new_record, output_handle, "fasta")

        

