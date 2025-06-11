from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

# Define input and output file paths
cds_fasta = "/Users/jaimetaitz/Downloads/mart_export_coding_seqs.txt"
full_fasta = "/Users/jaimetaitz/Downloads/mart_export_utr.txt"
output_fasta = "cds_with_3utr.fasta"

# Read sequences from both FASTA files into dictionaries
cds_dict = SeqIO.to_dict(SeqIO.parse(cds_fasta, "fasta"))
full_dict = SeqIO.to_dict(SeqIO.parse(full_fasta, "fasta"))

# Open the output file for writing
with open(output_fasta, "w") as output_handle:
    for seq_id, cds_record in cds_dict.items():
        cds_seq = cds_record.seq
        

        if seq_id in full_dict:
            full_seq = full_dict[seq_id].seq

            # Find the position of the CDS within the full transcript
            cds_start = full_seq.find(cds_seq)
            if cds_start == -1:
                #print(f"Warning: CDS for {seq_id} not found in full transcript. Writing CDS only.")
                combined_seq = cds_seq
                description = "" #"CDS only (CDS not found in full transcript)"
            else:
                # Calculate the end position of the CDS
                cds_end = cds_start + len(cds_seq)

                # Extract the 3' UTR sequence
                utr3_seq = full_seq[cds_end:]

                # Concatenate the CDS and 3' UTR sequences
                combined_seq = cds_seq + utr3_seq
                description = "" #"CDS + 3'UTR"
                
                
        else:
            #print(f"Warning: {seq_id} not found in full transcript file. Writing CDS only.")
            combined_seq = cds_seq
            description = "" #"CDS only (no full transcript available)"

        # Create a new SeqRecord with the combined sequence
        new_record = SeqRecord(
            combined_seq,
            id=seq_id,
            description=description
        )

        # Write the new record to the output file
        SeqIO.write(new_record, output_handle, "fasta")
        

