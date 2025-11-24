import requests
import csv
import time
import sys
import pandas as pd

HGNC_BASE = "https://rest.genenames.org"
UNIPROT_FASTA_BASE = "https://rest.uniprot.org/uniprotkb"

def fetch_hgnc_record(symbol_or_id):
    """
    Query HGNC REST API with a symbol or HGNC ID.
    Returns dict with keys including 'hgnc_id', 'symbol', 'uniprot_ids' (may be list or string) or None if not found.
    """
    headers = {"Accept": "application/json"}
    if symbol_or_id.upper().startswith("HGNC:"):
        field = "hgnc_id"
        value = symbol_or_id.upper()
    else:
        field = "symbol"
        value = symbol_or_id
    url = f"{HGNC_BASE}/fetch/{field}/{value}"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Warning: HGNC API returned status {resp.status_code} for {symbol_or_id}", file=sys.stderr)
        return None
    data = resp.json()
    docs = data.get("response", {}).get("docs", [])
    if not docs:
        print(f"No HGNC entry found for {symbol_or_id}", file=sys.stderr)
        return None
    rec = docs[0]
    return rec

def fetch_uniprot_sequence(uniprot_id):
    """
    Fetch the protein sequence (FASTA) for the given UniProt ID.
    Returns sequence string (amino acids) or None.
    """
    url = f"{UNIPROT_FASTA_BASE}/{uniprot_id}.fasta"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Warning: UniProt API returned status {resp.status_code} for {uniprot_id}", file=sys.stderr)
        return None
    fasta = resp.text
    lines = fasta.splitlines()
    seq = "".join(ln.strip() for ln in lines if not ln.startswith(">"))
    return seq

def build_gene_table(gene_list, out_csv, delay=0.5):
    """
    For each input gene symbol or HGNC ID:
      - Map to HGNC record (get hgnc_id, symbol, uniprot_ids)
      - Fetch sequence (first UniProt ID)
    Returns list of dicts:
      [{ "hgnc_id": ..., "symbol": ..., "sequence": ... }, ...]
    """
    # Write header first
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hgnc_id","symbol","sequence"])

    # Process genes one at a time
    for gene in gene_list:
        rec = fetch_hgnc_record(gene)
        if rec is None:
            hgnc_id = ""
            symbol = gene
            seq = ""
        else:
            hgnc_id = rec.get("hgnc_id", "")
            symbol = rec.get("symbol", "")
            uniprot_ids = rec.get("uniprot_ids")
            seq = ""
            if uniprot_ids:
                # uniprot_ids may be list or string or comma-separated
                if isinstance(uniprot_ids, list):
                    uid = uniprot_ids[0]
                else:
                    # string
                    uid = uniprot_ids.split(",")[0].strip()
                seq = fetch_uniprot_sequence(uid) or ""
                # if fail, leave seq as ""
            else:
                print(f"No UniProt ID found for {hgnc_id} / {symbol}", file=sys.stderr)

        # Append to CSV
        with open(out_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([hgnc_id, symbol, seq])
        time.sleep(delay)
    

if __name__ == "__main__":

    # Read from clinvar file
    clinvar = pd.read_csv("../data/clinvar_variant_summary.csv", usecols=["GeneSymbol"])
    gene_symbols = clinvar["GeneSymbol"].dropna().unique().tolist()
    print(f"Found {len(gene_symbols)} unique gene symbols in ClinVar data.")
    input_genes = gene_symbols  # limit for testing; remove or increase as needed
    build_gene_table(input_genes, "gene_sequences_with_ids.csv", delay=0.5)

    print("Done. CSV written to gene_sequences_with_ids.csv")
