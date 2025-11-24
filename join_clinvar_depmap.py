import pandas as pd
import re

# ==== 1. Load data ====
clinvar_path = "/home/cciamr.local/jtaitz/R_Drive/DDC/jaime/CellularScaleModel/data/clinvar_variant_summary.csv"
depmap_path  = "/home/cciamr.local/jtaitz/R_Drive/DDC/jaime/CellularScaleModel/data/update4_windowed_mutant_proteins.csv"
output_path  = "/home/cciamr.local/jtaitz/R_Drive/DDC/jaime/CellularScaleModel/data/clinvar_depmap_joined_update.csv"

print("=== Step 1: Reading files ===")
clinvar = pd.read_csv(clinvar_path)
depmap = pd.read_csv(depmap_path)
print(f"ClinVar rows: {len(clinvar)}")
print(f"DepMap rows: {len(depmap)}\n")

# ==== 2. Helper function: 3-letter → 1-letter amino acid codes ====
aa_map = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C",
    "Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
    "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P",
    "Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V",
    "Ter":"*"
}

def convert_protein_notation(s):
    if pd.isna(s):
        return s
    out = s
    for k, v in aa_map.items():
        out = re.sub(k, v, out)
    return out

# ==== 3. Clean ClinVar ====
print("=== Step 3: Cleaning ClinVar ===")
clinvar_clean = clinvar.copy()
clinvar_clean["HGNC_ID"] = clinvar_clean["HGNC_ID"].astype(str).str.strip()
clinvar_clean["transcript"] = clinvar_clean["Name"].str.extract(r"(NM_[0-9.]+)")
clinvar_clean["DNAChange"] = clinvar_clean["Name"].str.extract(r"(c\.[^ )]+)")
clinvar_clean["ProteinChange"] = clinvar_clean["Name"].str.extract(r"(p\.[^ )]+)")
clinvar_clean["ProteinChange"] = clinvar_clean["ProteinChange"].apply(convert_protein_notation)

print("Example ClinVar entries:")
print(clinvar_clean[["HGNC_ID","transcript","DNAChange","ProteinChange"]].head(), "\n")

# ==== 4. Clean DepMap ====
print("=== Step 4: Cleaning DepMap ===")
depmap_clean = depmap.copy()
depmap_clean["HGNC_ID"] = depmap_clean["VepHgncID"].astype(str).str.strip()
depmap_clean["transcript"] = depmap_clean["VepManeSelect"].astype(str).str.strip()
depmap_clean["DNAChange"] = depmap_clean["DNAChange"].str.extract(r"(c\.[^,]+)")
depmap_clean["ProteinChange"] = depmap_clean["ProteinChange"].str.extract(r"(p\.[^,]+)")

print("Example DepMap entries:")
print(depmap_clean[["HGNC_ID","transcript","DNAChange","ProteinChange"]].head(), "\n")

# ==== 5. Join by HGNC_ID ====
print("=== Step 5: Joining by HGNC_ID ===")
join1 = pd.merge(
    clinvar_clean, depmap_clean,
    on=["HGNC_ID"], how="inner", suffixes=(".clinvar", ".depmap")
)
print(f"HGNCID-level matches: {len(join1)}")
if not join1.empty:
    print(join1[["HGNC_ID","ProteinChange","DNAChange.clinvar","DNAChange.depmap"]].head(), "\n")



join1.to_csv(output_path, index=False)
print(f"Output saved to: {output_path}")
# Print how many entries there are
print("there are total of ", len(join1), " entries in the joined clinvar and depmap file.")
print("Done!")
