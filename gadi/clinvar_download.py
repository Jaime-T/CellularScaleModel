import pandas as pd
import os

url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

df = pd.read_csv(url, sep='\t', compression='gzip')
print(os.getcwd())
save_path = "/home/cciamr.local/jtaitz/R_Drive/DDC/jaime/CellularScaleModel/gadi/clinvar_variant_summary.csv"
df.to_csv(save_path, index=False)


print(df.columns)

# Set the option to display all columns
pd.set_option('display.max_columns', None)

print(df.head(5))