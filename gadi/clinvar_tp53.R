library(tidyverse)
getwd()
setwd("/Users/jtaitz/Downloads")
data <- read.table("clinvar_result_sort_location.txt", header = TRUE, sep = "\t")
# filter out rows with no protein change 
protein_change_data <- data[data$Protein.change != "", ]
protein_change_data <- protein_change_data[protein_change_data$Protein.change != "NA", ]
View(protein_change_data)

# select important columns 
filter_data <- protein_change_data %>%
  select(Name, Gene.s., Variant.type, Germline.classification)

View(filter_data)

{python}
aa_dict = {
    "GLY": "G",
    "ALA": "A",
    "VAL": "V",
    "LEU": "L",
    "ILE": "I",
    "THR": "T",
    "SER": "S",
    "MET": "M",
    "CYS": "C",
    "PRO": "P",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
    "HIS": "H",
    "LYS": "K",
    "ARG": "R",
    "ASP": "D",
    "GLU": "E",
    "ASN": "N",
    "GLN": "Q"
}

# make new columns with wildtype amino acid, mut amino acid, and position of mutation
library(dplyr)
library(stringr)
clinvar_mutation_data <- filter_data %>%
  mutate(
    # Extract the part inside parentheses that starts with "p."
    ProteinChange = str_extract(Name, "\\(p\\.[A-Za-z]+[0-9]+[A-Za-z]+\\)"),
    
    # Remove the '(p.' prefix and ')'
    ProteinChange = str_remove_all(ProteinChange, "\\(p\\.|\\)"),
  )
  

View(clinvar_mutation_data)
write.csv(clinvar_mutation_data, "clinvar_tp53_mutations.csv", row.names = FALSE)

print(nrow(clinvar_mutation_data))