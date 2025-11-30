# About this project
Cellular Scale Model (CSM) is a prototype protein large language–model (PLLM) pipeline
for enhancing variant interpretation in cancer.

It builds on the pretrained ESM-2 model and fine-tunes it on human, cancer-relevant
variants so that researchers can:</br>
- compare evolutionary vs cellular views of a mutation (ESM-2 vs CSM),
- visualise full mutational landscapes with heatmaps, and
- explore how CSM behaves on ClinVar-labelled variants (benign, pathogenic, VUS).


This repository accompanies the Honours thesis:</br>
“Enhancing Variant Interpretation in Cancer Using Protein Large Language Models” – Jaime Taitz (UNSW, 2025)

### Key Features

- ESM-2 + LoRA: Parameter-efficient fine-tuning of the 650M-parameter ESM-2 model.

- Cancer-aware scoring: Compute log-likelihood ratios (LLRs) and Δ-scores (CSM − ESM-2) for
  mutations of interest.
  

- ClinVar integration: Simple utilities for aligning model scores with ClinVar clinical
  significance labels.
  

- Visual diagnostics  
  - Global variant-effect heatmaps  
  - Delta-score kernel density plots  
  - CSM vs ESM-2 scatterplots

### Key Files

## Preprocessing data

1) add_utr3.py - this file adds the utr3 sequence to the coding sequence for each gene 
2) mutate_data.py - this file applies the mutation the coding sequence, 
    and translates it into an amino acid protein sequence. It creates a dataset of unique 
    protein sequences in csv format
3) process_data.py - selects and categorises sequences into missense or framehsift mutations.
    It applies a sliding window if sequences exceed 1022 sequence length

## Training Iterations (in **gadi** folder)

1) train_iteration1_run4.py - Iteration 1 (random token masking)
2) train_iteration2_run8.py - Iteration 2 (targeted masking at mutation sites)
3) train_iteration3_run9-1.py - Iteration 3 (balanced batch composition with 1:1 mutant-to-wildtype sequences; random masking for wildtype seqs, specific mutation position masking for mutant type seq)
4) train_iteration4_run10.py - Iteration 4 (training data maintains original mutation frequency)
5) train_iteration5_run11.py - Iteration 5 (weighted batch composition with 7:1 mutant-to-wildtype sequences; all seqs use specific mutation position masking)

summarised as below:

<p align="middle">
    <img src="https://github.com/user-attachments/assets/534b5c96-d4e9-4d7e-b114-4e1a2db77d9b" width="651" height="405" alt="Training iteration feature table" />
</p>

## Visualisations

1) heatmap_scaled_test.py - creates scaled heatmaps for ESM2 pretrained model and CSM finetuned model. Also, creates scaled heatmaps with special colour scheme for difference heatmap (CSM - ESM2)
2) delta_csm_esm_score_distro.py - creates Cartesian scatterplot of CSM vs ESM2 scores, and creates density distribution using ClinVar pathogenicity labels
3) heatmap.py - creates heatmaps with training data as dots 


### Quick Start

1. **Clone the repo**

```bash
git clone https://github.com/Jaime-T/CellularScaleModel.git
cd <your-repo>
```

2. **Environment setup**

This project was developed using a Conda/Mamba environment defined in `environment.yml`.  
A `requirements.txt` file is also provided for users who prefer `pip` and virtual environments.

### Option 1 – Conda/Mamba (recommended)

Create and activate the `csm` environment:

```bash
# Create environment
mamba env create -f environment.yml

# or, if you use conda:
# conda env create -f environment.yml

# Activate environment
mamba activate csm
# or: conda activate csm
```

### Option 2 - pip + virtualenv
```bash
# Create virtual environment (only once)
python -m venv csm_venv

# Activate it (Linux/macOS)
source csm_venv/bin/activate

# On Windows (PowerShell):
# .\csm_venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt
