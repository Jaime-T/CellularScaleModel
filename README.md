# CellularScaleModel
Honours project 

Instructions for environment creation/package mngement:
    mamba:
        ~ mamba env create -f environment.yaml
        ~ mamba activate csm

    pip:
        ~ source csm_venv/bin/activate
        ~ pip install -r requirements.txt

To run the program, run:
    ~ python3 file_name.py


Preprocessing data:

1) add_utr3.py - this file adds the utr3 sequence to the coding sequence for each gene 
2) mutate_data.py - this file applies the mutation the coding sequence, 
    and translates it into an amino acid protein sequence. It creates a dataset of unique 
    protein sequences in csv format
3) process_data.py - selects and categorises sequences into missense or framehsift mutations.
    It applies a sliding window if sequences exceed 1022 sequence length

Training Iterations:

1) train_iteration1_run4.py - Iteration 1 (random token masking)
2) train_iteration2_run8.py - Iteration 2 (targeted masking at mutation sites)
3) train_iteration3_run9-1.py - Iteration 3 (balanced batch composition with 1:1 mutant-to-wildtype sequences; random masking for wildtype seqs, specific mutation position masking for mutant type seq)
4) train_iteration4_run10.py - Iteration 4 (training data maintains original mutation frequency)
5) train_iteration5_run11.py - Iteration 5 (weighted batch composition with 7:1 mutant-to-wildtype sequences; all seqs use specific mutation position masking)

summarised as below:

<img width="651" height="405" alt="Screenshot 2025-11-25 at 1 09 38 pm" src="https://github.com/user-attachments/assets/534b5c96-d4e9-4d7e-b114-4e1a2db77d9b" />

Visualisations:

1) heatmap_scaled_test.py - creates scaled heatmaps for ESM2 pretrained model and CSM finetuned model. Also, creates scaled heatmaps with special colour scheme for difference heatmap (CSM - ESM2)
2) delta_csm_esm_score_distro.py - creates Cartesian scatterplot of CSM vs ESM2 scores, and creates density distribution using ClinVar pathogenicity labels
3) heatmap.py - creates heatmaps with training data as dots 

