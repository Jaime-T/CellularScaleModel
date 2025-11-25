# CellularScaleModel
Honours project 

**Problem:**
ESM-2 and similar models are trained on natural (wild-type) sequences, but not curated human disease-associated variants. This limits their ability to distinguish clinically relevant human variants from those that are evolutionarily rare but benign, and to distinguish between variants that are evolutionarily deleterious and those deleterious to cellular function. 

**Solution:**
This project proposes fine-tuning ESM-2 using a comprehensive dataset of cancer-related human protein sequences from DepMap database (DepMap Public 25Q3 release). This will allow the model to learn patterns from known human variations, improving its ability to differentiate between benign and pathogenic variants and potentially discover new cell vulnerabilities. The new model, termed Cell Scale Modelling (CSM), will be designed to identify clinically relevant mutations with greater precision. By integrating predictions from ESM-2 and CSM, we aim to establish a dual-model approach that significantly enhances the sensitivity and specificity of variant classification.



Instructions for environment creation/package mngement:
    mamba:
        ~ mamba env create -f environment.yaml
        ~ mamba activate csm

    pip:
        ~ source csm_venv/bin/activate
        ~ pip install -r requirements.txt

To run the program, run:
    ~ python3 file_name.py


Preprocessing data (in **training_data_processing** directory):

1) add_utr3.py - this file adds the utr3 sequence to the coding sequence for each gene 
2) mutate_data.py - this file applies the mutation the coding sequence, 
    and translates it into an amino acid protein sequence. It creates a dataset of unique 
    protein sequences in csv format
3) process_data.py - selects and categorises sequences into missense or framehsift mutations.
    It applies a sliding window if sequences exceed 1022 sequence length

Training Iterations (in **gadi** directory):

1) train_iteration1_run4.py - Iteration 1 (random token masking)
2) train_iteration2_run8.py - Iteration 2 (targeted masking at mutation sites)
3) train_iteration3_run9-1.py - Iteration 3 (balanced batch composition with 1:1 mutant-to-wildtype sequences; random masking for wildtype seqs, specific mutation position masking for mutant type seq)
4) train_iteration4_run10.py - Iteration 4 (training data maintains original mutation frequency)
5) train_iteration5_run11.py - Iteration 5 (weighted batch composition with 7:1 mutant-to-wildtype sequences; all seqs use specific mutation position masking)

summarised as below:

<img width="651" height="405" alt="Screenshot 2025-11-25 at 1 09 38 pm" src="https://github.com/user-attachments/assets/534b5c96-d4e9-4d7e-b114-4e1a2db77d9b" />


Visualisations (in **visualisations**  directory):

1) heatmap_scaled_test.py - creates scaled heatmaps for ESM2 pretrained model and CSM finetuned model. Also, creates scaled heatmaps with special colour scheme for difference heatmap (CSM - ESM2)
2) delta_csm_esm_score_distro.py - creates Cartesian scatterplot of CSM vs ESM2 scores, and creates density distribution using ClinVar pathogenicity labels
3) heatmap.py - creates heatmaps with training data as dots 

