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


Order of Files:
1) add_utr3.py - this file adds the utr3 sequence to the coding sequence for each gene 

2) mutate_data.py - this file applies the mutation the coding sequence, 
    and translates it into an amino acid protein sequence. It creates a dataset of unique 
    protein sequences in csv format

3) process_data.py - selects and categorises sequences into missense or framehsift mutations.
    It applies a sliding window if sequences exceed 1022 sequence length 

4) train.py - tokenises the input and masks the amino acid where the mutation occurs first, then 
trains model
