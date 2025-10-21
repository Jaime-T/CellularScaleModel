""" 
Compute and plot the distribution of CSM scores for TP53 mutations from ClinVar dataset.
Compare with base ESM2 model scores.

For missense run10 - just DepMap missense data (unique protein mutations)
"""


import pandas as pd
from peft import PeftModel, PeftConfig
from transformers import EsmForMaskedLM, EsmTokenizer
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ( ggplot, aes, geom_density, theme_minimal, scale_color_manual, 
    scale_fill_manual, xlim, ylim)
import re
import os
from sklearn.preprocessing import StandardScaler

def compute_mut_score(model, tokenizer, mutation, sequence):
    model.eval()

    wt, pos, mt = mutation[0], int(mutation[1:-1]), mutation[-1]
    # Adjust for 0-based indexing
    pos -= 1
    if sequence[pos] != wt:
        raise ValueError(f"Wildtype mismatch at position {pos+1} for mutation {mutation}: expected {sequence[pos]}, found {wt}")
    
    # Tokenize the sequence
    encoded = tokenizer(sequence, return_tensors="pt")
    input_ids = encoded["input_ids"]

    # Mask the site 
    masked_input_ids = input_ids.clone()
    masked_input_ids[0, pos + 1] = tokenizer.mask_token_id  # +1 due to BOS token

    with torch.no_grad():
        logits = model(masked_input_ids).logits
        log_probs = torch.nn.functional.log_softmax(logits[0, pos + 1], dim=0)
    
    # Get the log probability of the mutated amino acid at the mutation position
    aa_index = tokenizer.convert_tokens_to_ids(mt)
    score = log_probs[aa_index].item()
    return score

def mut_distro_plot(csm_data, save_dir, xlabel="csm_score", num=1):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # method 1: seaborn kdeplot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=csm_data,
        x=xlabel,
        hue="clinvar_label",
        fill=True,
        alpha=0.3
    )

    plt.title(f"Density of {xlabel} by ClinVar Label")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    sns.despine()
    save_path = os.path.join(save_dir, f"tp53_{xlabel}_plot{num}-1.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved to {save_path}")

    # method 2: plotnine (ggplot2-like)
    p = (
        ggplot(csm_data, aes(x=xlabel, color="clinvar_label", fill="clinvar_label"))
        + geom_density(alpha=0.3)
        + theme_minimal()
    )
    save_path = os.path.join(save_dir, f"tp53_{xlabel}_plot{num}-2.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")

def extremity_mut_distro_plot(csm_data, save_dir, xlabel="csm_score", num=1):

    # Standardise the data
    scaler = StandardScaler()
    csm_data = csm_data.copy()  # avoid modifying original
    csm_data[xlabel] = scaler.fit_transform(csm_data[[xlabel]])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Custom color mapping
    color_map = {
        "Pathogenic": "red",
        "Benign": "green",
        "Uncertain significance": "yellow"
    }

    # method 1: seaborn kdeplot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=csm_data,
        x=xlabel,
        hue="clinvar_label",
        fill=True,
        alpha=0.3,
        palette=color_map
    )

    plt.title(f"Density of {xlabel} by ClinVar Label")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    sns.despine()

    save_path = os.path.join(save_dir, f"tp53_{xlabel}_plot{num}-1.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved to {save_path}")

    # method 2: plotnine (ggplot2-like)
    p = (
        ggplot(csm_data, aes(x=xlabel, color="clinvar_label", fill="clinvar_label"))
        + geom_density(alpha=0.3)
        + theme_minimal()
        + scale_color_manual(values=color_map)
        + scale_fill_manual(values=color_map)
        + xlim(-30, 5)
        + ylim(0, 1.5)
        + theme_minimal()
    )
    save_path = os.path.join(save_dir, f"tp53_{xlabel}_plot{num}-2.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")



def main():
    
    # Load ESM2 base model and tokenizer
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    tokenizer = EsmTokenizer.from_pretrained(base_model_path)
    base_model = EsmForMaskedLM.from_pretrained(base_model_path)
    base_model.eval()

    # Load dataset with ClinVar labels and mutations
    tp53_clinvar = pd.read_csv("/g/data/gi52/jaime/clinvar/clinvar_tp53_mutations_1letter.csv")

    # tp53 protein sequence
    tp53 = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
        
    # Precompute ESM baseline scores once
    print("Precomputing baseline ESM2 scores...")
    for row in tp53_clinvar.itertuples():
        mutation = row.ProteinChange
        idx = row.Index
        if pd.isna(mutation):
            continue
        try:
            esm_score = compute_mut_score(base_model, tokenizer, mutation, tp53)
            tp53_clinvar.loc[idx, "esm_score"] = esm_score
        except ValueError as e:
            print(e)
            tp53_clinvar.loc[idx, "esm_score"] = None

    # Save intermediate file to reuse later
    esm_cache_path = "/g/data/gi52/jaime/clinvar/tp53_esm2_baseline_scores.csv"
    tp53_clinvar.to_csv(esm_cache_path, index=False)


    for batch_num in range(1000, 5001, 1000):

        print(f"Processing batch number: {batch_num}")
        base_model = EsmForMaskedLM.from_pretrained(base_model_path)
        # Load adapters for CSM finetuned model 
        csm_adapter_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/missense/run10/epoch0_batch{batch_num}"

        # Load the adapter into the model
        model = PeftModel.from_pretrained(base_model, csm_adapter_path, is_trainable=False )

        # Merge the adapter weights into the base model
        csm_model = model.merge_and_unload()
        csm_model.eval()

        # save path 
        save_dir = f"/g/data/gi52/jaime/clinvar/run10_ms/epoch0_batch{batch_num}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        df = tp53_clinvar.copy()

        # Calculate the CSM scores for each mutation and append to dataframe
        for row in df.itertuples():
            mutation = row.ProteinChange
            idx = row.Index
            if pd.isna(mutation):
                continue
            try:
                csm_score = compute_mut_score(csm_model, tokenizer, mutation, tp53)
                df.loc[idx, 'csm_score'] = csm_score
            except ValueError as e:
                print(e)
                df.loc[idx, 'csm_score'] = None


        # Save the final results
        save_path = os.path.join(save_dir, "tp53_clinvar_csm_scores.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved CSM results to {save_path}")

        # rename columns for plotting
        csm_data = df.rename(
            columns={"Germline.classification": "clinvar_label"}
        )

        # Graph the distribution of scores for each ClinVar category
        mut_distro_plot(csm_data, save_dir, xlabel="csm_score", num=1)
        mut_distro_plot(csm_data, save_dir, xlabel="esm_score", num=2)

        # filter out only Pathogenic, Benign, and Uncertain Significance
        filtered_data = csm_data[csm_data['clinvar_label'].isin(['Pathogenic', 'Benign', 'Uncertain significance'])]
        extremity_mut_distro_plot(filtered_data, save_dir, xlabel="csm_score", num=3)
        extremity_mut_distro_plot(filtered_data, save_dir, xlabel="esm_score", num=4)

        # delete models to free up memory
        del csm_model, model, base_model
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()   