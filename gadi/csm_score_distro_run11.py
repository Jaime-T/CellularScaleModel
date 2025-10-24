""" 
Compute and plot the distribution of CSM scores for TP53 mutations from ClinVar dataset.
Compare with base ESM2 model scores.

For missense run11 - 7:1 MT to WT batches of 8
"""


import pandas as pd
from peft import PeftModel
from transformers import EsmForMaskedLM, EsmTokenizer
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ( ggplot, aes, geom_density, theme_minimal, scale_color_manual, 
    scale_fill_manual)
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

def extremity_mut_distro_plot(csm_data, save_dir, batch_num, xlabel="csm_score"):

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
        "Uncertain significance": "yellow",
        "Conflicting classifications of pathogenicity": "grey"
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

    save_path = os.path.join(save_dir, f"epoch0_batch{batch_num}_tp53_{xlabel}_plot-1.png")
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
        + theme_minimal()
    )
    save_path = os.path.join(save_dir, f"epoch0_batch{batch_num}_tp53_{xlabel}_plot-2.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")



def main():
    
    # Load ESM2 base model and tokenizer
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    tokenizer = EsmTokenizer.from_pretrained(base_model_path)
    base_model = EsmForMaskedLM.from_pretrained(base_model_path)
    base_model.eval()

    # Load dataset with ClinVar labels and mutations, and precomputed esm scores 
    tp53_clinvar = pd.read_csv("/g/data/gi52/jaime/clinvar/tp53_esm2_baseline_scores.csv")

    # tp53 protein sequence
    tp53 = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    
    # save directory
    save_dir = f"/g/data/gi52/jaime/clinvar/run11_ms"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # rename columns for plotting
    df = tp53_clinvar.copy()
    df = df.rename(
        columns={"Germline.classification": "clinvar_label"}
    )

    # filter out only Pathogenic, Benign, Uncertain Significance, and Conflicting classifications of pathogenicity
    filtered_data = df[df['clinvar_label'].isin(['Pathogenic', 'Benign', 'Uncertain significance','Conflicting classifications of pathogenicity'])]
        
    extremity_mut_distro_plot(filtered_data, save_dir, 0, xlabel="esm_score")

    for batch_num in range(1000, 10001, 1000):

        print(f"Processing batch number: {batch_num}")
        base_model = EsmForMaskedLM.from_pretrained(base_model_path)
        # Load adapters for CSM finetuned model 
        csm_adapter_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/missense/run11/epoch0_batch{batch_num}"

        # Load the adapter into the model
        model = PeftModel.from_pretrained(base_model, csm_adapter_path, is_trainable=False )

        # Merge the adapter weights into the base model
        csm_model = model.merge_and_unload()
        csm_model.eval()     

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
        save_path = os.path.join(save_dir, f"epoch0_batch{batch_num}_tp53_clinvar_csm_scores.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved CSM results to {save_path}")

        # rename columns for plotting
        csm_data = df.rename(
            columns={"Germline.classification": "clinvar_label"}
        )

        # filter out only Pathogenic, Benign, and Uncertain Significance
        filtered_data = csm_data[csm_data['clinvar_label'].isin(['Pathogenic', 'Benign', 'Uncertain significance','Conflicting classifications of pathogenicity'])]
        extremity_mut_distro_plot(filtered_data, save_dir, batch_num, xlabel="csm_score")
        

        # delete models to free up memory
        del csm_model, model, base_model
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()   