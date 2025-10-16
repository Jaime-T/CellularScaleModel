""" 
Compute and plot the distribution of CSM scores for TP53 mutations from ClinVar dataset.
Compare with base ESM2 model scores.

For missense run8 - just DepMap missense data (unique protein mutations)
"""


import pandas as pd
from peft import PeftModel, PeftConfig
from transformers import EsmForMaskedLM, EsmTokenizer
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_density, theme_minimal
import re
import os

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

def mut_distro_plot(csm_data, xlabel="csm_score", num=1):

    save_dir = "/g/data/gi52/jaime/clinvar/run8_ms/distro_curves"

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
    save_path = os.join(save_dir, f"tp53_{xlabel}_plot{num}-1.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved to {save_path}")

    # method 2: plotnine (ggplot2-like)
    p = (
        ggplot(csm_data, aes(x=xlabel, color="clinvar_label", fill="clinvar_label"))
        + geom_density(alpha=0.3)
        + theme_minimal()
    )
    save_path = os.join(save_dir, f"tp53_{xlabel}_plot{num}-2.png")
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")


def main():

    # Load ESM2 base model and tokenizer
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    tokenizer = EsmTokenizer.from_pretrained(base_model_path)
    base_model = EsmForMaskedLM.from_pretrained(base_model_path)

    # Load adapters for CSM finetuned model 
    csm_adapter_path = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run8/epoch0_batch40000"

    # Load the adapter into the model
    model = PeftModel.from_pretrained(base_model, csm_adapter_path, is_trainable=False )

    # Merge the adapter weights into the base model
    csm_model = model.merge_and_unload()
    csm_model.eval()

    print(csm_model)


     # debugging print to verify adapter contents

     # Point directly to the adapter weights file
    import os
    from safetensors.torch import load_file
    adapter_weights = os.path.join(
        "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run8/epoch0_batch40000",
        "adapter_model.safetensors"
    )

    state_dict = load_file(adapter_weights)
    print("Number of keys:", len(state_dict.keys()))
    print("Example keys:", list(state_dict.keys())[:5])

    for name, module in base_model.named_modules():
        if "Linear" in str(type(module)):
            print(name)

    from copy import deepcopy

    # Load base model separately for comparison
    base_model_fresh = EsmForMaskedLM.from_pretrained(base_model_path)

    # Compare parameters between merged model and base
    diff_count = 0
    for (name1, p1), (name2, p2) in zip(base_model_fresh.named_parameters(), csm_model.named_parameters()):
        if not torch.allclose(p1, p2):
            diff_count += 1
            break

    print("Any parameter differences?", diff_count > 0)


    # Load dataset with ClinVar labels and mutations
    tp53_clinvar = pd.read_csv("/g/data/gi52/jaime/clinvar/clinvar_tp53_mutations_1letter.csv")

    # tp53 protein sequence
    tp53 = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    
    # save path 
    save_dir = "/g/data/gi52/jaime/clinvar/run8_ms"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate the CSM scores for each mutation and append to dataframe
    for row in tp53_clinvar.itertuples():
        
        mutation = row.ProteinChange
        idx = row.Index

        if pd.isna(mutation):
            continue
        try:
            csm_score = compute_mut_score(csm_model, tokenizer, mutation, tp53)
            esm_score = compute_mut_score(base_model_fresh, tokenizer, mutation, tp53)
            print(f"Mutation: {mutation}, CSM Score: {csm_score}, ESM Score: {esm_score}")

            tp53_clinvar.loc[idx, 'csm_score'] = csm_score
            tp53_clinvar.loc[idx, 'esm_score'] = esm_score


        except ValueError as e:
            print(e)
            tp53_clinvar.loc[idx, 'csm_score'] = None  
            tp53_clinvar.loc[idx, 'esm_score'] = None

        # save intermediate results
        if idx % 500 == 0:
            print(f"Processed {idx} mutations, saving intermediate results...")
            save_path = os.path.join(save_dir, f"tp53_clinvar_csm_esm_scores_intermediate{idx}.csv")
            tp53_clinvar.to_csv(save_path, index=False)

    # Save the final results
    save_path = os.path.join(save_dir, "tp53_clinvar_csm_esm_scores_final.csv")
    tp53_clinvar.to_csv(save_path, index=False)

    # rename columns for plotting
    csm_data = tp53_clinvar.rename(
        columns={"Germline.classification": "clinvar_label"}
    )

    # Graph the distribution of scores for each ClinVar category
    mut_distro_plot(csm_data, xlabel="csm_score", num=1)
    mut_distro_plot(csm_data, xlabel="esm_score", num=2)

    # filter out only Pathogenic, Benign, and Uncertain Significance
    filtered_data = csm_data[csm_data['clinvar_label'].isin(['Pathogenic', 'Benign', 'Uncertain significance'])]
    mut_distro_plot(filtered_data, xlabel="csm_score", num=3)
    mut_distro_plot(filtered_data, xlabel="esm_score", num=4)



if __name__ == "__main__":
    main()   