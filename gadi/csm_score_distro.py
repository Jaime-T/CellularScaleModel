import pandas as pd
from peft import PeftModel, PeftConfig
from transformers import EsmForMaskedLM, EsmTokenizer
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_density, theme_minimal
import re

def compute_mut_score(model, tokenizer, mutation, sequence):
    wt, pos, mt = mutation[0], int(mutation[1:-1]), mutation[-1]
    # Adjust for 0-based indexing
    pos -= 1
    if sequence[pos] != wt:
        raise ValueError(f"Wildtype mismatch at position {pos+1} for mutation {mutation}: expected {sequence[pos]}, found {wt}")
    
    mutated_sequence = sequence[:pos] + mt + sequence[pos+1:]
    
    inputs = tokenizer(mutated_sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Get the log probability of the mutated amino acid at the mutation position
    aa_index = tokenizer.convert_tokens_to_ids(mt)
    score = log_probs[0, pos+1, aa_index].item()  # +1 for special token offset
    print(f"Computed score for mutation {mutation}: {score}")
    return score

def mut_distro_plot(csm_data, xlabel="csm_score", num=1):

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
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/missense/run9.1/distro_curves/tp53_{xlabel}_plot{num}-1.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved to {save_path}")

    # method 2: plotnine (ggplot2-like)
    p = (
        ggplot(csm_data, aes(x=xlabel, color="clinvar_label", fill="clinvar_label"))
        + geom_density(alpha=0.3)
        + theme_minimal()
    )
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/missense/run9.1/distro_curves/tp53_{xlabel}_plot{num}-2.png"
    p.save(save_path, width=8, height=5, dpi=300)
    print(f"Saved to {save_path}")


def main():

    # Load ESM2 base model and tokenizer
    base_model_path = "/g/data/gi52/jaime/esm2_650M_model"
    tokenizer = EsmTokenizer.from_pretrained(base_model_path)
    base_model = EsmForMaskedLM.from_pretrained(base_model_path)

    # Load adapters for CSM finetuned model 
    csm_adapter_path = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run9.1/epoch0_batch11000"

    # Load the PEFT adapter configuration
    peft_config = PeftConfig.from_pretrained(csm_adapter_path)

    # Load the adapter into the model
    model = PeftModel.from_pretrained(base_model, csm_adapter_path)

    # Merge the adapter weights into the base model
    csm_model = model.merge_and_unload()
    csm_model.eval()

    # Load dataset with ClinVar labels and mutations
    tp53_clinvar = pd.read_csv("/g/data/gi52/jaime/clinvar/clinvar_tp53_mutations_1letter.csv")

    # tp53 protein sequence
    tp53 = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    
    # Calculate the CSM scores for each mutation and append to dataframe
    for row in tp53_clinvar.itertuples():
        mutation = row.ProteinChange
        if pd.isna(mutation) or len(mutation) < 3:
            continue
        try:
            csm_score = compute_mut_score(csm_model, tokenizer, mutation, tp53)
            tp53_clinvar.loc[row.Index, 'csm_score'] = csm_score

            # TO DO: add esm score and compare 
            esm_score = compute_mut_score(base_model, tokenizer, mutation, tp53)
            tp53_clinvar.loc[row.Index, 'esm_score'] = esm_score

        except ValueError as e:
            print(e)
            tp53_clinvar.loc[row.Index, 'csm_score'] = None  
            tp53_clinvar.loc[row.Index, 'esm_score'] = None

    tp53_clinvar.to_csv("/g/data/gi52/jaime/clinvar/tp53_clinvar_csm_esm_scores.csv", index=False)

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