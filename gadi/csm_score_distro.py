import pandas as pd
from peft import PeftModel, PeftConfig
from transformers import EsmForMaskedLM, EsmTokenizer
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_density, theme_minimal
import re

def compute_csm_score(csm_model, tokenizer, mutation, sequence):
    wt, pos, mt = mutation[0], int(mutation[1:-1]), mutation[-1]
    # Adjust for 0-based indexing
    pos -= 1
    if sequence[pos] != wt:
        raise ValueError(f"Wildtype mismatch at position {pos+1} for mutation {mutation}: expected {sequence[pos]}, found {wt}")
    
    mutated_sequence = sequence[:pos] + mt + sequence[pos+1:]
    
    inputs = tokenizer(mutated_sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = csm_model(**inputs)
    
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Get the log probability of the mutated amino acid at the mutation position
    aa_index = tokenizer.convert_tokens_to_ids(mt)
    score = log_probs[0, pos+1, aa_index].item()  # +1 for special token offset
    print(f"Computed CSM score for mutation {mutation}: {score}")
    print('testing ')
    print(tokenizer(sequence)["input_ids"][:10])
    return score

def mut_distro_plot(csm_data):

    # method 1: seaborn kdeplot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=csm_data,
        x="csm_score",
        hue="clinvar_label",
        fill=True,
        alpha=0.3
    )

    plt.title("Density of CSM Scores by ClinVar Label")
    plt.xlabel("csm_score")
    plt.ylabel("Density")
    sns.despine()
    plt.savefig("/g/data/gi52/jaime/trained/esm2_650M_model/missense/run9.1/distro_curves/tp53_csm_score_plot1.png", dpi=300)
    plt.close()
    print("Saved to /g/data/gi52/jaime/trained/esm2_650M_model/missense/run9.1/distro_curves/tp53_csm_score_plot1.png")

    # method 2: plotnine (ggplot2-like)
    p = (
        ggplot(csm_data, aes(x="csm_score", color="clinvar_label", fill="clinvar_label"))
        + geom_density(alpha=0.3)
        + theme_minimal()
    )
    p.save("/g/data/gi52/jaime/trained/esm2_650M_model/missense/run9.1/distro_curves/tp53_csm_score_plot2.png", width=8, height=5, dpi=300)
    print("Saved to /g/data/gi52/jaime/trained/esm2_650M_model/missense/run9.1/distro_curves/tp53_csm_score_plot2.png")


def main():

    # Load ESM base model

    # Load original ESM-2 model
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
            score = compute_csm_score(csm_model, tokenizer, mutation, tp53)
            tp53_clinvar.at[row.Index, 'csm_score'] = score
        except ValueError as e:
            print(e)
            tp53_clinvar.loc[row.Index, 'csm_score'] = None  

    tp53_clinvar.to_csv("/g/data/gi52/jaime/clinvar/tp53_clinvar_with_csm_scores.csv", index=False)

    # rename columns for plotting
    csm_data = tp53_clinvar.rename(
        columns={"Germline.classification": "clinvar_label"}
    )

    # Graph the distribution of scores for each ClinVar category
    mut_distro_plot(csm_data)



if __name__ == "__main__":
    main()   