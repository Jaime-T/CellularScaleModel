'''
    Print out the quantitative measure of the csm vs esm score using labels
    Including ROC, AUC 
'''
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    precision_score, recall_score
  
)

def main():
    print('hi')
    


    df = pd.read_csv("/g/data/gi52/jaime/clinvar/run11_ms/epoch0_batch5000_tp53_clinvar_csm_scores.csv")

    df_filtered = df[df["Germline.classification"].isin(['Pathogenic', 'Benign'])].copy()

    # Drop rows with NaN values in key columns
    df_filtered = df_filtered.dropna(subset=["Germline.classification", "csm_score", "esm_score"])

    # y_true: 1D array of 0/1 labels. 0 = benign, 1 = pathogenic 
    y_true = df_filtered["Germline.classification"].map({'Benign': 0, 'Pathogenic': 1}).values

    # y_score: 1D array of float scores from your model (higher = more positive)
    y_csm = df_filtered["csm_score"].values
    y_esm = df_filtered["esm_score"].values

    # Compute ROC AUC and PR AUC for csm
    # Apply negative since leower scores = more pathogenic 
    roc_auc_csm = roc_auc_score(y_true, -y_csm)
    ap_csm = average_precision_score(y_true, -y_csm)  # PR AUC

    # Compute ROC AUC and PR AUC for esm
    roc_auc_esm = roc_auc_score(y_true, -y_esm)
    ap_esm = average_precision_score(y_true, -y_esm)  # PR AUC

    # Get ROC and PR curve data
    fpr_csm, tpr, roc_thresh = roc_curve(y_true, -y_csm)
    prec_csm, rec, pr_thresh = precision_recall_curve(y_true, -y_csm)

    # Get ROC and PR curve data
    fpr_esm, tpr, roc_thresh = roc_curve(y_true, -y_esm)
    prec_esm, rec, pr_thresh = precision_recall_curve(y_true, -y_esm)

    # Print results
    print("CSM:")
    print(f"Number of variants analyzed: {len(y_true)}")
    print(f"ROC AUC: {roc_auc_csm:.3f}")
    print(f"PR AUC (Average Precision): {ap_csm:.3f}")
    print(f"ROC curve points: {len(fpr_csm)}")
    print(f"PR curve points: {len(prec_csm)}")

    # Do same for ESM:
    # Print results
    print("ESM:")
    print(f"Number of variants analyzed: {len(y_true)}")
    print(f"ROC AUC: {roc_auc_esm:.3f}")
    print(f"PR AUC (Average Precision): {ap_esm:.3f}")
    print(f"ROC curve points: {len(fpr_esm)}")
    print(f"PR curve points: {len(prec_esm)}")


if __name__ == "__main__":
    main()   