import torch
from torch.utils.data import Dataset
import pandas as pd
import re

class TorchDataset(Dataset):
    """Wraps tokenized+masked dataframe into torch Dataset."""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }

def tokenize_and_mask_seqs(batch, tokenizer, window_size: int = 1022):
    encoded_seqs = tokenizer(
        batch['windowed_seq'].tolist(),
        padding="max_length",
        truncation=True,
        max_length=min(window_size, tokenizer.model_max_length),
        return_tensors="pt"
    )

    input_ids = encoded_seqs["input_ids"]
    attention_mask = encoded_seqs["attention_mask"]
    targets = input_ids.clone()

    for i, (prot_change, start_index) in enumerate(zip(batch['ProteinChange'], batch['start_index'])):
        m = re.match(r"p\.\D+(\d+)", prot_change)
        if m:
            mut_pos = int(m.group(1))
            window_pos = mut_pos - start_index
            token_index = window_pos

            if 0 <= token_index < input_ids.shape[1]:
                targets[i, :] = -100
                targets[i, token_index] = encoded_seqs["input_ids"][i, token_index]
                input_ids[i, token_index] = tokenizer.mask_token_id

    df = pd.DataFrame({
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": targets.tolist()
    })
    return df
    
