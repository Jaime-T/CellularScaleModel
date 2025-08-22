# -*- coding: utf-8 -*-

"""
For frameshift mutation data:
Train test set and tokenize inputs

"""
import os
import tempfile
import pandas as pd
import numpy as np

tmpdir = os.getenv('TMPDIR', tempfile.gettempdir())
mpl_cache = os.path.join(tmpdir, 'matplotlib-cache')
os.makedirs(mpl_cache, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_cache

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import EsmForMaskedLM, EsmTokenizer
import random
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F

class TorchDataset(Dataset):
    def __init__(self, data):
        """
        data: can be a pandas DataFrame or a dictionary of lists
        """
        if hasattr(data, "to_dict"):  # Convert DataFrame to dict
            data = data.to_dict(orient="list")
        self.data = data

    def __len__(self):
        # Return number of samples
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        # Return one sample as a dictionary of tensors
        return {
            key: torch.tensor(self.data[key][idx]) for key in self.data
        }

def tokenize_and_mask_seqs(batch, tokenizer, window_size: int = 1022, mlm_probability: float = 0.15):
    # Tokenize the batch
    encoded_seqs = tokenizer(
        batch['windowed_seq'].tolist(),
        padding="max_length",
        truncation=True,
        max_length=min(window_size, tokenizer.model_max_length),
        return_tensors="pt"  # use PyTorch for masking logic
    )

    input_ids = encoded_seqs["input_ids"]
    attention_mask = encoded_seqs["attention_mask"]

    # Clone to create targets
    targets = input_ids.clone()

    # Create probability mask (randomly choose tokens to mask)
    probability_matrix = torch.full(targets.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in targets.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Sample masked indices
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Replace selected input_ids with [MASK] token
    input_ids[masked_indices] = tokenizer.mask_token_id

    # Only keep targets for masked tokens
    targets[~masked_indices] = -100

    df = pd.DataFrame({
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": targets.tolist()
    })

    return df

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Validation loss: {avg_loss:.4f}")

def plot_loss(loss_values, descr, base_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', label='LOSS')
    plt.title(f'Loss vs Epochs for {descr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_dir}/loss_plot.png", dpi=300)
    plt.close()

def train_model(tokenizer, model, descr, train_dataset, lora_config, batch_size=6, epochs=3, lr=5e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print(f"Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")

    loss_per_epoch = []
    base_dir = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}"

    for epoch in range(epochs):
        print(f"Starting training for epoch {epoch+1}...")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"[{descr}] Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[{descr}] Epoch {epoch+1} training loss: {avg_loss:.4f}")
        loss_per_epoch.append(avg_loss)

        # save after each epoch
        print(f"Saving checkpoint epoch{epoch+1} ...")
        checkpoint_dir = os.path.join(base_dir, f"epoch{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved epoch{epoch+1} to {checkpoint_dir}")


    # Save model
    final_dir = os.path.join(base_dir, "final_merged")
    os.makedirs(final_dir, exist_ok=True)
    print("Merging and saving final model...")
    model = model.merge_and_unload()
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final {descr} model and tokenizer to {final_dir}")

    # Plot and save loss plot
    plot_loss(loss_per_epoch, descr, base_dir)

    return f"{descr} training complete"



def main():
    print('start')
    # Set reproducibility
    set_seeds(0)

    # Configuration
    batch_size = 8
    window_size = 1022
    model_params_millions = 650
    descr = "frameshift"

    # Load frameshift data
    data_path = Path("./data")
    fs_df = pd.read_parquet(data_path / "fs_train_split.parquet")

    # Split Data (80% train, 20% validate)
    valid_size = 0.25
    fs_train_df, fs_valid_df = train_test_split(fs_df, test_size=valid_size, random_state=0)
    training_sample_indices = fs_train_dataset.index
    print(training_sample_indices)

    # Load original ESM-2 model
    model_path = f"/g/data/gi52/jaime/esm2_{model_params_millions}M_model"
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)
    print('loaded esm2 tokenizer')

    fs_tokenized_df = tokenize_and_mask_seqs(fs_train_df, tokenizer, window_size)
    fs_train_dataset = TorchDataset(fs_tokenized_df)
    print("Number of ms training samples:", len(fs_train_dataset))


    # Set up LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["query"],  # PEFT will insert LoRA into matching linear layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.TOKEN_CLS,  # Best fit for masked token modelling
    )

    print('\n\nstarting training!')

    train_model(tokenizer, model, descr, fs_train_dataset, lora_config, batch_size)



if __name__ == '__main__':
    main()