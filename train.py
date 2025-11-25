#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train test set and tokenize inputs

Code inspiration: https://github.com/naity/finetune-esm/blob/main/notebooks/cafa5_train.ipynb

"""
import os
import ray
from ray.data.dataset import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
<<<<<<< HEAD
=======
from transformers import AutoTokenizer, EsmModel, AutoModelForMaskedLM
>>>>>>> 271fb01b4ab02989319fa4fe194eb54244560d99
from transformers import EsmForMaskedLM, EsmTokenizer 
import random
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
<<<<<<< HEAD
from tqdm.auto import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
  
=======
>>>>>>> 271fb01b4ab02989319fa4fe194eb54244560d99

class RayTorchDataset(Dataset):
    def __init__(self, ray_dataset):
        self.data = ray_dataset.to_pandas()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }
<<<<<<< HEAD
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
=======


def tokenize_seqs(batch, tokenizer, window_size: int = 1022):
    encoded_seqs = tokenizer(
        batch['windowed_seq'].tolist(),        
        padding="max_length",                 
        truncation=True,
        max_length=min(window_size, tokenizer.model_max_length),
        return_tensors="np",  # or "pt" for PyTorch, "tf" for TensorFlow
    )
    return dict(
        input_ids=encoded_seqs["input_ids"],
        attention_mask=encoded_seqs["attention_mask"],
    )

def mask_input_ids(input_ids, tokenizer, mlm_probability=0.15):
    labels = input_ids.copy()

    # Create a mask of which tokens to mask
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Replace selected tokens with the mask token
    input_ids[masked_indices] = tokenizer.mask_token_id

    # Only compute loss on masked tokens
    labels[~masked_indices] = -100

    return input_ids, labels
>>>>>>> 271fb01b4ab02989319fa4fe194eb54244560d99

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

<<<<<<< HEAD
    df = pd.DataFrame({
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": targets.tolist()
    })
 
    return df
=======
    # Return everything as numpy (optional, depending on downstream)
    return dict(
        input_ids=input_ids.numpy(),
        attention_mask=attention_mask.numpy(),
        labels=targets.numpy()
    )
>>>>>>> 271fb01b4ab02989319fa4fe194eb54244560d99

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

<<<<<<< HEAD
=======
def train(model, train_loader, val_loader, epochs=3, lr=5e-5, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

        # Optional: Add validation loss
        if val_loader:
            evaluate(model, val_loader, device)

>>>>>>> 271fb01b4ab02989319fa4fe194eb54244560d99
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

<<<<<<< HEAD
MODEL_MAP = {
    8:   "facebook/esm2_t6_8M_UR50D",
    35:  "facebook/esm2_t12_35M_UR50D",
    150: "facebook/esm2_t30_150M_UR50D",
    650: "facebook/esm2_t33_650M_UR50D",
}

def get_base_model(num_params_millions: int):
    try:
        model_name = MODEL_MAP[num_params_millions]
    except KeyError:
        raise ValueError(
            f"Unknown model size {num_params_millions}M. "
            f"Available options: {list(MODEL_MAP.keys())}"
        )

    model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    return model, tokenizer

def plot_loss(loss_values, descr, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', label='LOSS')
    plt.title(f'Loss vs Epochs for {descr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_plot.png", dpi=300)
    plt.close()

# @ray.remote(num_cpus=8, num_gpus=1)
def train_model(tokenizer, model, description, save_dir, train_dataset, lora_config, batch_size, epochs=5, lr=5e-5):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    # NEW STUFF!!
    accelerator = Accelerator()
    device = accelerator.device

    # attach PEFT adapter
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # set up optimiser
    optimizer = AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
   

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )


    print("Batches per epoch:", len(train_loader))

    loss_per_epoch = []
    for epoch in range(epochs):
        print(f'Epoch {epoch+1} starting')
        model.train()
        total_loss = 0
        for batch in train_loader:
 
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            '''
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()'''

        avg_loss = total_loss / len(train_loader)
        print(f"[{description}] Epoch {epoch+1} training loss: {avg_loss:.4f}")
        loss_per_epoch.append(avg_loss)
        

    # Save model
    print("Merging and saving model...")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    tokenizer.save_pretrained(save_dir)

    #model = model.merge_and_unload()
    #model.save_pretrained(save_dir)
    #tokenizer.save_pretrained(save_dir)
    print(f"Saved model and tokenizer to {save_dir}")

    # Plot and save loss plot
    plot_loss(loss_per_epoch, description, save_dir)

    return f"{description} training complete"



=======
>>>>>>> 271fb01b4ab02989319fa4fe194eb54244560d99
def main():
    # Set reproducibility 
    set_seeds(0)

<<<<<<< HEAD
    # Configuration
    batch_size = 8
    window_size = 1022
    model_params_millions = 35

    # Load data
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "all_ms_samples.parquet")
    fs_df = pd.read_parquet(data_path / "all_fs_samples.parquet")

    ### NEW!!
    # Split data into 60% train, 20% validate, 20% test 

    test_size = 0.2
    ms_train_df, ms_test_df = train_test_split(ms_df, test_size=test_size, random_state=0)
    fs_train_df, fs_test_df = train_test_split(fs_df, test_size=test_size, random_state=0)

    # Split Data 0.25 of 80% = 20%
    valid_size = 0.25  
    ms_train_df, ms_valid_df = train_test_split(ms_train_df, test_size=valid_size, random_state=0)
    fs_train_df, fs_valid_df = train_test_split(fs_train_df, test_size=valid_size, random_state=0)

    # Save test samples for later:
    data_path = Path.cwd() / "data"
    ms_test_df.to_parquet(data_path / "ms_test_samples.parquet")
    fs_test_df.to_parquet(data_path / "fs_test_samples.parquet")

    model, tokenizer = get_base_model(model_params_millions)
    ms_tokenized_df = tokenize_and_mask_seqs(ms_train_df, tokenizer, window_size)
    ms_train_dataset = TorchDataset(ms_tokenized_df)  
    print("Number of ms training samples:", len(ms_train_dataset))

    fs_tokenized_df = tokenize_and_mask_seqs(fs_train_df, tokenizer, window_size)
    fs_train_dataset = TorchDataset(fs_tokenized_df)  
    print("Number of fs training samples:", len(fs_train_dataset))

=======
    # Initialise Ray, a distributed computing framework
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    #print(ray.cluster_resources())

    # Set ray to be deterministic 
    ray.data.DatasetContext.get_current().execution_options.preserve_order = (
        True  
    )

    num_workers = 1
    num_devices = 1
    resources_per_worker = {"CPU": 8} # add later: , "GPU": 1 

    # Load data
    data_path = Path("./data")
    ms_df = pd.read_parquet(data_path / "ms_train_split.parquet")
    fs_df = pd.read_parquet(data_path / "fs_train_split.parquet")

    # Split Data (80% train, 20% validate)
    valid_size = 0.25  
    ms_train_df, ms_valid_df = train_test_split(ms_df, test_size=valid_size, random_state=0)
    fs_train_df, fs_valid_df = train_test_split(fs_df, test_size=valid_size, random_state=0)

    # Load Tokeniser and Model
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)
    model.eval()  

    window_size = 1022

    # Convert DataFrames to Ray for batch processing
    ms_ray_ds = ray.data.from_pandas(ms_train_df)
    fs_ray_ds = ray.data.from_pandas(fs_train_df)

    # Tokenize and mask 
    ms_ray_ds = ms_ray_ds.map_batches(
        lambda batch: tokenize_and_mask_seqs(batch, tokenizer, window_size),
        batch_format="pandas"
    )

    fs_ray_ds = fs_ray_ds.map_batches(
        lambda batch: tokenize_and_mask_seqs(batch, tokenizer, window_size),
        batch_format="pandas"
    )

    
    # Convert Ray Datasets to PyTorch Datasets
    ms_train_dataset = RayTorchDataset(ms_ray_ds)
    fs_train_dataset = RayTorchDataset(fs_ray_ds)

    # Create DataLoaders
    batch_size = 8
    ms_loader = DataLoader(ms_train_dataset, batch_size=batch_size, shuffle=True)
    fs_loader = DataLoader(fs_train_dataset, batch_size=batch_size, shuffle=True)

    # Train on one dataset for now (e.g. ms)
    train(model, ms_loader, val_loader=None, epochs=3)


    exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  
>>>>>>> 271fb01b4ab02989319fa4fe194eb54244560d99

    # Set up LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["query"],  # PEFT will insert LoRA into matching linear layers
<<<<<<< HEAD
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,   # ! Changed from TOKEN_CLS !
    )

    print('\n\nstarting training!')

    train_model(tokenizer, model, "model_missense", "./train_{params_millions}_2/model_missense", ms_train_dataset, lora_config, batch_size)
    train_model(tokenizer, model, "model_frameshift", "./train_{params_millions}_2/model_frameshift", fs_train_dataset, lora_config, batch_size)

if __name__ == '__main__':
    main()
   
=======
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.TOKEN_CLS,  # Best fit for masked token modelling
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    wildtype = ["MKTFFVAGV<mask>AGK", "GV<mask>AGK"]  # Random dummy sequence with one mask
    inputs = tokenizer(wildtype, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # optimiser  setup
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = optim.Adam(lora_params, lr=1e-3)

    # Train model on masked tokens, predict original tokens
    # Training loop
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        prediction = logits[mask_token_index]
        target_id = tokenizer.convert_tokens_to_ids("G")  # Assume we want 'G' at <mask>
        loss = nn.CrossEntropyLoss()(prediction, torch.tensor([target_id], device=device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        prediction = logits[mask_token_index]
        probs = torch.softmax(prediction, dim=-1)
        topk = torch.topk(probs, k=5, dim=-1)
        top_tokens = tokenizer.convert_ids_to_tokens(topk.indices[0].tolist())
        print("Top predictions at <mask>:", top_tokens)


    

    



if __name__ == '__main__':
    main()
   
>>>>>>> 271fb01b4ab02989319fa4fe194eb54244560d99
