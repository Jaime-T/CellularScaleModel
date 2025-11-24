# -*- coding: utf-8 -*-

"""
For frameshift mutation data: train pretrained ESM2 model 

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
from accelerate import Accelerator

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

def plot_loss(loss_per_epoch, descr, base_dir):
    # Unpack lists
    epochs = [e[0] for e in loss_per_epoch]
    train_losses = [e[1] for e in loss_per_epoch]
    val_losses = [e[2] for e in loss_per_epoch]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='s', label='Validation Loss')
    plt.title(f'Loss vs Epochs for {descr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(base_dir, "loss_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved loss plot to {plot_path}")

def print_process_info(accelerator, tag=""):
    print(
        f"[Process {accelerator.process_index}/{accelerator.num_processes}] "
        f"Local rank: {accelerator.local_process_index} | "
        f"Device: {accelerator.device} | Tag: {tag}"
    )

def generate_heatmap(protein_sequence, model, tokenizer, accelerator, start_pos=1, end_pos=None):
    model.eval()
    device = accelerator.device

    decoded = tokenizer(protein_sequence, return_tensors="pt")
    decoded = {k: v.to(device) for k, v in decoded.items()}
    input_ids = decoded['input_ids']
    sequence_length = input_ids.shape[1] - 2

    if end_pos is None:
        end_pos = sequence_length

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    heatmap = np.zeros((20, end_pos - start_pos + 1))

    for position in range(start_pos, end_pos + 1):
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, position] = tokenizer.mask_token_id
        masked_input_ids = masked_input_ids.to(device)

        with torch.no_grad():
            logits = model(masked_input_ids).logits
            probabilities = torch.nn.functional.softmax(logits[0, position], dim=0)
            log_probabilities = torch.log(probabilities)

        wt_residue = input_ids[0, position].item()
        log_prob_wt = log_probabilities[wt_residue].item()

        for i, aa in enumerate(amino_acids):
            aa_id = tokenizer.convert_tokens_to_ids(aa)
            log_prob_mt = log_probabilities[aa_id].item()
            heatmap[i, position - start_pos] = log_prob_mt - log_prob_wt

    return heatmap, amino_acids

def plot_heatmap(gene, data, title, sequence, amino_acids, base_dir):
    plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis_r", aspect="auto", vmin=-20, vmax=20)
    plt.yticks(range(20), amino_acids)
    plt.ylabel("Amino Acid Mutations")

    seq_len = len(sequence)
    xticks_positions = list(range(0, seq_len, 50)) # mark every 50th position
    # ensure last position is shown too
    if seq_len - 1 not in xticks_positions:
        xticks_positions.append(seq_len - 1)
    # set ticks and labels
    plt.xticks(xticks_positions, [str(pos) for pos in xticks_positions])
    plt.xlabel("Position in Protein Sequence")
    plt.title(title + ' 650M')
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.tight_layout()
    
    # Define the path
    save_path = os.path.join(base_dir, f"heatmaps/{gene}/{title.replace(' ', '_')}.png")
    folder = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300)

def train_model(accelerator, tokenizer, base_model, frozen_base_model, descr, train_dataset, valid_dataset, lora_config, batch_size=6, epochs=3, lr=5e-5):    
    device = accelerator.device

    # LoRA-wrapped trainable model
    model = get_peft_model(base_model, lora_config)
    accelerator.print(f"LoRA added. Verifying process {accelerator.process_index}")

    # set up optimiser
    optimizer = AdamW(model.parameters(), lr=lr)

    # set up scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # set up dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    accelerator.print(f"Train Batches per epoch: {len(train_loader)}")

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,shuffle=False, num_workers=4)
    accelerator.print(f"Valid Batches per epoch: {len(valid_loader)}")

    # prepare for multigpu 
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )

    # directories
    base_dir = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/acc/run2"
    if accelerator.is_main_process:
        os.makedirs(base_dir, exist_ok=True)
        accelerator.print(f'Base dir is: {base_dir}')

    # make heatmaps for frozen model:
    if accelerator.is_main_process:
        frozen_base_model = frozen_base_model.to(device)
        frozen_base_model.eval()
        # myc
        myc_gene = "myc"
        myc_sequence = "MDFFRVVENQQPPATMPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPSEDIWKKFELLPTPPLSPSRRSGLCSPSYVAVTPFSLRGDNDGGGGSFSTADQLEMVTELLGGDMVNQSFICDPDDETFIKNIIIQDCMWSGFSAAAKLVSEKLASYQAARKDSGSPNPARGHSVCSTSSLYLQDLSAAASECIDPSVVFPYPLNDSSSPKSCASQDSSAFSPSSDSLLSSTESSPQGSPEPLVLHEETPPTTSSDSEEEQEDEEEIDVVSVEKRQAPGKRSESGSPSAGGHSKPPHSPLVLKRCHVSTHQHNYAAPPSTRKDYPAAKRVKLDSVRVLRQISNNRKCTSPRSSDTEENVKRRTHNVLERQRRNELKRSFFALRDQIPELENNEKAPKVVILKKATAYILSVQAEEQKLISEEDLLRKRREQLKHKLEQLRNSCA"
        myc_base_heatmap, amino_acids = generate_heatmap(myc_sequence, frozen_base_model, tokenizer, accelerator)
        plot_heatmap(myc_gene, myc_base_heatmap, "Original ESM2 Model (LLRs)", myc_sequence, amino_acids, base_dir)

            # tp53
        tp53_gene = "tp53"
        tp53_sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
        tp53_base_heatmap, amino_acids = generate_heatmap(tp53_sequence, frozen_base_model, tokenizer, accelerator)
        plot_heatmap(tp53_gene, tp53_base_heatmap, "Original ESM2 Model (LLRs)", tp53_sequence, amino_acids, base_dir)
    
    # Make sure non-main processes do NOT call generate_heatmap for the frozen model
    accelerator.wait_for_everyone()
    loss_per_epoch = []

    # -- Training loop --
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        if accelerator.is_main_process:
            accelerator.print(f"==== Starting Epoch {epoch} on process 0 ====")
        print_process_info(accelerator, tag=f"Epoch{epoch} start")
        
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if accelerator.is_main_process and (batch_idx + 1) % 1000 == 0:  

                    # debug:
                    for name, param in model.named_parameters():
                        if param.requires_grad and "lora" in name.lower():
                            accelerator.print(name, param.view(-1)[0:5])  # Inspect first few weights
                            break

                    print_process_info(accelerator, tag=f"Epoch{epoch}, Batch{batch_idx} start")
                    batch_num = batch_idx + 1 

                    # plot heatmap for myc gene 
                    myc_fs_heatmap, _ = generate_heatmap(myc_sequence, model, tokenizer, accelerator)

                    # debug:
                    accelerator.print("Base:", myc_base_heatmap[:10])
                    accelerator.print("Fine:", myc_fs_heatmap[:10])
                    accelerator.print("Diff:", (myc_fs_heatmap - myc_base_heatmap)[:10])

                    myc_fs_diff_heatmap = myc_fs_heatmap - myc_base_heatmap
                    plot_heatmap(myc_gene, myc_fs_heatmap, f"Epoch {epoch}, Batch {batch_num}: Fine-tuned Frameshift Model (LLRs)", myc_sequence, amino_acids, base_dir)
                    plot_heatmap(myc_gene, myc_fs_diff_heatmap, f"Epoch {epoch}, Batch {batch_num}: Difference (Fine-tuned Frameshift - Original)", myc_sequence, amino_acids, base_dir)
                    
                    # plot heatmap for tp53 gene 
                    tp53_fs_heatmap, _ = generate_heatmap(tp53_sequence, model, tokenizer, accelerator)
                    tp53_fs_diff_heatmap = tp53_fs_heatmap - tp53_base_heatmap
                    plot_heatmap(tp53_gene, tp53_fs_heatmap, f"Epoch {epoch}, Batch {batch_num}: Fine-tuned Frameshift Model (LLRs)", tp53_sequence, amino_acids, base_dir)
                    plot_heatmap(tp53_gene, tp53_fs_diff_heatmap, f"Epoch {epoch}, Batch {batch_num}: Difference (Fine-tuned Frameshift - Original)", tp53_sequence, amino_acids, base_dir)
                    print_process_info(accelerator, tag=f"Heatmap at Epoch{epoch}, Batch{batch_idx} for MYC and TP53")

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valid_loader)

        accelerator.print(f"Epoch {epoch}: train loss = {avg_train_loss:.4f}, valid loss = {avg_val_loss:.4f}")

        if accelerator.is_main_process:
            loss_per_epoch.append((epoch, avg_train_loss, avg_val_loss))

        accelerator.wait_for_everyone()
        # Save model every epoch 
        if accelerator.is_main_process:
            epoch_dir = os.path.join(base_dir, f"epoch{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                epoch_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            tokenizer.save_pretrained(epoch_dir)

            accelerator.print(f"Saved epoch{epoch} model and tokenizer to {epoch_dir}")

    # Save final model after all training is done
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir = os.path.join(base_dir, "final_merged")
        os.makedirs(save_dir, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(save_dir)

        accelerator.print(f"Saved final {descr} model and tokenizer to {save_dir}")

        # Plot and save loss plot
        plot_loss(loss_per_epoch, descr, base_dir)

    return f"{descr} training complete"



def main():
    accelerator = Accelerator()
    accelerator.print("Accelerator initialized")
    print_process_info(accelerator, tag="After init")

    # Set reproducibility
    set_seeds(0)

    # Configuration
    batch_size = 8
    window_size = 1022
    model_params_millions = 650
    descr = "frameshift"

    # Load frameshift data
    data_path = Path("./data")
    fs_df = pd.read_parquet(data_path / "update2_all_fs_samples.parquet")

    # Split data into 75% train, 5% validate, 20% test 
    test_size = 0.20
    fs_train_df, fs_test_df = train_test_split(fs_df, test_size=test_size, random_state=0)
    valid_size = 0.0625 
    fs_train_df, fs_valid_df = train_test_split(fs_train_df, test_size=valid_size, random_state=0)
    accelerator.print("Train, Valid, Test split is:", len(fs_train_df), len(fs_valid_df), len(fs_test_df))

    # Load original ESM-2 model
    model_path = f"/g/data/gi52/jaime/esm2_{model_params_millions}M_model"
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    base_model = EsmForMaskedLM.from_pretrained(model_path)

    # frozen baseline
    frozen_base_model = EsmForMaskedLM.from_pretrained(model_path)
    frozen_base_model.eval()
    for p in frozen_base_model.parameters():
        p.requires_grad = False

    # tokenise and mask training data 
    fs_tokenized_df = tokenize_and_mask_seqs(fs_train_df, tokenizer, window_size)
    fs_train_dataset = TorchDataset(fs_tokenized_df)

    # tokenise and mask validation data 
    valid_tokenized_df = tokenize_and_mask_seqs(fs_valid_df, tokenizer, window_size)
    fs_valid_dataset = TorchDataset(valid_tokenized_df)

    # Set up LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["query"],  # PEFT will insert LoRA into matching linear layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.TOKEN_CLS,  # Best fit for masked token modelling
    )

    accelerator.print("Starting training...")

    if accelerator.is_main_process:
        print("\n====== MAIN PROCESS CONFIRMED ======\n")
    else:
        print(f"Worker process {accelerator.process_index} ready.")


    train_model(accelerator, tokenizer, base_model, frozen_base_model, descr, fs_train_dataset, fs_valid_dataset, lora_config, batch_size)
    accelerator.print("Training complete.")

if __name__ == '__main__':
    main()
