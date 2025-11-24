import torch
import os

base_dir = "/g/data/gi52/jaime/trained/esm2_650M_model/missense/run6"
progress_file = os.path.join(base_dir, "progress_backup.pt")

# --- Load the file ---
if os.path.exists(progress_file):
    checkpoint = torch.load(progress_file)

    print("Before:", checkpoint)


    # --- Modify the epoch ---
    checkpoint["epoch"] = checkpoint["epoch"] + 1

    # --- Save it back ---
    torch.save(checkpoint, progress_file)

    print("After:", torch.load(progress_file))
else:
    print("progress.pt not found!")



    #Before: {'epoch': 1, 'batch_idx': 34989}