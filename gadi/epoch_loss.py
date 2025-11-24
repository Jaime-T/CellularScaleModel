import re
import matplotlib.pyplot as plt
# Path to your log file
log_file = "training_log_ms_run8.txt"

# Lists to store extracted batch numbers and losses
batches = []
losses = []

# Regular expression to match the loss line
pattern = re.compile(r"Epoch\s+(\d+),\s+Batch\s+(\d+)/\d+\s+\|\s+Avg Loss during training:\s+([\d.]+)")

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            batch = int(match.group(2))
            loss = float(match.group(3))
            batches.append(batch)
            losses.append(loss)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(batches, losses, marker='o', linewidth=1.5)
plt.xlabel("Batch number")
plt.ylabel("Average Training Loss")
plt.title("Missense: Average Loss During Training Epoch 0")
plt.grid(True)
plt.tight_layout()
plt.savefig("epoch_loss_plot_ms_run8.png",dpi=300)
plt.show()