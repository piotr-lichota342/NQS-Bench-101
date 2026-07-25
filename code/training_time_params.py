from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

'''root_folder = Path("all_architectures/20260613_193047")

# Find all CSV files recursively
csv_files = list(root_folder.rglob("*.csv"))

# Read and concatenate all CSVs
combined_df = pd.concat(
    (pd.read_csv(file) for file in csv_files),
    ignore_index=True
)'''

# Existing dataframe
# existing_df = pd.read_csv("existing.csv")

df = pd.read_csv('all_architectures_metrics/all_architectures_metrics.csv')
df = df.dropna(subset=['train_time (s)', 'valid_loss',
                       'train_params', 'epochs'])

# Append the combined CSV data to the existing dataframe
#df = pd.concat([df, combined_df], ignore_index=True)
df = df[df['date'] == '20260613_193047'] 

#df = df[df['regime'] == 'h=0.5'] 


print(df.shape)
print(df[['date', 'regime']].head())
print(df['epochs'].value_counts())

params = df['train_params'].values
training_time = df['train_time (s)'].values
accuracy = df['valid_loss'].values
epochs = df['epochs'].astype(int).values

fig, ax = plt.subplots(figsize=(6, 5))

hilbert_space_size = 4096
ax.axvline(
    x=hilbert_space_size,
    color='gray',
    linestyle='--',
    linewidth=1.5
)

best_confs = int(4096*0.75)
ax.axvline(
    x=best_confs,
    color='blue',
    linestyle='--',
    linewidth=1.5
)

# Marker cycle (add more if needed)
markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>', 'h', '8']

# Get the 5 largest unique epoch values
top_epochs = (
    df['epochs']
    .dropna()
    .value_counts()
    .head(2)
    .index
    .tolist()
)

# Sort ascending for nicer legend ordering
top_epochs = sorted(top_epochs)

print("Top epochs:", top_epochs)

epoch_counts = (
    df[df['epochs'].isin(top_epochs)]
    .groupby('epochs')
    .size()
)

print(epoch_counts)

unique_epochs = top_epochs

for i, epoch in enumerate(unique_epochs):
    mask = epochs == epoch

    sc = ax.scatter(
        params[mask],
        training_time[mask],
        c=accuracy[mask],
        cmap='viridis',
        marker=markers[i % len(markers)],
        s=30,
        alpha=0.5,
        linewidths=0.5,
        edgecolors='black',
        label=f'{epoch} epochs'
    )



ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("# of trainable parameters")
ax.set_ylabel("training time (seconds) ")

cbar = plt.colorbar(sc)
cbar.set_label("Validation Loss")

ax.legend(
    title="Training epochs (h=0.5)",
    loc="best",
    frameon=True
)

plt.tight_layout()
plt.savefig("dataset_statistics/training_time_vs_params.png", dpi=300)
plt.close()