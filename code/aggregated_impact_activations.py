import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd

csv_file_path = 'dataset_statistics/dataset_statistics.csv'
all_achitectures_csv_path = 'all_architectures_metrics/all_architectures_metrics.csv'

df_all_architectures = pd.read_csv(all_achitectures_csv_path)
df_all_architectures = df_all_architectures.dropna(subset=['activation_fn', 'R2_test'])

df_filtered = df_all_architectures[
    (df_all_architectures['epochs'] >= 120)
]

# Find rows with same A, B, C values
arch_cols = [
    'regime',
    'train_params',
    'epochs',
    'batch_size',
    'network_width',
    'hidden_layers'
]

required = {'gelu', 'relu', 'tanh'}

grouped = df_filtered.groupby(arch_cols)

valid_archs = grouped.filter(
    lambda g: required.issubset(set(g['activation_fn']))
)

single_valid = valid_archs[['activation_fn'] + arch_cols].sort_values(arch_cols + ['activation_fn'])
print(single_valid.iloc[0:3])


arch_key = valid_archs[arch_cols].drop_duplicates().iloc[0].to_dict()

df_arch = df_all_architectures.merge(
    pd.DataFrame([arch_key]),
    on=arch_cols
)
# Data
# Activation order
acts = ['gelu', 'relu', 'tanh']

# Compute mean R2_test per activation (for ONE architecture)
r2_by_act = (
    df_arch.groupby('activation_fn')['R2_test']
    .mean()
    .reindex(acts)
)

# Build pairwise difference matrix (or replace with absolute values if you prefer)
data = np.zeros((3, 3))

for i, a in enumerate(acts):
    for j, b in enumerate(acts):
        data[i, j] = r2_by_act[b] - r2_by_act[a]

rows = cols = acts

fig, ax = plt.subplots(figsize=(5, 4))

# Heatmap
im = ax.imshow(data, cmap='plasma')

# Tick labels
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(rows)))
ax.set_xticklabels(cols)
ax.set_yticklabels(rows)

# Put x labels on top
ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)

# Annotate cells
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        txt = ax.text(
            j, i,
            f"{data[i, j]:.2f}%",
            ha="center", va="center",
            color="white",
            fontsize=14,
            path_effects=[
                pe.withStroke(linewidth=2, foreground="black")
            ]
        )

# Axis labels

ax.set_ylabel("activation (baseline)")
ax.set_xlabel("activation (target)")



# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Optional: no colorbar to match your figure
# plt.colorbar(im)

fig.text(0.5, 0.03,
         "R² (test set) difference between activations" +  "\nh=0.5, 500 epochs, \nwidth sequence 8,16,32,64,128,246,512,2048",
         ha="center")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()