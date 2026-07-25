import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd

csv_file_path = 'dataset_statistics/dataset_statistics.csv'
all_achitectures_csv_path = 'all_architectures_metrics/all_architectures_metrics.csv'

df_all_architectures = pd.read_csv(all_achitectures_csv_path)
df_all_architectures = df_all_architectures.dropna(subset=['hidden_layers', 'R2_test'])

from pathlib import Path

'''root_folder = Path("all_architectures")

# Find all CSV files recursively
csv_files = list(root_folder.rglob("*.csv"))

# Read and concatenate all CSVs
combined_df = pd.concat(
    (pd.read_csv(file) for file in csv_files),
    ignore_index=True
)'''
'''combined_df.to_csv('all_architectures_metrics/all_architectures_metrics.csv', index=False)
combined_df = combined_df.drop_duplicates()'''

# Existing dataframe
# existing_df = pd.read_csv("existing.csv")

'''df = pd.read_csv('all_architectures_metrics/all_architectures_metrics.csv')
df = df.dropna(subset=['train_time (s)', 'valid_loss',
                       'train_params', 'epochs'])'''

df = df_all_architectures
df = df.copy()



# Append the combined CSV data to the existing dataframe
#df = pd.concat([df, combined_df], ignore_index=True)

#
#df = df[df['date'] == '20260613_193047'] 

best_row = df.loc[df["R2_test"].idxmax()]

print("The best row is: ",best_row)


df = df[df['activation_fn'] == 'gelu'] 
print("After filtering for gelu, the shape of the dataframe is: ", df.shape)
df = df[df['epochs'] == 60] 
print("After filtering for 60 epochs, the shape of the dataframe is: ", df.shape)
'''df = df[df['hidden_layers'] == '2'] 
print("After filtering for 2 hidden layers, the shape of the dataframe is: ", df.shape)'''

df = df[df['regime'] == 'h=0.5'] 
print("After filtering for the regime the dataframe is: ", df.shape)

df = df[df['hidden_layers'].isin(
    ['custom']
)]

# Average R² for each network width
widths = ['1', '2', '3']

df["n_hidden_layers"] = (
    df["network_width"]
          .astype(str)
          .str.count("_") + 1
)

r2_by_act = (
    df.groupby('n_hidden_layers')['R2_test']
      .mean()
      .reindex(widths)
)

print(r2_by_act)

df.to_csv('all_architectures_metrics/filtered_architectures.csv', index=False)

print("After filtering for the network widths, the dataframe is:", df.shape)






print(df['network_width'].unique())

print(df['hidden_layers'].dtype)



'''df_filtered = df[
    df["epochs"] >= 50
]

# Find rows with same A, B, C values
arch_cols = [
    'regime',
    'hidden_layers'
]

required = {'16', '32', '64', '128', '256', '512', '2048', '4096', '8192'}

grouped = df_filtered.groupby(arch_cols)

valid_archs = grouped.filter(
    lambda g: required.issubset(set(g['network_width']))
)

single_valid = valid_archs[['network_width'] + arch_cols].sort_values(arch_cols + ['network_width'])
print(single_valid.iloc[0:3])


df_arch = valid_archs.copy()
# Data
# Activation order
widths = ['16', '32', '64', '128', '256', '512', '2048', '4096', '8192']

print(sorted(df_arch["network_width"].unique()))
print(df_arch["network_width"].value_counts().sort_index())
'''


layers = [1, 2, 3]
# Compute mean R2_test per activation (for ONE architecture)
r2_by_act = (
    df.groupby('n_hidden_layers')['R2_test']
    .mean()
    .reindex(layers)
)

# Build pairwise difference matrix (or replace with absolute values if you prefer)
n = len(widths)
data = np.zeros((n, n))

n = len(layers)

for i, a in enumerate(layers):
    for j, b in enumerate(layers):
        data[i, j] = 100 * (r2_by_act[b] - r2_by_act[a])

rows = cols = layers

fig, ax = plt.subplots(figsize=(11, 11))

# Heatmap
im = ax.imshow(data, cmap='plasma')

# Tick labels
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(rows)))
ax.set_xticklabels(cols)
ax.set_yticklabels(rows)

# Put x labels on top
ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False, labelsize=23)

# Annotate cells
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        txt = ax.text(
            j, i,
            f"{data[i, j]:+.2f}%",
            ha="center", va="center",
            color="white",
            fontsize=36,
            path_effects=[
                pe.withStroke(linewidth=2, foreground="black")
            ]
        )

# Axis labels

ax.set_ylabel("number of hidden layers (baseline)", fontsize=23)
ax.set_xlabel("number of hidden layers (target)", fontsize=23)



# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Optional: no colorbar to match your figure
# plt.colorbar(im)

fig.text(0.5, 0.03,
         "R² (test set) " +  "\nh=0.5, gelu, 60 epochs",
         ha="center", fontsize=23)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()