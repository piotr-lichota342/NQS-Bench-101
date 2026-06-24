from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Planned plots (separately for each regime):
- Training time vs number of parameters (NAS-Bench-101) ✔,
- Performance vs network width ✔
- Performance vs number of hidden layers ✔
- Performance vs activation function ✔
- Performance vs trained regimes ✔
- Performance vs network depth (NAS-Bench-101) ✔ (HL+2)
- Performance for custom architectures (width sequences)
- Descriptive statistics for each regime ✔
- Performance: fixed widths vs custom architectures
- Statistics of datasets (configurations and amplitudes) ✔
- Over and underparameterized regions
- Training time vs performance (NAS-Bench-101)
- Aggregated impact (NAS-Bench-101)
- Locality vs depth
- Locality vs width

"""

csv_file_path = 'dataset_statistics/dataset_statistics.csv'
all_achitectures_csv_path = 'all_architectures_metrics/all_architectures_metrics.csv'

df_all_architectures = pd.read_csv(all_achitectures_csv_path)
df_all_architectures = df_all_architectures.dropna(subset=['train_time (s)', 'hellinger_dist_test', 'train_params'])

train_time = df_all_architectures['train_time (s)'].dropna().values
infid = df_all_architectures['infidelity'].dropna().values
hell_test = df_all_architectures['hellinger_dist_test'].dropna().values
test_loss = df_all_architectures['test_loss'].dropna().values
r2_test = df_all_architectures['R2_test'].dropna().values
mse_test = df_all_architectures['MSE_test'].dropna().values
mae_test = df_all_architectures['MAE_test'].dropna().values
params = df_all_architectures['train_params'].dropna().values
epochs = df_all_architectures['epochs'].dropna().values
loss_function = df_all_architectures['loss_fn'].dropna().values
regimes = df_all_architectures['regime'].dropna().values

activations = df_all_architectures['activation_fn'].dropna().values
optimizers = df_all_architectures['optimizer_name'].dropna().values
hell_test = df_all_architectures['hellinger_dist_test'].dropna().values
test_loss = df_all_architectures['test_loss'].dropna().values

# Simulated training time
training_time = train_time

# Simulated validation accuracy
accuracy = hell_test

# Plot
fig, ax = plt.subplots(figsize=(6, 5))

sc = ax.scatter(
    params,
    training_time,
    c=accuracy,
    cmap="viridis",
    s=8,
    alpha=0.5,
    norm=LogNorm(),
    linewidths=1,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("# of trainable parameters (log-scale)")
ax.set_ylabel("training time (seconds) (log-scale)")

cbar = plt.colorbar(sc)
cbar.set_label("Hellinger Distance (test set) (log-scale)")

plt.savefig("dataset_statistics/training_time_vs_params.png")
plt.close()