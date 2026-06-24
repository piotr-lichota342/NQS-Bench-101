import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



metrics = ["R2_test"]

csv_file_path = 'dataset_statistics/dataset_statistics.csv'
all_achitectures_csv_path = 'all_architectures_metrics/all_architectures_metrics.csv'

df_all_architectures = pd.read_csv(all_achitectures_csv_path)
df_all_architectures = df_all_architectures.dropna(subset=['hidden_layers', 'R2_test'])
df_all_architectures = df_all_architectures[df_all_architectures['hidden_layers'] != 'custom']

df_all_architectures["hidden_layers"] = df_all_architectures["hidden_layers"].astype(float).astype(int)

print(df_all_architectures['hidden_layers'].unique())

df = df_all_architectures

x = (df["hidden_layers"]).values
print(f"Hidden layers: {x}")
width = 0.25

plt.figure(figsize=(8, 5))

for i, m in enumerate(metrics):
    plt.bar(x + i * width, df[m], width, label=m)

plt.xticks(x + width, df["hidden_layers"])
plt.xlabel("Number of Hidden Layers")
plt.ylabel("Score")
plt.title("Performance metrics vs Hidden Layers")
plt.legend()

plt.show()