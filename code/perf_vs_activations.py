import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



metrics = ["hellinger_dist_test"]

csv_file_path = 'dataset_statistics/dataset_statistics.csv'
all_achitectures_csv_path = 'all_architectures_metrics/all_architectures_metrics.csv'

df_all_architectures = pd.read_csv(all_achitectures_csv_path)
df_all_architectures = df_all_architectures.dropna(subset=['activation_fn', 'hellinger_dist_test'])
#df_all_architectures = df_all_architectures[df_all_architectures['hidden_layers'] != 'custom']

df_all_architectures["activation_fn"] = df_all_architectures["activation_fn"].astype(str)

print(df_all_architectures['activation_fn'].unique())

df = df_all_architectures

x = (df["activation_fn"]).values
print(f"Activation functions: {x}")
width = 0.25

plt.figure(figsize=(8, 5))

df.boxplot(column="hellinger_dist_test", by="activation_fn")

plt.xlabel("Activation Function")
plt.ylabel("Hellinger Distance Test")
plt.title("Hellinger Distance Test Distribution by Activation Function")
plt.suptitle("")
plt.tight_layout()
plt.show()