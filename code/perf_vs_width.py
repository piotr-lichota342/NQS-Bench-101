import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

metrics = ["R2_test"]

all_achitectures_csv_path = 'all_architectures_metrics/all_architectures_metrics.csv'

df = pd.read_csv(all_achitectures_csv_path)

df = df.dropna(subset=['network_width', 'R2_test'])
df = df[pd.to_numeric(df['network_width'], errors='coerce').notna()]

df["network_width"] = df["network_width"].astype(float).astype(int)

print("Unique network widths:", sorted(df["network_width"].unique()))

# Aggregate by network width
df_plot = (
    df.groupby("network_width")[metrics]
      .mean()
      .reset_index()
      .sort_values("network_width")
)

x = np.arange(len(df_plot))
width = 0.25

plt.figure(figsize=(8, 5))

for i, m in enumerate(metrics):
    plt.bar(x + i * width, df_plot[m], width, label=m)

plt.xticks(
    x + width * (len(metrics) - 1) / 2,
    df_plot["network_width"]
)

plt.xlabel("Network Width")
plt.ylabel("Score")
plt.title("Performance Metrics vs Network Width")
plt.legend()
plt.tight_layout()
plt.show()