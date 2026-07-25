import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Load data
# =============================================================================

csv_path = "all_architectures_metrics/all_architectures_metrics.csv"

df = pd.read_csv(csv_path)

df = df.dropna(subset=[
    "hidden_layers",
    "network_width",
    "R2_test",
    "train_time (s)"
])

# =============================================================================
# Apply filters
# =============================================================================
df = df[df['date'] == '20260613_193047']
df = df[df["activation_fn"] == "gelu"]
df = df[df["epochs"] == 1000]
df = df[df["regime"] == "h=2.0"]

print(df["train_time (s)"].describe())
print(df[df["train_time (s)"] < 0])


print("Filtered shape:", df.shape)


# =============================================================================
# Aggregate by width
# =============================================================================
df["network_width"] = df["network_width"].astype(int)
width_summary = (
    df.groupby("network_width")
      .agg(
          mean_r2=("R2_test", "mean"),
          mean_time=("train_time (s)", "mean")
      )
      .sort_index()
)

# =============================================================================
# Aggregate by hidden layers
# =============================================================================

df["hidden_layers"] = df["hidden_layers"].astype(int)

layers_summary = (
    df.groupby("hidden_layers")
      .agg(
          mean_r2=("R2_test", "mean"),
          mean_time=("train_time (s)", "mean")
      )
      .sort_index()
)

print(layers_summary)

print(width_summary)

# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bar_width = 0.35

# =============================================================================
# Hidden layers
# =============================================================================

ax = axes[0]
ax2 = ax.twinx()

x = np.arange(len(layers_summary))

ax.bar(
    x - bar_width/2,
    layers_summary["mean_r2"],
    width=bar_width,
    color="royalblue",
    edgecolor="black",
    label="R² (test set)"
)

ax2.bar(
    x + bar_width/2,
    layers_summary["mean_time"],
    width=bar_width,
    color="tomato",
    edgecolor="black",
    label="Training time (s)"
)

ax.set_xticks(x)
ax.set_xticklabels(layers_summary.index)

ax.set_xlabel("Number of hidden layers")
ax.set_ylabel("R² (test set)")
ax2.set_ylabel("Training time (s)")

ax.set_title("Performance vs depth")

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0
)


# =============================================================================
# Width
# =============================================================================

ax = axes[1]
ax2 = ax.twinx()

x = np.arange(len(width_summary))

ax.bar(
    x - bar_width/2,
    width_summary["mean_r2"],
    width=bar_width,
    color="royalblue",
    edgecolor="black",
    label="R² (test set)"
)

ax2.bar(
    x + bar_width/2,
    width_summary["mean_time"],
    width=bar_width,
    color="tomato",
    edgecolor="black",
    label="Training time (s)"
)

ax.set_xticks(x)
ax.set_xticklabels(width_summary.index)

ax.set_xlabel("Network width")
ax.set_ylabel("R² (test set)")
ax2.set_ylabel("Training time (s)")

ax.set_title("Performance vs width")

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0
)


plt.suptitle("gelu • 1000 epochs • h=2.0", fontsize=13)
plt.tight_layout()
plt.show()