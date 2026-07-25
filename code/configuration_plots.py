import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("datasets/1d_tfim_N12_h1.0_full_dataset.csv")

# Preserve original order index BEFORE sorting
df["orig_idx"] = np.arange(len(df))

# Sort by amplitude/probability (for splitting only)
df_sorted = df.sort_values(by="probability", ascending=False).reset_index(drop=True)

n = len(df_sorted)
train_end = int(0.7 * n)
test_end  = int(0.9 * n)

# Assign split labels on SORTED data
df_sorted["split"] = "valid"
df_sorted.loc[:train_end-1, "split"] = "train"
df_sorted.loc[train_end:test_end-1, "split"] = "test"

# Bring labels back to ORIGINAL order
df_labeled = df_sorted.sort_values("orig_idx")

# Plot in original order
x = df_labeled["orig_idx"].values
y = df_labeled["probability"].values

plt.figure(figsize=(6, 3))

mask_train = df_labeled["split"] == "train"
mask_test  = df_labeled["split"] == "test"
mask_valid = df_labeled["split"] == "valid"

plt.scatter(x[mask_train], y[mask_train], s=10, c="black", alpha=0.5, label="Train (70%)")
plt.scatter(x[mask_test],  y[mask_test],  s=10, c="red",   alpha=0.6, label="Test (20%)")
plt.scatter(x[mask_valid], y[mask_valid], s=10, c="blue",  alpha=0.6, label="Validation (10%)")

plt.yscale("log")

plt.xlabel("original configuration index")
plt.ylabel(r"$\|\Omega(\hat{\sigma}_i)\|^2$")

plt.legend()
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("datasets/1d_tfim_N12_h1.0_full_dataset.csv")

# Sort by probability descending
df = df.sort_values("probability", ascending=False).reset_index(drop=True)

# Rank index i = 1,...,2^N
df["rank"] = np.arange(1, len(df) + 1)

# Split exactly as in panel (a):
# highest-probability configurations -> train
n = len(df)
train_end = int(0.75 * n)

df["split"] = "test"
df.loc[:train_end - 1, "split"] = "train"

# Data
x = df["rank"].values
y = df["probability"].values

mask_train = df["split"] == "train"
mask_test = df["split"] == "test"

# Plot
plt.figure(figsize=(6,4))

plt.scatter(
    x[mask_train],
    y[mask_train],
    s=10,
    c="black",
    alpha=0.6,
    label="Train"
)

plt.scatter(
    x[mask_test],
    y[mask_test],
    s=20,
    marker="*",
    c="0.7",
    alpha=0.8,
    label="Test"
)

plt.yscale("log")
plt.xscale("log")

plt.xlabel(r"$i$")
plt.ylabel(r"$|\Omega(\sigma_i)|^2$")

plt.xlim(1, len(df))
plt.legend(frameon=False)

plt.tight_layout()
plt.show()