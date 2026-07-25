import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("all_architectures_metrics/all_architectures_metrics.csv")
df = df.dropna(subset=["train_time (s)", "infidelity"])
#df = df[df["regime"] == "h=0.5"]
df = df[df["date"] == "20260613_193047"]

# Keep only the 5 most common epoch budgets
top5_budgets = df["epochs"].value_counts().head(5).index
df = df[df["epochs"].isin(top5_budgets)]

fig, ax = plt.subplots(figsize=(6, 5))

# Color palette
colors = plt.cm.tab10.colors

for color, budget in zip(colors, sorted(top5_budgets)):
    subset = df[df["epochs"] == budget]

    ax.scatter(
        subset["train_time (s)"] / 60,
        subset["infidelity"],
        s=6,
        color=color,
        alpha=0.5,
        edgecolors="black",
        linewidths=0.3,
        label=f"Budget = {budget}"
    )

ax.set_xscale("log")
ax.set_xlabel("Training time (minutes) (log scale)")
ax.set_ylabel(r"$\mathrm{Infidelity}$")

ax.legend(
    loc="best",
    frameon=True,
    facecolor="white",
    edgecolor="black",
    fancybox=False,
    fontsize=10
)
ax.set_title("Training Time vs Infidelity (h=2.0)")
plt.tight_layout()
plt.show()