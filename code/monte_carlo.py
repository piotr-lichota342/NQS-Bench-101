import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.special import softmax

df = pd.read_csv('all_architectures_metrics/all_architectures_metrics.csv')
df = df[df["regime"] == "h=1.0"].copy()

desired_metric = "hellinger_dist_test"

n_iterations = 150
batch_size = 20
alpha = 0.1

metric_list = []

df["theta"] = 0.0


for t in range(n_iterations):

    probs = softmax(df["theta"].values)

    batch = df.sample(n=batch_size, weights=probs, replace=True)

    rewards = -batch[desired_metric].values  # maximize reward

    baseline = rewards.mean()

    metric_list.append(-baseline)

    for i, (idx, row) in enumerate(batch.iterrows()):

        advantage = rewards[i] - baseline

        df.loc[row.name, "theta"] += alpha * advantage

    df["theta"] = softmax(df["theta"].values)


metric_list = [float(x) for x in metric_list]

plt.plot(range(n_iterations), metric_list, marker='*', linestyle='--', label='MC-NAS', c="black")

plt.xlabel("iterations")
plt.ylabel("Hellinger distance (test set)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()