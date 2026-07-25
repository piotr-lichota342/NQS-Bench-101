from scipy.stats import pearsonr
import pandas as pd
import numpy as np

df = pd.read_csv('all_architectures_metrics/all_architectures_metrics.csv')

data = (
    df[['train_params', 'test_loss']]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)
r, p = pearsonr(data["train_params"], data["test_loss"])

print(f"Pearson r = {r:.3f}")
print(f"p-value = {p:.3e}")