import pandas as pd
import matplotlib.pyplot as plt

all_achitectures_csv_path = 'all_architectures_metrics/all_architectures_metrics.csv'

df = pd.read_csv(all_achitectures_csv_path)

df = df.dropna(subset=['regime', 'hellinger_dist_test'])
df["regime"] = df["regime"].astype(str)

# Merge equivalent regimes
df["regime"] = df["regime"].replace({
    "h=1.0⁻⁶": "h=10⁻⁶"
})

print(df["regime"].unique())

plt.figure(figsize=(8, 5))

df.boxplot(
    column="hellinger_dist_test",
    by="regime"
)

plt.xlabel("Regime")
plt.ylabel("Hellinger Distance Test")
plt.title("Hellinger Distance Test Distribution by Regime")
plt.suptitle("")
plt.tight_layout()
plt.show()