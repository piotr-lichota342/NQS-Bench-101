import pandas as pd
import matplotlib.pyplot as plt

"""
Mutations to be added:
- activation function change
- depth change +/- 1
- width change
"""

all_archs = pd.read_csv('all_architectures_metrics/all_architectures_metrics.csv')
all_archs = all_archs.drop_duplicates(subset=["regime", "epochs", "activation_fn", "network_width", "hidden_layers"])



desired_regime = "h=0.5"
desired_metric = "hellinger_dist_test"
n_iterations = 100
population_size = 5
update_size = 5


metric_list = []

arch_population = all_archs[all_archs["regime"]==desired_regime].sample(n=population_size)

for iter in range(n_iterations):
    df_sorted = arch_population.sort_values(by=desired_metric, ascending=True)
    metric_list.append(df_sorted.iloc[0][desired_metric])
    df_new = df_sorted.iloc[:-2]
    pool = all_archs[all_archs["regime"] == desired_regime]
    available_pool = pool.drop(arch_population.index, errors="ignore")

    new_samples = available_pool.sample(n=update_size)
    merged_df = pd.concat([df_new, new_samples])

    arch_population = merged_df


#metric_list = [int(x) for x in metric_list]
metric_list = [float(x) for x in metric_list]
print("metric_list: ", metric_list)
plt.plot(range(n_iterations), metric_list, marker='^', linestyle='--', label='RS')

plt.xlabel("iterations")
plt.ylabel("Hellinger distance (test set)")

plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()
