import pandas as pd
import matplotlib.pyplot as plt
import random

"""
Mutations to be added:
- activation function change
- depth change +/- 1
- width change
"""

available_pool = pd.read_csv('all_architectures_metrics/all_architectures_metrics.csv')
available_pool = available_pool.copy()

print(available_pool.shape)

available_pool = available_pool[available_pool['date'].isin(
    ['20260613_193047']
)]
'''
available_pool = available_pool[available_pool['epochs'].isin(
    [60]
)]'''



print(available_pool.shape)

# Count hidden layers
available_pool["n_custom_hidden_layers"] = (
    available_pool["network_width"]
    .astype(str)
    .str.split("_")
    .str.len()
)

desired_regime = "h=2.0"
desired_metric = "hellinger_dist_test"

n_iterations = 50
population_size = 15



metric_list = []
#print("All desired rows: ", all_archs[all_archs["regime"]==desired_regime][desired_metric].shape)
#print("Unique rows: ", all_archs[all_archs["regime"]==desired_regime][desired_metric].unique()


arch_population = available_pool[available_pool["regime"]==desired_regime].sample(n=population_size)
available_pool = available_pool.drop(arch_population.index, errors="ignore")

arch_population["birth_iteration"] = 0

#population_new = df_sorted.iloc[:-2]
'''pool = available_pool[available_pool["regime"] == desired_regime]
available_pool = pool.drop(arch_population.index, errors="ignore")

new_samples = available_pool.sample(n=update_size)
merged_df = pd.concat([population_new, new_samples])

arch_population = merged_df'''

n_experiments = 20

all_metrics = []

for experiment in range(n_experiments):

    print(f"Experiment {experiment+1}/{n_experiments}")

    # Reset pool for every experiment
    pool = available_pool.copy()

    metric_list = []

    # Initialize population
    arch_population = pool[
        pool["regime"] == desired_regime
    ].sample(n=population_size)

    pool = pool.drop(arch_population.index, errors="ignore")

    arch_population["birth_iteration"] = 0

    for iter in range(n_iterations):

        # Record best metric
        best = arch_population[desired_metric].min()
        metric_list.append(float(best))

        # Select best architecture for mutation
        first_arch = arch_population.loc[
            arch_population[desired_metric].idxmin()
        ]

        # Remove oldest architecture
        population_new = arch_population.sort_values(
            "birth_iteration"
        ).iloc[1:]

        mutation_choice = random.randint(0, 2)

        match mutation_choice:

            case 0:  # activation function

                possible_choices = pool[
                    (pool["n_custom_hidden_layers"] == first_arch["n_custom_hidden_layers"]) &
                    (pool["activation_fn"] != first_arch["activation_fn"]) &
                    (pool["network_width"] == first_arch["network_width"])
                ]

            case 1:  # depth

                possible_choices = pool[
                    (pool["n_custom_hidden_layers"] != first_arch["n_custom_hidden_layers"]) &
                    (pool["activation_fn"] == first_arch["activation_fn"]) &
                    (pool["network_width"] == first_arch["network_width"])
                ]

            case 2:  # width

                possible_choices = pool[
                    (pool["n_custom_hidden_layers"] == first_arch["n_custom_hidden_layers"]) &
                    (pool["activation_fn"] == first_arch["activation_fn"]) &
                    (pool["network_width"] != first_arch["network_width"])
                ]

        # Skip if no mutation exists
        if possible_choices.empty:
            continue

        random_row = possible_choices.sample(n=1).copy()
        random_row["birth_iteration"] = iter + 1

        population_new = pd.concat(
            [population_new, random_row],
            ignore_index=True
        )

        pool = pool.drop(
            random_row.index,
            errors="ignore"
        )

        arch_population = population_new

    all_metrics.append(metric_list)


# Convert to dataframe:
# rows = experiments, columns = iterations
metrics_df = pd.DataFrame(all_metrics)

mean_metric = metrics_df.mean(axis=0)
std_metric = metrics_df.std(axis=0)

# -------------------------------
# Random search baseline
# -------------------------------

random_metrics = []

for experiment in range(n_experiments):

    print(f"Random search experiment {experiment+1}/{n_experiments}")

    pool = available_pool[
        available_pool["regime"] == desired_regime
    ].copy()

    experiment_metrics = []

    best_so_far = float("inf")

    for iteration in range(n_iterations):

        # Randomly choose an architecture
        random_arch = pool.sample(n=1)

        metric = float(random_arch[desired_metric].iloc[0])

        # Keep best result found so far
        best_so_far = min(best_so_far, metric)

        experiment_metrics.append(best_so_far)

    random_metrics.append(experiment_metrics)


random_metrics_df = pd.DataFrame(random_metrics)

random_mean = random_metrics_df.mean(axis=0)
random_std = random_metrics_df.std(axis=0)


plt.plot(
    range(n_iterations),
    mean_metric,
    marker='^',
    linestyle='--',
    label=f'GE average ({n_experiments} runs)'
)

plt.plot(
    range(n_iterations),
    random_mean,
    marker='o',
    linestyle='--',
    label=f'Random search ({n_experiments} runs)'
)

# optional uncertainty band
plt.fill_between(
    range(n_iterations),
    mean_metric - std_metric,
    mean_metric + std_metric,
    alpha=0.2
)

plt.xlabel("iterations")
plt.ylabel("Hellinger distance (test set)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.3)

plt.title(
    f"Regime: {desired_regime}, Population size: {population_size} "
)

plt.tight_layout()
plt.show()
