import pandas as pd
import matplotlib.pyplot as plt
all_archs = pd.read_csv('all_architectures_metrics/all_architectures_metrics.csv')

desired_regime = "h=0.5"
desired_metric = "hellinger_dist_test"
n_iterations = 500

metric_list = []
for iter in range(n_iterations):
    sampled_arch = all_archs[all_archs["regime"]==desired_regime].sample(n=1)
    metric_list.append(sampled_arch[desired_metric].values)

#metric_list = [int(x) for x in metric_list]
metric_list = [float(x[0]) for x in metric_list]
#print("metric_list type: ", metric_list)
plt.plot(range(n_iterations), metric_list, marker='^', linestyle='--', label='RS')

plt.xlabel("iterations")
plt.ylabel("Hellinger distance (test set)")

plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()
