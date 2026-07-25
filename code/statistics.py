import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

csv_file_path = 'dataset_statistics/dataset_statistics.csv'
all_achitectures_csv_path = 'all_architectures_metrics/all_architectures_metrics.csv'

df_all_architectures = pd.read_csv(all_achitectures_csv_path)
#df_all_architectures = df_all_architectures[df_all_architectures['date'] == '20260613_193047']
df_all_architectures = df_all_architectures[df_all_architectures['regime'] == 'h=2.0']
DECIMALS = 4

stats_data = {
    'train_params': None,
    'test_loss':None,
    'train_loss':None,
    'valid_loss':None,
    'train_time (s)': None,
    'epochs': None,
    'network_width': None,
    'hidden_layers': None,
   
    'R2_test': None,
    'R2_train': None,
    'R2_valid': None,
    
    'MSE_test': None,
    'MSE_train': None,
    'MSE_valid': None,

    'MAE_test': None,
    'MAE_train': None,
    'MAE_valid': None,

    'hellinger_dist_test': None,
    'hellinger_dist_train': None,
    'hellinger_dist_valid': None,
    'infidelity': None
    
}

cols_list = list(stats_data.keys())

stats_list = [
    'count',
    'sum',
    'mean',
    'median',
    'mode',
    'min',
    'max',
    'std',
    'var',
    'skew',
    'kurt',
    '25th_quantile',
    '50th_quantile',
    '75th_quantile'
]

df_stats = pd.DataFrame(
    columns=cols_list,
    index=stats_list
)

for col_name in cols_list:
    if col_name == 'network_width' or col_name == 'hidden_layers':
        continue
    for stat in stats_list:
        if stat == 'count':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].count(), DECIMALS)
        elif stat == 'sum':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].sum(), DECIMALS)
        elif stat == 'mean':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].mean(), DECIMALS)
        elif stat == 'median':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].median(), DECIMALS)
        elif stat == 'mode':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].mode()[0], DECIMALS)
        elif stat == 'min':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].min(), DECIMALS)
        elif stat == 'max':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].max(), DECIMALS)
        elif stat == 'std':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].std(), DECIMALS)
        elif stat == 'var':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].var(), DECIMALS)
        elif stat == 'skew':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].skew(), DECIMALS)
        elif stat == 'kurt':
            df_stats.loc[stat, col_name] = round(df_all_architectures[col_name].kurt(), DECIMALS)
        elif stat == '25th_quantile':
            quantile_value = df_all_architectures[col_name].quantile(0.25)
            df_stats.loc[stat, col_name] = round(quantile_value, DECIMALS)
        elif stat == '50th_quantile':
            quantile_value = df_all_architectures[col_name].quantile(0.5)
            df_stats.loc[stat, col_name] = round(quantile_value, DECIMALS)
        elif stat == '75th_quantile':
            quantile_value = df_all_architectures[col_name].quantile(0.75)
            df_stats.loc[stat, col_name] = round(quantile_value, DECIMALS)

train_time = df_all_architectures['train_time (s)'].dropna().values
infid = df_all_architectures['infidelity'].dropna().values
hell_test = df_all_architectures['hellinger_dist_test'].dropna().values
test_loss = df_all_architectures['test_loss'].dropna().values
r2_test = df_all_architectures['R2_test'].dropna().values
mse_test = df_all_architectures['MSE_test'].dropna().values
mae_test = df_all_architectures['MAE_test'].dropna().values
params = df_all_architectures['train_params'].dropna().values
epochs = df_all_architectures['epochs'].dropna().values
loss_function = df_all_architectures['loss_fn'].dropna().values
regimes = df_all_architectures['regime'].dropna().values

activations = df_all_architectures['activation_fn'].dropna().values
optimizers = df_all_architectures['optimizer_name'].dropna().values

counts = Counter(loss_function)

plt.figure(figsize=(10, 6))
plt.pie(
    counts.values(),
    autopct="%1.1f%%"
)
plt.legend(
    counts.keys(),
    title="Loss Functions",
    loc="best"
)
plt.title("(NQS-Bench-101): Loss function distribution of all architectures")
plt.savefig("dataset_statistics/loss_function_piechart.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(counts.keys(), counts.values())
plt.title("(NQS-Bench-101): Loss functions of all architectures—histogram")
plt.savefig("dataset_statistics/loss_function_histogram.png")
plt.close()

counts = Counter(optimizers)

plt.figure(figsize=(10, 6))
plt.pie(
    counts.values(),
    autopct="%1.1f%%"
)
plt.legend(
    counts.keys(),
    title="Optimizers",
    loc="best"
)
plt.title("(NQS-Bench-101): Optimizer distribution of all architectures")
plt.savefig("dataset_statistics/optimizer_piechart.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(counts.keys(), counts.values())
plt.title("(NQS-Bench-101): Optimizers of all architectures—histogram")
plt.savefig("dataset_statistics/optimizer_histogram.png")
plt.close()

counts = Counter(activations)

plt.figure(figsize=(10, 6))
plt.pie(
    counts.values(),
    autopct="%1.1f%%"
)
plt.legend(
    counts.keys(),
    title="Activations",
    loc="best"
)
plt.title("(NQS-Bench-101): Activation function distribution of all architectures")
plt.savefig("dataset_statistics/activation_function_piechart.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(counts.keys(), counts.values())
plt.title("(NQS-Bench-101): Activations of all architectures—histogram")
plt.savefig("dataset_statistics/activation_function_histogram.png")
plt.close()

counts = Counter(regimes)
counts['h=10⁻⁶'] = counts['h=10⁻⁶'] + counts['h=1.0⁻⁶']
del counts['h=1.0⁻⁶']
del counts['True']

print("Regime counts: ", counts)

plt.figure(figsize=(10, 6))
plt.pie(
    counts.values(),
    autopct="%1.1f%%"
)
plt.legend(
    counts.keys(),
    title="Regimes",
    loc="best"
)
plt.title("(NQS-Bench-101): Regime distribution of all architectures")
plt.savefig("dataset_statistics/regime_piechart.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(counts.keys(), counts.values())
plt.title("(NQS-Bench-101): Regimes of all architectures—histogram")
plt.savefig("dataset_statistics/regime_histogram.png")
plt.close()

fig_time = plt.figure(figsize =(10, 7))
fig_infid = plt.figure(figsize =(10, 7))
fig_hell_test = plt.figure(figsize =(10, 7))


print(type(train_time))
print(train_time.shape)
print(train_time.dtype)
print(train_time[:10])
print(np.isnan(train_time).sum())
print("Train time: ", type(train_time))

plt.figure(figsize=(10, 6))
plt.boxplot(train_time)
plt.title("(NQS-Bench-101): Training time (s) of all architectures—boxplots")
plt.savefig("dataset_statistics/training_time_boxplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(counts.keys(), counts.values())
plt.title("(NQS-Bench-101): Training time (s) of all architectures—histogram")
plt.savefig("dataset_statistics/training_time_histogram.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.violinplot(train_time)
plt.title("(NQS-Bench-101): Training time (s) of all architectures—violin plots")
plt.savefig("dataset_statistics/training_time_violinplot.png")
plt.close()


plt.figure(figsize=(10, 6))
plt.boxplot(infid)
plt.title("(NQS-Bench-101): Infidelity of all architectures—boxplots")
plt.savefig("dataset_statistics/infidelity_boxplot.png")
plt.close()


plt.figure(figsize=(10, 6))
plt.violinplot(infid)
plt.title("(NQS-Bench-101): Infidelity of all architectures—violin plots")
plt.savefig("dataset_statistics/infidelity_violinplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.boxplot(hell_test)
plt.title("(NQS-Bench-101): Hellinger distance (test) of all architectures—boxplots")
plt.savefig("dataset_statistics/hellinger_test_boxplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.violinplot(hell_test)
plt.title("(NQS-Bench-101): Hellinger distance (test) of \nall architectures—violin plots (h=2.0)")
plt.savefig("dataset_statistics/hellinger_test_violinplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.boxplot(test_loss)
plt.title("(NQS-Bench-101): Test loss of all architectures—boxplots")
plt.savefig("dataset_statistics/test_loss_boxplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.violinplot(test_loss)
plt.title("(NQS-Bench-101): Test loss of all architectures—violin plots")
plt.savefig("dataset_statistics/test_loss_violinplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.boxplot(r2_test)
plt.title("(NQS-Bench-101): R2 (test) of all architectures—boxplots")
plt.savefig("dataset_statistics/r2_test_boxplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.violinplot(r2_test)
plt.title("(NQS-Bench-101): R2 (test) of all architectures—violin plots")
plt.savefig("dataset_statistics/r2_test_violinplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.boxplot(mse_test)
plt.title("(NQS-Bench-101): MSE (test) of all architectures—boxplots")
plt.savefig("dataset_statistics/mse_test_boxplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.violinplot(mse_test)
plt.title("(NQS-Bench-101): MSE (test) of all architectures—violin plots")
plt.savefig("dataset_statistics/mse_test_violinplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.boxplot(mae_test)
plt.title("(NQS-Bench-101): MAE (test) of all architectures—boxplots")
plt.savefig("dataset_statistics/mae_test_boxplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.violinplot(mae_test)
plt.title("(NQS-Bench-101): MAE (test) of all architectures—violin plots")
plt.savefig("dataset_statistics/mae_test_violinplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.boxplot(params)
plt.title("(NQS-Bench-101): Number of parameters of all architectures—boxplots")
plt.savefig("dataset_statistics/params_boxplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.violinplot(params)
plt.title("(NQS-Bench-101): Number of parameters of all architectures—violin plots")
plt.savefig("dataset_statistics/params_violinplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.boxplot(epochs)
plt.title("(NQS-Bench-101): Number of epochs of all architectures—boxplots")
plt.savefig("dataset_statistics/epochs_boxplot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.violinplot(epochs)
plt.title("(NQS-Bench-101): Number of epochs of all architectures—violin plots")
plt.savefig("dataset_statistics/epochs_violinplot.png")
plt.close()

print(df_stats)

df_stats.to_csv(csv_file_path, index=True)



