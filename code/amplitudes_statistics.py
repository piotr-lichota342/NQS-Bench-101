import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = 'dataset_statistics/amplitudes_statistics.csv'

DECIMALS = 4

csv_filenames_dict = {
    'h=0.5': 'datasets/1d_tfim_N12_h0.5_full_dataset.csv',
    'h=1.0': 'datasets/1d_tfim_N12_h1.0_full_dataset.csv',
    'h=10⁻⁶': 'datasets/1d_tfim_N12_h1.0e-6_full_dataset.csv',
    'h=2.0': 'datasets/1d_tfim_N12_h2.0_full_dataset.csv'
}

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
    index=stats_list,
    columns=csv_filenames_dict.keys()
)

for regime, filepath in csv_filenames_dict.items():

    df = pd.read_csv(filepath)
    amplitude = df['probability']

    df_stats.loc['count', regime] = round(amplitude.count(), DECIMALS)

    df_stats.loc['sum', regime] = round(amplitude.sum(), DECIMALS)
    

    df_stats.loc['mean', regime] = round(amplitude.mean(), DECIMALS)
    df_stats.loc['median', regime] = round(amplitude.median(), DECIMALS)
    df_stats.loc['mode', regime] = round(amplitude.mode().iloc[0], DECIMALS)
    df_stats.loc['min', regime] = round(amplitude.min(), DECIMALS)
    df_stats.loc['max', regime] = round(amplitude.max(), DECIMALS)
    df_stats.loc['std', regime] = round(amplitude.std(), DECIMALS)
    df_stats.loc['var', regime] = round(amplitude.var(), DECIMALS)
    df_stats.loc['skew', regime] = round(amplitude.skew(), DECIMALS)
    df_stats.loc['kurt', regime] = round(amplitude.kurt(), DECIMALS)
    df_stats.loc['25th_quantile', regime] = round(amplitude.quantile(0.25), DECIMALS)
    df_stats.loc['50th_quantile', regime] = round(amplitude.quantile(0.50), DECIMALS)
    df_stats.loc['75th_quantile', regime] = round(amplitude.quantile(0.75), DECIMALS)

    plt.figure(figsize=(10, 6))
    plt.violinplot(amplitude)
    plt.title("(NQS-Bench-101): Training time (s) of all architectures—violin plots")
    plt.savefig(f"dataset_statistics/amplitudes_violinplot_{regime}.png")
    plt.close()

print(df_stats)

df_stats.to_csv(csv_file_path)

