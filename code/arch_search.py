import pandas as pd
x = pd.read_csv('NQS-Bench-101\\all_architectures_metrics\\all_architectures_metrics.csv')
h_0_5 = x[x['regime']=='h=0.5'].sort_values(by='R2_test', ascending=False)
print(h_0_5)