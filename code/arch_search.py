import pandas as pd
x = pd.read_csv('all_architectures_metrics/all_architectures_metrics.csv')
print("The shape of the global csv is: ", x.shape)
#x = x.dropna(subset=['infidelity'])
#h_0_5 = x[x['regime']=='h=0.5'].sort_values(by='R2_test', ascending=False)
#print(h_0_5)

best_records = pd.DataFrame()

h_0_5_best = x[x['regime']=='h=0.5'].sort_values(by='MAE_test', ascending=True).head(1)
best_records = pd.concat([best_records, h_0_5_best])
print(h_0_5_best)



h_1_0_best = x[x['regime']=='h=1.0'].sort_values(by='MAE_test', ascending=True).head(1)
best_records = pd.concat([best_records, h_1_0_best])
print(h_1_0_best)

h_2_0_best = x[x['regime']=='h=2.0'].sort_values(by='MAE_test', ascending=True).head(1)
best_records = pd.concat([best_records, h_2_0_best])
print(h_2_0_best)

h_1_0e6_best = x[x['regime']=='h=1.0⁻⁶'].sort_values(by='MAE_test', ascending=True).head(1)
best_records = pd.concat([best_records, h_1_0e6_best])
print(h_1_0e6_best)

best_records.to_csv('all_architectures_metrics/best_architectures_metrics.csv', index=False)

large_params = x[x['train_params']>1e6]
large_params.to_csv('all_architectures_metrics/large_parameter_architectures.csv', index=False)
