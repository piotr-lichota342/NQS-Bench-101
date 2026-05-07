import matplotlib.pyplot as plt

from main import train_losses_h1_0e6, train_losses_h0_5, train_losses_h1_0, train_losses_h2_0
from main import model_h0_5, model_h1_0, model_h2_0, model_h1_0e6
from dataset_loading import train_dataloader_h0_5, train_dataloader_h1_0, train_dataloader_h2_0, train_dataloader_h1_0e6
from dataset_loading import test_dataloader_h0_5, test_dataloader_h1_0, test_dataloader_h2_0, test_dataloader_h1_0e6
from dataset_loading import valid_dataloader_h0_5, valid_dataloader_h1_0, valid_dataloader_h2_0, valid_dataloader_h1_0e6

from main import valid_losses_h1_0e6, valid_losses_h0_5, valid_losses_h1_0, valid_losses_h2_0
from main import loss_fn, y_true_h1_0e6, y_pred_h1_0e6, y_true_h0_5, y_pred_h0_5, y_true_h1_0, y_pred_h1_0, y_true_h2_0, y_pred_h2_0, optimizer_h0_5, optimizer_h1_0, optimizer_h1_0e6, optimizer_h2_0, total_training_time
from config import trained_regimes, EPOCHS, BATCH_SIZE, W, TEST_PROPORTION, TRAIN_PROPORTION, VALID_PROPORTION, HIDDEN_LAYERS, INPUT_SIZE, device, trained_regimes, DECIMAL_PLACES_METRICS, SAVING_WEIGHTS
from torchinfo import summary
from test import test
from valid import valid
from train import train
import plotly.graph_objects as go
import torch
import math
import os
import itertools
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_log_error, mean_squared_log_error
import pandas as pd

from datetime import datetime


"""
Each training log consists of (for each trained regime):
- R squared
- test loss
- training time
- valid loss
- learning curve
- epochs
- optimizer
- batch size
- which architecture was trained
- loss function(s)
- hyperparameters settings
- when it was trained (timestamp)
- train/test/valid proportion
- number of trainable parameters

"""



def smape(a, f):
    a_np, f_np = np.array(a), np.array(f)
    return 1/len(a) * np.sum(2 * np.abs(f_np-a_np) / (np.abs(a_np) + np.abs(f_np))*100)

def hellinger_distance(p,q):
    print("Hellinger distance")
    #Turning into probabilities
    p, q = np.array(p), np.array(q)
    #p_prob, q_prob = [np.abs(a) for a in p], [np.abs(a) for a in q]
    p, q = np.abs(p), np.abs(q)
    #p_prob, q_prob = [a/np.sum(p_prob) for a in p_prob], [a/np.sum(q_prob) for a in q_prob]
    p = p / np.sum(p)
    q = q / np.sum(q)
    print(f"p sum: {np.sum(p)}, q sum: {np.sum(q)}")
    
    
    #print(f"p_prob, q prob: {p_prob}, {q_prob}")

    #final_result = 0.0   
    #for i in range(len(p_prob)):
    diff = ((p)**(0.5) - (q)**(0.5))**2
    print(f"sum diff: {np.sum(diff)}")
    diff = np.sum(diff)
    diff = diff**(0.5)
    print(f"diff: {diff}")
    diff = (1/(2**(0.5))) * diff
    print(f"diff: {diff}")
    
    final_result = diff
   
    #final_result = round(final_result,DECIMAL_PLACES_METRICS)
    
    return final_result

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(os.path.join('NQS-Bench-101\\training_logs', f"{timestamp}"))
os.makedirs(os.path.join(f'NQS-Bench-101\\training_logs\\{timestamp}', "curves"))
os.makedirs(os.path.join(f'NQS-Bench-101\\training_logs\\{timestamp}', "evaluation_metrics"))
os.makedirs(os.path.join(f'NQS-Bench-101\\training_logs\\{timestamp}', "model_weights"))

save_path_loss_curve = f"NQS-Bench-101\\training_logs\\{timestamp}\\curves\\loss_curve.png"
save_path_pred_true = f"NQS-Bench-101\\training_logs\\{timestamp}\\curves\\pred_true_curve.png"

save_path_loss_curve_html = f"NQS-Bench-101\\training_logs\\{timestamp}\\curves\\loss_curve.html"
save_path_pred_true_html = f"NQS-Bench-101\\training_logs\\{timestamp}\\curves\\pred_true_curve.html"

model_weights_h0_5_path = f"NQS-Bench-101\\training_logs\\{timestamp}\\model_weights\\model_weights_h0_5.pth"
model_weights_h1_0_path = f"NQS-Bench-101\\training_logs\\{timestamp}\\model_weights\\model_weights_h1_0.pth"
model_weights_h2_0_path = f"NQS-Bench-101\\training_logs\\{timestamp}\\model_weights\\model_weights_h2_0.pth"
model_weights_h1_0e6_path = f"NQS-Bench-101\\training_logs\\{timestamp}\\model_weights\\model_weights_h1_0e6.pth"

csv_file_path = f'NQS-Bench-101\\training_logs\\{timestamp}\\evaluation_metrics\\metrics.csv'

'''
r2 = round(10⁻⁶(y_pred_h0_5, y_true_h0_5),4)
mse = round(mean_squared_error(y_true_h0_5, y_pred_h0_5),4)
mae = round(mean_absolute_error(y_true_h0_5, y_pred_h0_5),4)    
var = round(np.var(y_pred_h0_5),4)
rmse = round(root_mean_squared_error(y_true_h0_5, y_pred_h0_5),4)

mape = round(mean_absolute_percentage_error(y_true_h0_5, y_pred_h0_5),4)
smape = round(smape(y_true_h0_5, y_pred_h0_5),4)
rmse = round(root_mean_squared_error(y_true_h0_5, y_pred_h0_5),4)
rmsle = round(root_mean_squared_log_error(y_true_h0_5, y_pred_h0_5),4)
msle = round(mean_squared_log_error(y_true_h0_5, y_pred_h0_5),4)
hell_dist = hellinger_distance(y_pred_h0_5, y_true_h0_5)

print(f"R square (h=0.5): {r2}")
print(f"Variance (h=0.5): {var}")
print(f"Root mean squared error (h=0.5): {rmse}")
print(f"Mean squared error (h=0.5): {mse}")
print(f"Mean absolute error: {mae}")
print(f"Mean absolute percentage error: {mape}")
#print(f"Symmetric mean absolute percentage error (h=0.5): {smape}")
print(f"Root mean squared error (h=0.5): {rmse}")
print(f"Root mean squared log error (h=0.5): {rmsle}")
print(f"Mean squared log error (h=0.5): {msle}")
'''



metrics_data = {
    'regime': None,
    'total_params':None,
    'train_params': None,
    'non_train_params': None,
    #'model_summary': None, # summary_str
    'test_loss':None,
    'train_loss':None,
    'valid_loss':None,
    'train_time (s)': total_training_time,
    'epochs': EPOCHS,
    'input_size': INPUT_SIZE,
    'batch_size': BATCH_SIZE,
    'network_width': W,
    'hidden_layers': HIDDEN_LAYERS,
    'train_proportion': TRAIN_PROPORTION,
    'test_proportion': TEST_PROPORTION,
    'valid_proportion': VALID_PROPORTION,
    'device': str(device),
    'optimizer_name': None, # optimizer_h0_5.__class__.__name__
    'optimizer_params': None, # str(dict_optimizer)
    'loss_fn': str(loss_fn.__class__.__name__),
    #'bias': None,
    #'avr_res': None,
    #'MBE': None,
    'R2_test': None,
    'R2_train': None,
    'R2_valid': None,
    #'RSS': None,
    #'TSS': None,
    #'adjusted_R2': None,
    'MSE_test': None,
    'MSE_train': None,
    'MSE_valid': None,
    #'RMSE': None,
    'MAE_test': None,
    'MAE_train': None,
    'MAE_valid': None,
    #'MAPE': None,
    #'wMAPE': None,
    #'sMAPE': None,
    #'MSLE': None,
    #'RMSLE': None,
    #'AIC': None,
    #'BIC': None,
    #'ESS': None,
    'hellinger_dist_test': None,
    'hellinger_dist_train': None,
    'hellinger_dist_valid': None
    
}


#print(f"Train loss length: {train_losses}")
all_values = []
df_metrics_all = pd.DataFrame()

f1 = plt.figure(edgecolor='black')
f2 = plt.figure()
ax1 = f1.add_subplot(1,1,1)
ax2 = f2.add_subplot(1,1,1)
ax1.set_xlabel(r"True $\log\psi_\omega(\vec{\sigma})$")
ax1.set_ylabel(r"Predicted $\log\psi_\omega(\vec{\sigma})$")

#f1.set_edgecolor("black")


ax1.grid(True)
ax2.set_xlabel("Epoch")
ax2.set_ylabel(f"{loss_fn.__class__.__name__} (log scale)")
ax2.set_title("(NQS-Bench-101): Training vs Validation Loss")

ax2.grid(True)
ax2.set_yscale("log")

graph = go.Figure()
graph2 = go.Figure()

graph.update_layout(
        xaxis_title="True " + "logψ_ω(vec{σ})", 
        yaxis_title="Predicted " + "logψ_ω(vec{σ})", 
        title="(NQS-Bench-101): True vs. Predicted " + "logψ_ω(vec{σ})", 
        showlegend=True,
        legend_title_text="Legend"
    )

graph2.update_layout(
        xaxis_title="Epochs",
        yaxis_title=f"{loss_fn.__class__.__name__} (log scale)",
        title='(NQS-Bench-101): Train vs Valid Loss',
        showlegend=True,
        legend_title_text="Legend"
    )




if trained_regimes["h=10⁻⁶"]:
    
    avg_test_loss_h1_0e6, target_test_h1_0e6, pred_test_h1_0e6 = test(test_dataloader_h1_0e6, model_h1_0e6, loss_fn)
    avg_train_loss_h1_0e6, target_train_h1_0e6, pred_train_h1_0e6 = train(train_dataloader_h1_0e6, model_h1_0e6, loss_fn, optimizer_h1_0e6)
    avg_valid_loss_h1_0e6, target_valid_h1_0e6, pred_valid_h1_0e6 = valid(valid_dataloader_h1_0e6, model_h1_0e6, loss_fn)
    #perfect_prediction_x = [0,max(max(y_true_h1_0e6), max(y_pred_h1_0e6))]
    dict_optimizer_h1_0e6 = optimizer_h1_0e6.param_groups[0]
    dict_optimizer_h1_0e6.pop('params')
    
    total_params = sum(p.numel() for p in model_h1_0e6.parameters())
    train_params = sum(p.numel() for p in model_h1_0e6.parameters() if p.requires_grad)
    

    
    
    #ax1 = f1.add_axes(train_losses_h1_0e6)
    df_metrics_h1_0e6 = metrics_data.copy()
    df_metrics_h1_0e6['regime'] = "h=10⁻⁶"
    df_metrics_h1_0e6['test_loss'] = round(avg_test_loss_h1_0e6,DECIMAL_PLACES_METRICS)
    df_metrics_h1_0e6['train_loss'] = round(avg_train_loss_h1_0e6,DECIMAL_PLACES_METRICS)
    if avg_valid_loss_h1_0e6==np.nan or len(target_valid_h1_0e6)==0 or len(pred_valid_h1_0e6)==0:
        df_metrics_h1_0e6['valid_loss'] = 'NaN'
        df_metrics_h1_0e6['R2_valid'] = 'NaN'
        df_metrics_h1_0e6["MSE_valid"] = 'NaN'
        df_metrics_h1_0e6["MAE_valid"] = 'NaN'
        df_metrics_h1_0e6["hellinger_dist_valid"] = 'NaN'
    else:
        df_metrics_h1_0e6['valid_loss'] = round(avg_valid_loss_h1_0e6,DECIMAL_PLACES_METRICS) 
        df_metrics_h1_0e6['R2_valid'] = round(r2_score(target_valid_h1_0e6, pred_valid_h1_0e6),DECIMAL_PLACES_METRICS)
        df_metrics_h1_0e6["MSE_valid"] = round(mean_squared_error(target_valid_h1_0e6, pred_valid_h1_0e6),DECIMAL_PLACES_METRICS)
        df_metrics_h1_0e6["MAE_valid"] = round(mean_absolute_error(target_valid_h1_0e6, pred_valid_h1_0e6),DECIMAL_PLACES_METRICS)
        df_metrics_h1_0e6["hellinger_dist_valid"] = np.round(hellinger_distance(target_valid_h1_0e6, pred_valid_h1_0e6),decimals=DECIMAL_PLACES_METRICS)
    #df_metrics_h1_0e6['model_summary'] = str(summary(model_h1_0e6, INPUT_SIZE))
    df_metrics_h1_0e6['total_params'] = total_params
    df_metrics_h1_0e6['train_params'] = train_params
    df_metrics_h1_0e6['non_train_params'] = total_params - train_params
    df_metrics_h1_0e6['optimizer_name'] = optimizer_h1_0e6.__class__.__name__
    df_metrics_h1_0e6['optimizer_params'] = str(dict_optimizer_h1_0e6)
    
    df_metrics_h1_0e6['R2_test'] = round(r2_score(target_test_h1_0e6, pred_test_h1_0e6),DECIMAL_PLACES_METRICS)
    df_metrics_h1_0e6['R2_train'] = round(r2_score(target_train_h1_0e6, pred_train_h1_0e6),DECIMAL_PLACES_METRICS) 
    
    df_metrics_h1_0e6["MSE_test"] = round(mean_squared_error(target_test_h1_0e6, pred_test_h1_0e6),DECIMAL_PLACES_METRICS)
    df_metrics_h1_0e6["MSE_train"] = round(mean_squared_error(target_train_h1_0e6, pred_train_h1_0e6),DECIMAL_PLACES_METRICS)
    

    df_metrics_h1_0e6["MAE_test"] = round(mean_absolute_error(target_test_h1_0e6, pred_test_h1_0e6),DECIMAL_PLACES_METRICS)
    df_metrics_h1_0e6["MAE_train"] = round(mean_absolute_error(target_train_h1_0e6, pred_train_h1_0e6),DECIMAL_PLACES_METRICS)

    
    df_metrics_h1_0e6["hellinger_dist_test"] = np.round(hellinger_distance(target_test_h1_0e6, pred_test_h1_0e6),decimals=DECIMAL_PLACES_METRICS)
    df_metrics_h1_0e6["hellinger_dist_train"] = np.round(hellinger_distance(target_train_h1_0e6, pred_train_h1_0e6),decimals=DECIMAL_PLACES_METRICS)
        
    
    #df_metrics_h1_0e6["RMSE"] = round(root_mean_squared_error(y_true_h1_0e6, y_pred_h1_0e6),DECIMAL_PLACES_METRICS)
    #df_metrics_h1_0e6["MAPE"] = round(mean_absolute_percentage_error(y_true_h1_0e6, y_pred_h1_0e6),DECIMAL_PLACES_METRICS)
    #df_metrics_h1_0e6["sMAPE"] = round(smape(y_true_h1_0e6, y_pred_h1_0e6),DECIMAL_PLACES_METRICS)
    
    #df_metrics_h1_0e6["RMSLE"] = round(root_mean_squared_log_error(y_true_h1_0e6, y_pred_h1_0e6),DECIMAL_PLACES_METRICS)
    #df_metrics_h1_0e6["MSLE"] = round(mean_squared_log_error(y_true_h1_0e6, y_pred_h1_0e6),DECIMAL_PLACES_METRICS)
    
    #df_metrics_h1_0e6["hellinger_dist"] = round(hellinger_distance(y_true_h1_0e6, y_pred_h1_0e6),DECIMAL_PLACES_METRICS)
    '''
    var = round(np.var(y_pred_h1_0e6),4)
    rmse = round(root_mean_squared_error(y_true_h1_0e6, y_pred_h1_0e6),4)

    mape = round(mean_absolute_percentage_error(y_true_h1_0e6, y_pred_h1_0e6),4)
    smape = round(smape(y_true_h1_0e6, y_pred_h1_0e6),4)
    rmse = round(root_mean_squared_error(y_true_h1_0e6, y_pred_h1_0e6),4)
    rmsle = round(root_mean_squared_log_error(y_true_h1_0e6, y_pred_h1_0e6),4)
    msle = round(mean_squared_log_error(y_true_h1_0e6, y_pred_h1_0e6),4)
    hell_dist = hellinger_distance(y_pred_h1_0e6, y_true_h1_0e6)
        
    '''
    plot_metrics = "\n"+r"$R^2$"+f"={round(r2_score(target_test_h1_0e6, pred_test_h1_0e6),DECIMAL_PLACES_METRICS)}, MSE={round(mean_squared_error(target_test_h1_0e6, pred_test_h1_0e6),DECIMAL_PLACES_METRICS)}, MAE={round(mean_absolute_error(target_test_h1_0e6, pred_test_h1_0e6),DECIMAL_PLACES_METRICS)}"
    plot_title = r"(NQS-Bench-101): True vs. Predicted $\log\psi_\omega(\vec{\sigma})$"
    ax1.set_title(plot_title + plot_metrics)
    ax2.plot(train_losses_h1_0e6, label=r"Train loss ($h=10^{-6}$)")
    ax2.plot(valid_losses_h1_0e6, label=r"Valid loss ($h=10^{-6}$)")
    ax1.plot(target_test_h1_0e6, pred_test_h1_0e6, 'o', markersize=5, label=r"$h=10^{-6}$", alpha=0.5, mec='black')
    
    #ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
    #graph = go.Figure()
    graph.add_trace(go.Scatter(x=target_test_h1_0e6, y=pred_test_h1_0e6, mode='markers', name=r"$h=10^{-6}$", opacity=0.5))
    
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=valid_losses_h1_0e6, mode='lines', name=r"Valid loss ($h=10^{-6}$)"))
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=train_losses_h1_0e6, mode='lines', name=r"Train loss ($h=10^{-6}$)"))
    
    
    torch.save(model_h1_0e6.state_dict(), model_weights_h1_0e6_path) if SAVING_WEIGHTS else None
    df_metrics_all = pd.concat([df_metrics_all, pd.DataFrame([df_metrics_h1_0e6])], ignore_index=True)
    
    all_values.append(list(target_test_h1_0e6) + 
        list(pred_test_h1_0e6))
    

if trained_regimes["h=0.5"]:
    avg_test_loss_h0_5, target_test_h0_5, pred_test_h0_5 = test(test_dataloader_h0_5, model_h0_5, loss_fn)
    avg_train_loss_h0_5, target_train_h0_5, pred_train_h0_5 = train(train_dataloader_h0_5, model_h0_5, loss_fn, optimizer_h0_5)
    avg_valid_loss_h0_5, target_valid_h0_5, pred_valid_h0_5 = valid(valid_dataloader_h0_5, model_h0_5, loss_fn)
    #perfect_prediction_x = [0,max(max(y_true_h0_5), max(y_pred_h0_5))]
    dict_optimizer_h0_5 = optimizer_h0_5.param_groups[0]
    dict_optimizer_h0_5.pop('params')
    
    total_params = sum(p.numel() for p in model_h0_5.parameters())
    train_params = sum(p.numel() for p in model_h0_5.parameters() if p.requires_grad)
    

    
    
    #ax1 = f1.add_axes(train_losses_h0_5)
    df_metrics_h0_5 = metrics_data.copy()
    df_metrics_h0_5['regime'] = "h=0.5"
    df_metrics_h0_5['test_loss'] = round(avg_test_loss_h0_5,DECIMAL_PLACES_METRICS)
    df_metrics_h0_5['train_loss'] = round(avg_train_loss_h0_5,DECIMAL_PLACES_METRICS)
    if avg_valid_loss_h0_5==np.nan or len(target_valid_h0_5)==0 or len(pred_valid_h0_5)==0:
        df_metrics_h0_5['valid_loss'] = 'NaN'
        df_metrics_h0_5['R2_valid'] = 'NaN'
        df_metrics_h0_5["MSE_valid"] = 'NaN'
        df_metrics_h0_5["MAE_valid"] = 'NaN'
        df_metrics_h0_5["hellinger_dist_valid"] = 'NaN'
    else:
        df_metrics_h0_5['valid_loss'] = round(avg_valid_loss_h0_5,DECIMAL_PLACES_METRICS) 
        df_metrics_h0_5['R2_valid'] = round(r2_score(target_valid_h0_5, pred_valid_h0_5),DECIMAL_PLACES_METRICS)
        df_metrics_h0_5["MSE_valid"] = round(mean_squared_error(target_valid_h0_5, pred_valid_h0_5),DECIMAL_PLACES_METRICS)
        df_metrics_h0_5["MAE_valid"] = round(mean_absolute_error(target_valid_h0_5, pred_valid_h0_5),DECIMAL_PLACES_METRICS)
        df_metrics_h0_5["hellinger_dist_valid"] = np.round(hellinger_distance(target_valid_h0_5, pred_valid_h0_5),decimals=DECIMAL_PLACES_METRICS)
    #df_metrics_h0_5['model_summary'] = str(summary(model_h0_5, INPUT_SIZE))
    df_metrics_h0_5['total_params'] = total_params
    df_metrics_h0_5['train_params'] = train_params
    df_metrics_h0_5['non_train_params'] = total_params - train_params
    df_metrics_h0_5['optimizer_name'] = optimizer_h0_5.__class__.__name__
    df_metrics_h0_5['optimizer_params'] = str(dict_optimizer_h0_5)
    
    df_metrics_h0_5['R2_test'] = round(r2_score(target_test_h0_5, pred_test_h0_5),DECIMAL_PLACES_METRICS)
    df_metrics_h0_5['R2_train'] = round(r2_score(target_train_h0_5, pred_train_h0_5),DECIMAL_PLACES_METRICS) 
    
    df_metrics_h0_5["MSE_test"] = round(mean_squared_error(target_test_h0_5, pred_test_h0_5),DECIMAL_PLACES_METRICS)
    df_metrics_h0_5["MSE_train"] = round(mean_squared_error(target_train_h0_5, pred_train_h0_5),DECIMAL_PLACES_METRICS)
    

    df_metrics_h0_5["MAE_test"] = round(mean_absolute_error(target_test_h0_5, pred_test_h0_5),DECIMAL_PLACES_METRICS)
    df_metrics_h0_5["MAE_train"] = round(mean_absolute_error(target_train_h0_5, pred_train_h0_5),DECIMAL_PLACES_METRICS)

    
    df_metrics_h0_5["hellinger_dist_test"] = np.round(hellinger_distance(target_test_h0_5, pred_test_h0_5),decimals=DECIMAL_PLACES_METRICS)
    df_metrics_h0_5["hellinger_dist_train"] = np.round(hellinger_distance(target_train_h0_5, pred_train_h0_5),decimals=DECIMAL_PLACES_METRICS)
        
    
    #df_metrics_h0_5["RMSE"] = round(root_mean_squared_error(y_true_h0_5, y_pred_h0_5),DECIMAL_PLACES_METRICS)
    #df_metrics_h0_5["MAPE"] = round(mean_absolute_percentage_error(y_true_h0_5, y_pred_h0_5),DECIMAL_PLACES_METRICS)
    #df_metrics_h0_5["sMAPE"] = round(smape(y_true_h0_5, y_pred_h0_5),DECIMAL_PLACES_METRICS)
    
    #df_metrics_h0_5["RMSLE"] = round(root_mean_squared_log_error(y_true_h0_5, y_pred_h0_5),DECIMAL_PLACES_METRICS)
    #df_metrics_h0_5["MSLE"] = round(mean_squared_log_error(y_true_h0_5, y_pred_h0_5),DECIMAL_PLACES_METRICS)
    
    #df_metrics_h0_5["hellinger_dist"] = round(hellinger_distance(y_true_h0_5, y_pred_h0_5),DECIMAL_PLACES_METRICS)
    '''
    var = round(np.var(y_pred_h0_5),4)
    rmse = round(root_mean_squared_error(y_true_h0_5, y_pred_h0_5),4)

    mape = round(mean_absolute_percentage_error(y_true_h0_5, y_pred_h0_5),4)
    smape = round(smape(y_true_h0_5, y_pred_h0_5),4)
    rmse = round(root_mean_squared_error(y_true_h0_5, y_pred_h0_5),4)
    rmsle = round(root_mean_squared_log_error(y_true_h0_5, y_pred_h0_5),4)
    msle = round(mean_squared_log_error(y_true_h0_5, y_pred_h0_5),4)
    hell_dist = hellinger_distance(y_pred_h0_5, y_true_h0_5)
        
    '''
    plot_metrics = "\n"+r"$R^2$"+f"={round(r2_score(target_test_h0_5, pred_test_h0_5),DECIMAL_PLACES_METRICS)}, MSE={round(mean_squared_error(target_test_h0_5, pred_test_h0_5),DECIMAL_PLACES_METRICS)}, MAE={round(mean_absolute_error(target_test_h0_5, pred_test_h0_5),DECIMAL_PLACES_METRICS)}"
    plot_title = r"(NQS-Bench-101): True vs. Predicted $\log\psi_\omega(\vec{\sigma})$"
    ax1.set_title(plot_title + plot_metrics)
    ax2.plot(train_losses_h0_5, label="Train loss (h=0.5)")
    ax2.plot(valid_losses_h0_5, label="Valid loss (h=0.5)")
    ax1.plot(target_test_h0_5, pred_test_h0_5, 'o', markersize=5, label="h=0.5", alpha=0.5, mec='black')
    
    #ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
    #graph = go.Figure()
    graph.add_trace(go.Scatter(x=target_test_h0_5, y=pred_test_h0_5, mode='markers', name="h=0.5", opacity=0.5))
    
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=valid_losses_h0_5, mode='lines', name="Valid loss (h=0.5)"))
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=train_losses_h0_5, mode='lines', name="Train loss (h=0.5)"))
    
    
    torch.save(model_h0_5.state_dict(), model_weights_h0_5_path) if SAVING_WEIGHTS else None
    df_metrics_all = pd.concat([df_metrics_all, pd.DataFrame([df_metrics_h0_5])], ignore_index=True)
    
    all_values.append(list(target_test_h0_5) + 
        list(pred_test_h0_5))
    
    
    
    
    #f1.savefig(save_path_pred_true, dpi=300, bbox_inches="tight")
    
    
    
if trained_regimes["h=1.0"]:
    avg_test_loss_h1_0, target_test_h1_0, pred_test_h1_0 = test(test_dataloader_h1_0, model_h1_0, loss_fn)
    avg_train_loss_h1_0, target_train_h1_0, pred_train_h1_0 = train(train_dataloader_h1_0, model_h1_0, loss_fn, optimizer_h1_0)
    avg_valid_loss_h1_0, target_valid_h1_0, pred_valid_h1_0 = valid(valid_dataloader_h1_0, model_h1_0, loss_fn)
    #perfect_prediction_x = [0,max(max(y_true_h1_0), max(y_pred_h1_0))]
    dict_optimizer_h1_0 = optimizer_h1_0.param_groups[0]
    dict_optimizer_h1_0.pop('params')
    
    total_params = sum(p.numel() for p in model_h1_0.parameters())
    train_params = sum(p.numel() for p in model_h1_0.parameters() if p.requires_grad)
    

    
    
    #ax1 = f1.add_axes(train_losses_h1_0)
    df_metrics_h1_0 = metrics_data.copy()
    df_metrics_h1_0['regime'] = "h=1.0"
    df_metrics_h1_0['test_loss'] = round(avg_test_loss_h1_0,DECIMAL_PLACES_METRICS)
    df_metrics_h1_0['train_loss'] = round(avg_train_loss_h1_0,DECIMAL_PLACES_METRICS)
    if avg_valid_loss_h1_0==np.nan or len(target_valid_h1_0)==0 or len(pred_valid_h1_0)==0:
        df_metrics_h1_0['valid_loss'] = 'NaN'
        df_metrics_h1_0['R2_valid'] = 'NaN'
        df_metrics_h1_0["MSE_valid"] = 'NaN'
        df_metrics_h1_0["MAE_valid"] = 'NaN'
        df_metrics_h1_0["hellinger_dist_valid"] = 'NaN'
    else:
        df_metrics_h1_0['valid_loss'] = round(avg_valid_loss_h1_0,DECIMAL_PLACES_METRICS) 
        df_metrics_h1_0['R2_valid'] = round(r2_score(target_valid_h1_0, pred_valid_h1_0),DECIMAL_PLACES_METRICS)
        df_metrics_h1_0["MSE_valid"] = round(mean_squared_error(target_valid_h1_0, pred_valid_h1_0),DECIMAL_PLACES_METRICS)
        df_metrics_h1_0["MAE_valid"] = round(mean_absolute_error(target_valid_h1_0, pred_valid_h1_0),DECIMAL_PLACES_METRICS)
        df_metrics_h1_0["hellinger_dist_valid"] = np.round(hellinger_distance(target_valid_h1_0, pred_valid_h1_0),decimals=DECIMAL_PLACES_METRICS)
    #df_metrics_h1_0['model_summary'] = str(summary(model_h1_0, INPUT_SIZE))
    df_metrics_h1_0['total_params'] = total_params
    df_metrics_h1_0['train_params'] = train_params
    df_metrics_h1_0['non_train_params'] = total_params - train_params
    df_metrics_h1_0['optimizer_name'] = optimizer_h1_0.__class__.__name__
    df_metrics_h1_0['optimizer_params'] = str(dict_optimizer_h1_0)
    
    df_metrics_h1_0['R2_test'] = round(r2_score(target_test_h1_0, pred_test_h1_0),DECIMAL_PLACES_METRICS)
    df_metrics_h1_0['R2_train'] = round(r2_score(target_train_h1_0, pred_train_h1_0),DECIMAL_PLACES_METRICS) 
    
    df_metrics_h1_0["MSE_test"] = round(mean_squared_error(target_test_h1_0, pred_test_h1_0),DECIMAL_PLACES_METRICS)
    df_metrics_h1_0["MSE_train"] = round(mean_squared_error(target_train_h1_0, pred_train_h1_0),DECIMAL_PLACES_METRICS)
    

    df_metrics_h1_0["MAE_test"] = round(mean_absolute_error(target_test_h1_0, pred_test_h1_0),DECIMAL_PLACES_METRICS)
    df_metrics_h1_0["MAE_train"] = round(mean_absolute_error(target_train_h1_0, pred_train_h1_0),DECIMAL_PLACES_METRICS)

    
    df_metrics_h1_0["hellinger_dist_test"] = np.round(hellinger_distance(target_test_h1_0, pred_test_h1_0),decimals=DECIMAL_PLACES_METRICS)
    df_metrics_h1_0["hellinger_dist_train"] = np.round(hellinger_distance(target_train_h1_0, pred_train_h1_0),decimals=DECIMAL_PLACES_METRICS)
        
    
    #df_metrics_h1_0["RMSE"] = round(root_mean_squared_error(y_true_h1_0, y_pred_h1_0),DECIMAL_PLACES_METRICS)
    #df_metrics_h1_0["MAPE"] = round(mean_absolute_percentage_error(y_true_h1_0, y_pred_h1_0),DECIMAL_PLACES_METRICS)
    #df_metrics_h1_0["sMAPE"] = round(smape(y_true_h1_0, y_pred_h1_0),DECIMAL_PLACES_METRICS)
    
    #df_metrics_h1_0["RMSLE"] = round(root_mean_squared_log_error(y_true_h1_0, y_pred_h1_0),DECIMAL_PLACES_METRICS)
    #df_metrics_h1_0["MSLE"] = round(mean_squared_log_error(y_true_h1_0, y_pred_h1_0),DECIMAL_PLACES_METRICS)
    
    #df_metrics_h1_0["hellinger_dist"] = round(hellinger_distance(y_true_h1_0, y_pred_h1_0),DECIMAL_PLACES_METRICS)
    '''
    var = round(np.var(y_pred_h1_0),4)
    rmse = round(root_mean_squared_error(y_true_h1_0, y_pred_h1_0),4)

    mape = round(mean_absolute_percentage_error(y_true_h1_0, y_pred_h1_0),4)
    smape = round(smape(y_true_h1_0, y_pred_h1_0),4)
    rmse = round(root_mean_squared_error(y_true_h1_0, y_pred_h1_0),4)
    rmsle = round(root_mean_squared_log_error(y_true_h1_0, y_pred_h1_0),4)
    msle = round(mean_squared_log_error(y_true_h1_0, y_pred_h1_0),4)
    hell_dist = hellinger_distance(y_pred_h1_0, y_true_h1_0)
        
    '''
    plot_metrics = "\n"+r"$R^2$"+f"={round(r2_score(target_test_h1_0, pred_test_h1_0),DECIMAL_PLACES_METRICS)}, MSE={round(mean_squared_error(target_test_h1_0, pred_test_h1_0),DECIMAL_PLACES_METRICS)}, MAE={round(mean_absolute_error(target_test_h1_0, pred_test_h1_0),DECIMAL_PLACES_METRICS)}"
    plot_title = r"(NQS-Bench-101): True vs. Predicted $\log\psi_\omega(\vec{\sigma})$"
    ax1.set_title(plot_title + plot_metrics)
    ax2.plot(train_losses_h1_0, label="Train loss (h=1.0)")
    ax2.plot(valid_losses_h1_0, label="Valid loss (h=1.0)")
    ax1.plot(target_test_h1_0, pred_test_h1_0, 'o', markersize=5, label="h=1.0", alpha=0.5, mec='black')
    
    #ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
    #graph = go.Figure()
    graph.add_trace(go.Scatter(x=target_test_h1_0, y=pred_test_h1_0, mode='markers', name="h=1.0", opacity=0.5))
    
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=valid_losses_h1_0, mode='lines', name="Valid loss (h=1.0)"))
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=train_losses_h1_0, mode='lines', name="Train loss (h=1.0)"))
    
    
    torch.save(model_h1_0.state_dict(), model_weights_h1_0_path) if SAVING_WEIGHTS else None
    df_metrics_all = pd.concat([df_metrics_all, pd.DataFrame([df_metrics_h1_0])], ignore_index=True)
    
    all_values.append(list(target_test_h1_0) + 
        list(pred_test_h1_0))
    
if trained_regimes["h=2.0"]:
    avg_test_loss_h2_0, target_test_h2_0, pred_test_h2_0 = test(test_dataloader_h2_0, model_h2_0, loss_fn)
    avg_train_loss_h2_0, target_train_h2_0, pred_train_h2_0 = train(train_dataloader_h2_0, model_h2_0, loss_fn, optimizer_h2_0)
    avg_valid_loss_h2_0, target_valid_h2_0, pred_valid_h2_0 = valid(valid_dataloader_h2_0, model_h2_0, loss_fn)
    #perfect_prediction_x = [0,max(max(y_true_h2_0), max(y_pred_h2_0))]
    dict_optimizer_h2_0 = optimizer_h2_0.param_groups[0]
    dict_optimizer_h2_0.pop('params')
    
    total_params = sum(p.numel() for p in model_h2_0.parameters())
    train_params = sum(p.numel() for p in model_h2_0.parameters() if p.requires_grad)
    

    
    
    #ax1 = f1.add_axes(train_losses_h2_0)
    df_metrics_h2_0 = metrics_data.copy()
    df_metrics_h2_0['regime'] = "h=2.0"
    df_metrics_h2_0['test_loss'] = round(avg_test_loss_h2_0,DECIMAL_PLACES_METRICS)
    df_metrics_h2_0['train_loss'] = round(avg_train_loss_h2_0,DECIMAL_PLACES_METRICS)
    if avg_valid_loss_h2_0==np.nan or len(target_valid_h2_0)==0 or len(pred_valid_h2_0)==0:
        df_metrics_h2_0['valid_loss'] = 'NaN'
        df_metrics_h2_0['R2_valid'] = 'NaN'
        df_metrics_h2_0["MSE_valid"] = 'NaN'
        df_metrics_h2_0["MAE_valid"] = 'NaN'
        df_metrics_h2_0["hellinger_dist_valid"] = 'NaN'
    else:
        df_metrics_h2_0['valid_loss'] = round(avg_valid_loss_h2_0,DECIMAL_PLACES_METRICS) 
        df_metrics_h2_0['R2_valid'] = round(r2_score(target_valid_h2_0, pred_valid_h2_0),DECIMAL_PLACES_METRICS)
        df_metrics_h2_0["MSE_valid"] = round(mean_squared_error(target_valid_h2_0, pred_valid_h2_0),DECIMAL_PLACES_METRICS)
        df_metrics_h2_0["MAE_valid"] = round(mean_absolute_error(target_valid_h2_0, pred_valid_h2_0),DECIMAL_PLACES_METRICS)
        df_metrics_h2_0["hellinger_dist_valid"] = np.round(hellinger_distance(target_valid_h2_0, pred_valid_h2_0),decimals=DECIMAL_PLACES_METRICS)
    #df_metrics_h2_0['model_summary'] = str(summary(model_h2_0, INPUT_SIZE))
    df_metrics_h2_0['total_params'] = total_params
    df_metrics_h2_0['train_params'] = train_params
    df_metrics_h2_0['non_train_params'] = total_params - train_params
    df_metrics_h2_0['optimizer_name'] = optimizer_h2_0.__class__.__name__
    df_metrics_h2_0['optimizer_params'] = str(dict_optimizer_h2_0)
    
    df_metrics_h2_0['R2_test'] = round(r2_score(target_test_h2_0, pred_test_h2_0),DECIMAL_PLACES_METRICS)
    df_metrics_h2_0['R2_train'] = round(r2_score(target_train_h2_0, pred_train_h2_0),DECIMAL_PLACES_METRICS) 
    
    df_metrics_h2_0["MSE_test"] = round(mean_squared_error(target_test_h2_0, pred_test_h2_0),DECIMAL_PLACES_METRICS)
    df_metrics_h2_0["MSE_train"] = round(mean_squared_error(target_train_h2_0, pred_train_h2_0),DECIMAL_PLACES_METRICS)
    

    df_metrics_h2_0["MAE_test"] = round(mean_absolute_error(target_test_h2_0, pred_test_h2_0),DECIMAL_PLACES_METRICS)
    df_metrics_h2_0["MAE_train"] = round(mean_absolute_error(target_train_h2_0, pred_train_h2_0),DECIMAL_PLACES_METRICS)

    
    df_metrics_h2_0["hellinger_dist_test"] = np.round(hellinger_distance(target_test_h2_0, pred_test_h2_0),decimals=DECIMAL_PLACES_METRICS)
    df_metrics_h2_0["hellinger_dist_train"] = np.round(hellinger_distance(target_train_h2_0, pred_train_h2_0),decimals=DECIMAL_PLACES_METRICS)
        
    
    #df_metrics_h2_0["RMSE"] = round(root_mean_squared_error(y_true_h2_0, y_pred_h2_0),DECIMAL_PLACES_METRICS)
    #df_metrics_h2_0["MAPE"] = round(mean_absolute_percentage_error(y_true_h2_0, y_pred_h2_0),DECIMAL_PLACES_METRICS)
    #df_metrics_h2_0["sMAPE"] = round(smape(y_true_h2_0, y_pred_h2_0),DECIMAL_PLACES_METRICS)
    
    #df_metrics_h2_0["RMSLE"] = round(root_mean_squared_log_error(y_true_h2_0, y_pred_h2_0),DECIMAL_PLACES_METRICS)
    #df_metrics_h2_0["MSLE"] = round(mean_squared_log_error(y_true_h2_0, y_pred_h2_0),DECIMAL_PLACES_METRICS)
    
    #df_metrics_h2_0["hellinger_dist"] = round(hellinger_distance(y_true_h2_0, y_pred_h2_0),DECIMAL_PLACES_METRICS)
    '''
    var = round(np.var(y_pred_h2_0),4)
    rmse = round(root_mean_squared_error(y_true_h2_0, y_pred_h2_0),4)

    mape = round(mean_absolute_percentage_error(y_true_h2_0, y_pred_h2_0),4)
    smape = round(smape(y_true_h2_0, y_pred_h2_0),4)
    rmse = round(root_mean_squared_error(y_true_h2_0, y_pred_h2_0),4)
    rmsle = round(root_mean_squared_log_error(y_true_h2_0, y_pred_h2_0),4)
    msle = round(mean_squared_log_error(y_true_h2_0, y_pred_h2_0),4)
    hell_dist = hellinger_distance(y_pred_h2_0, y_true_h2_0)
        
    '''
    plot_metrics = "\n"+r"$R^2$"+f"={round(r2_score(target_test_h2_0, pred_test_h2_0),DECIMAL_PLACES_METRICS)}, MSE={round(mean_squared_error(target_test_h2_0, pred_test_h2_0),DECIMAL_PLACES_METRICS)}, MAE={round(mean_absolute_error(target_test_h2_0, pred_test_h2_0),DECIMAL_PLACES_METRICS)}"
    plot_title = r"(NQS-Bench-101): True vs. Predicted $\log\psi_\omega(\vec{\sigma})$"
    ax1.set_title(plot_title + plot_metrics)
    ax2.plot(train_losses_h2_0, label="Train loss (h=2.0)")
    ax2.plot(valid_losses_h2_0, label="Valid loss (h=2.0)")
    ax1.plot(target_test_h2_0, pred_test_h2_0, 'o', markersize=5, label="h=2.0", alpha=0.5, mec='black')
    
    #ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
    #graph = go.Figure()
    graph.add_trace(go.Scatter(x=target_test_h2_0, y=pred_test_h2_0, mode='markers', name="h=2.0", opacity=0.5))
    
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=valid_losses_h2_0, mode='lines', name="Valid loss (h=2.0)"))
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=train_losses_h2_0, mode='lines', name="Train loss (h=2.0)"))
    
    
    torch.save(model_h2_0.state_dict(), model_weights_h2_0_path) if SAVING_WEIGHTS else None
    df_metrics_all = pd.concat([df_metrics_all, pd.DataFrame([df_metrics_h2_0])], ignore_index=True)
    
    all_values.append(list(target_test_h2_0) + 
        list(pred_test_h2_0))
    

all_values = list(itertools.chain.from_iterable(all_values))
min_val = min(all_values)
max_val = max(all_values)

perfect_prediction_x = [min_val, max_val]
print(f"Perferct prediction x: {perfect_prediction_x}")
ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y=x (perfect prediction)", linewidth=0.7, color='cyan', alpha=0.3)
graph.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name="y=x (perfect prediction)", opacity=0.3, marker=dict(color='cyan')))
    
ax1.legend(title="Legend")
ax2.legend(title="Legend")
    
f1.savefig(save_path_pred_true, dpi=300, bbox_inches="tight")
f2.savefig(save_path_loss_curve, dpi=300, bbox_inches="tight")
plt.show()





graph.show()
graph.write_html(save_path_pred_true_html)
graph2.show()
graph2.write_html(save_path_loss_curve_html)

df_metrics_all.to_csv(csv_file_path, index=False)
