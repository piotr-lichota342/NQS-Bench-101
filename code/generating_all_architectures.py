import matplotlib.pyplot as plt

from main import train_losses_h1_0e6, train_losses_h0_5, train_losses_h1_0, train_losses_h2_0
from main import model_h0_5, model_h1_0, model_h2_0, model_h1_0e6
from dataset_loading import train_dataloader_h0_5, train_dataloader_h1_0, train_dataloader_h2_0, train_dataloader_h1_0e6
from dataset_loading import test_dataloader_h0_5, test_dataloader_h1_0, test_dataloader_h2_0, test_dataloader_h1_0e6
from dataset_loading import valid_dataloader_h0_5, valid_dataloader_h1_0, valid_dataloader_h2_0, valid_dataloader_h1_0e6
from math import e
from main import valid_losses_h1_0e6, valid_losses_h0_5, valid_losses_h1_0, valid_losses_h2_0
from main import loss_fn, y_true_h1_0e6, y_pred_h1_0e6, y_true_h0_5, y_pred_h0_5, y_true_h1_0, y_pred_h1_0, y_true_h2_0, y_pred_h2_0, optimizer_h0_5, optimizer_h1_0, optimizer_h1_0e6, optimizer_h2_0, total_training_time
from config import CUSTOM_ARCH, EXPONENTIAL_LR, trained_regimes, EPOCHS, BATCH_SIZE, W, TEST_PROPORTION, TRAIN_PROPORTION, VALID_PROPORTION, HIDDEN_LAYERS, INPUT_SIZE, device, trained_regimes, DECIMAL_PLACES_METRICS, SAVING_WEIGHTS, TIMESTAMP, ACT_FUNCTION, DATASET_SIZE, WIDTH_SEQUENCE
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
from errorbars import infidelities_graph

from datetime import datetime

print("Hello generating all architectures.py")


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

def infidelity(true, pred):
    true_np, pred_np = np.exp(np.abs(true)), np.exp(np.abs(pred)) # converting from log amplitudes

    true_np /= np.sqrt(np.sum(true_np**2))
    pred_np /= np.sqrt(np.sum(pred_np**2))

    overlap = np.sum(np.multiply(np.squeeze(pred_np),np.squeeze(true_np)))
    return 1 - np.linalg.norm(overlap)**2

def smape(a, f):
    a_np, f_np = np.array(a), np.array(f)
    return 1/len(a) * np.sum(2 * np.abs(f_np-a_np) / (np.abs(a_np) + np.abs(f_np))*100)

def hellinger_distance(p,q):
    #print("Hellinger distance")
    #Turning into probabilities
    p, q = np.array(p), np.array(q)
    #p_prob, q_prob = [np.abs(a) for a in p], [np.abs(a) for a in q]
    p, q = np.abs(p), np.abs(q)
    #p_prob, q_prob = [a/np.sum(p_prob) for a in p_prob], [a/np.sum(q_prob) for a in q_prob]
    p = p / np.sum(p)
    q = q / np.sum(q)
    #print(f"p sum: {np.sum(p)}, q sum: {np.sum(q)}")
    
    
    #print(f"p_prob, q prob: {p_prob}, {q_prob}")

    #final_result = 0.0   
    #for i in range(len(p_prob)):
    diff = ((p)**(0.5) - (q)**(0.5))**2
    #print(f"sum diff: {np.sum(diff)}")
    diff = np.sum(diff)
    diff = diff**(0.5)
    #print(f"diff: {diff}")
    diff = (1/(2**(0.5))) * diff
    #print(f"diff: {diff}")
    
    final_result = diff
   
    #final_result = round(final_result,DECIMAL_PLACES_METRICS)
    
    return final_result


from pandas.errors import EmptyDataError

try:
    global_metrics_dataframe = pd.read_csv('all_architectures_metrics/all_architectures_metrics.csv')
except EmptyDataError:
    global_metrics_dataframe = pd.DataFrame()


epoch_range = [30,60]
print(f"Number of epochs: {EPOCHS}")
timestamp = TIMESTAMP
print(f"The Timestamp inside generating_all_architectures: is: {timestamp}")

str_regimes=["h_1_0e6", "h_0_5", "h_1_0", "h_2_0"]
str_epochs = [f"{x}_epochs" for x in epoch_range]
if not CUSTOM_ARCH:
    str_h_layers = ["1_hidden_layers", "2_hidden_layers", "3_hidden_layers"]
    str_widths = ["width_16", "width_32", "width_64", "width_128", "width_256", "width_512", "width_2048", "width_4096", "width_8192"]
else:
    str_h_layers = ["custom_hidden_layers"]
    str_widths = [f"width_{W}"]
    #print(f"This is the form of with: width_{W}")

str_act_fn = ["gelu", "tanh", "relu"]
# str_widths = ["width_1"]
# ["width_16", "width_32", "width_64", "width_128", "width_256", "width_512", "width_2048", "width_4096", "width_8192"]


activation=None
match int(ACT_FUNCTION):
    case 0:
        activation = 'gelu'
    case 1:
        activation = 'tanh'
    case 2:
        activation = 'relu'

os.makedirs(os.path.join('all_architectures', f"{timestamp}"), exist_ok=True)

for regim in str_regimes:
    os.makedirs(os.path.join(f'all_architectures/{timestamp}', regim), exist_ok=True)
    for hl in str_h_layers:
        os.makedirs(os.path.join(f'all_architectures/{timestamp}/{regim}', hl), exist_ok=True)
        for af in str_act_fn:
            os.makedirs(os.path.join(f'all_architectures/{timestamp}/{regim}/{hl}', af), exist_ok=True)
            for w in str_widths:
                os.makedirs(os.path.join(f'all_architectures/{timestamp}/{regim}/{hl}/{af}', w), exist_ok=True)
                for epoch in str_epochs:
                    os.makedirs(os.path.join(f'all_architectures/{timestamp}/{regim}/{hl}/{af}/{w}', epoch), exist_ok=True)

                    os.makedirs(os.path.join(f'all_architectures/{timestamp}/{regim}/{hl}/{af}/{w}/{epoch}', "curves"), exist_ok=True)
                    os.makedirs(os.path.join(f'all_architectures/{timestamp}/{regim}/{hl}/{af}/{w}/{epoch}', "evaluation_metrics"), exist_ok=True)
                    os.makedirs(os.path.join(f'all_architectures/{timestamp}/{regim}/{hl}/{af}/{w}/{epoch}', "model_weights"), exist_ok=True)

'''
os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_0_5/{EPOCHS}_epochs', "curves"), exist_ok=True)
os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_0_5/{EPOCHS}_epochs', "evaluation_metrics"), exist_ok=True)
os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_0_5/{EPOCHS}_epochs', "model_weights"), exist_ok=True)



os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_1_0/{EPOCHS}_epochs', "curves"), exist_ok=True)
os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_1_0/{EPOCHS}_epochs', "evaluation_metrics"), exist_ok=True)
os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_1_0/{EPOCHS}_epochs', "model_weights"), exist_ok=True)

os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_2_0/{EPOCHS}_epochs', "curves"), exist_ok=True)
os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_2_0/{EPOCHS}_epochs', "evaluation_metrics"), exist_ok=True)
os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_2_0/{EPOCHS}_epochs', "model_weights"), exist_ok=True)

os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_1_0e6/{EPOCHS}_epochs', "curves"), exist_ok=True)
os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_1_0e6/{EPOCHS}_epochs', "evaluation_metrics"), exist_ok=True)
os.makedirs(os.path.join(f'NQS-Bench-101/all_architectures/{timestamp}/h_1_0e6/{EPOCHS}_epochs', "model_weights"), exist_ok=True)
'''
save_path_loss_curve = None
save_path_pred_true = None

save_path_loss_curve_html = None
save_path_pred_true_html = None

csv_file_path = None

f'all_architectures/{timestamp}/h_0_5/{EPOCHS}_epochs/{HIDDEN_LAYERS}_hidden_layers/{activation}/model_weights'
model_weights_h0_5_path = f"all_architectures/{timestamp}/h_0_5/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/model_weights/model_weights_h0_5.pth"
model_weights_h1_0_path = f"all_architectures/{timestamp}/h_1_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/model_weights/model_weights_h1_0.pth"
model_weights_h2_0_path = f"all_architectures/{timestamp}/h_2_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/model_weights/model_weights_h2_0.pth"
model_weights_h1_0e6_path = f"all_architectures/{timestamp}/h_1_0e6/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/model_weights/model_weights_h1_0e6.pth"


'''
r2 = round(1.0⁻⁶(y_pred_h0_5, y_true_h0_5),4)
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
    'date': timestamp,
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
    'exponential_lr': EXPONENTIAL_LR,
    'hidden_layers': HIDDEN_LAYERS,
    'activation_fn': activation,
    'train_proportion': TRAIN_PROPORTION,
    'test_proportion': TEST_PROPORTION,
    'valid_proportion': VALID_PROPORTION,
    'device': str(device),
    'optimizer_name': None, # optimizer_h0_5.__class__.__name__
    'optimizer_params': None, # str(dict_optimizer)
    'loss_fn': str(loss_fn.__class__.__name__),
    'custom_arch': CUSTOM_ARCH,
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
    'hellinger_dist_valid': None,
    'infidelity': None
    
}


#print(f"Train loss length: {train_losses}")
all_values_h10e6 = []
all_values_h0_5 = []
all_values_h1_0 = []
all_values_h2_0 = []

df_metrics_all = pd.DataFrame()

f1_h10e6 = plt.figure(edgecolor='black')
f2_h10e6 = plt.figure()

ax1_h10e6 = f1_h10e6.add_subplot(1,1,1)
ax2_h10e6 = f2_h10e6.add_subplot(1,1,1)

ax1_h10e6.set_xlabel(r"True $\log\psi_\omega(\vec{\sigma})$")
ax1_h10e6.set_ylabel(r"Predicted $\log\psi_\omega(\vec{\sigma})$")




#f1.set_edgecolor("black")


ax1_h10e6.grid(True)
ax2_h10e6.set_xlabel("Epoch")
ax2_h10e6.set_ylabel(f"{loss_fn.__class__.__name__} (log scale)")
ax2_h10e6.set_title("(NQS-Bench-101): Training vs Validation Loss")


ax2_h10e6.grid(True)
ax2_h10e6.set_yscale("log")

graph_h10e6 = go.Figure()
graph2_h10e6 = go.Figure()

graph_h10e6.update_layout(
        xaxis_title="True " + "logψ_ω(vec{σ})", 
        yaxis_title="Predicted " + "logψ_ω(vec{σ})", 
        title="(NQS-Bench-101): True vs. Predicted " + "logψ_ω(vec{σ})", 
        showlegend=True,
        legend_title_text="Legend"
    )

graph2_h10e6.update_layout(
        xaxis_title="Epochs",
        yaxis_title=f"{loss_fn.__class__.__name__} (log scale)",
        title='(NQS-Bench-101): Train vs Valid Loss',
        showlegend=True,
        legend_title_text="Legend"
    )

###

f1_h0_5 = plt.figure(edgecolor='black')
f2_h0_5 = plt.figure()

ax1_h0_5 = f1_h0_5.add_subplot(1,1,1)
ax2_h0_5 = f2_h0_5.add_subplot(1,1,1)

ax1_h0_5.set_xlabel(r"True $\log\psi_\omega(\vec{\sigma})$")
ax1_h0_5.set_ylabel(r"Predicted $\log\psi_\omega(\vec{\sigma})$")




#f1.set_edgecolor("black")


ax1_h0_5.grid(True)
ax2_h0_5.set_xlabel("Epoch")
ax2_h0_5.set_ylabel(f"{loss_fn.__class__.__name__} (log scale)")
ax2_h0_5.set_title("(NQS-Bench-101): Training vs Validation Loss")


ax2_h0_5.grid(True)
ax2_h0_5.set_yscale("log")

graph_h0_5 = go.Figure()
graph2_h0_5 = go.Figure()

graph_h0_5.update_layout(
        xaxis_title="True " + "logψ_ω(vec{σ})", 
        yaxis_title="Predicted " + "logψ_ω(vec{σ})", 
        title="(NQS-Bench-101): True vs. Predicted " + "logψ_ω(vec{σ})", 
        showlegend=True,
        legend_title_text="Legend"
    )

graph2_h0_5.update_layout(
        xaxis_title="Epochs",
        yaxis_title=f"{loss_fn.__class__.__name__} (log scale)",
        title='(NQS-Bench-101): Train vs Valid Loss',
        showlegend=True,
        legend_title_text="Legend"
    )

###

f1_h1_0 = plt.figure(edgecolor='black')
f2_h1_0 = plt.figure()

ax1_h1_0 = f1_h1_0.add_subplot(1,1,1)
ax2_h1_0 = f2_h1_0.add_subplot(1,1,1)

ax1_h1_0.set_xlabel(r"True $\log\psi_\omega(\vec{\sigma})$")
ax1_h1_0.set_ylabel(r"Predicted $\log\psi_\omega(\vec{\sigma})$")




#f1.set_edgecolor("black")


ax1_h1_0.grid(True)
ax2_h1_0.set_xlabel("Epoch")
ax2_h1_0.set_ylabel(f"{loss_fn.__class__.__name__} (log scale)")
ax2_h1_0.set_title("(NQS-Bench-101): Training vs Validation Loss")


ax2_h1_0.grid(True)
ax2_h1_0.set_yscale("log")

graph_h1_0 = go.Figure()
graph2_h1_0 = go.Figure()

graph_h1_0.update_layout(
        xaxis_title="True " + "logψ_ω(vec{σ})", 
        yaxis_title="Predicted " + "logψ_ω(vec{σ})", 
        title="(NQS-Bench-101): True vs. Predicted " + "logψ_ω(vec{σ})", 
        showlegend=True,
        legend_title_text="Legend"
    )

graph2_h1_0.update_layout(
        xaxis_title="Epochs",
        yaxis_title=f"{loss_fn.__class__.__name__} (log scale)",
        title='(NQS-Bench-101): Train vs Valid Loss',
        showlegend=True,
        legend_title_text="Legend"
    )

###

f1_h2_0 = plt.figure(edgecolor='black')
f2_h2_0 = plt.figure()

ax1_h2_0 = f1_h2_0.add_subplot(1,1,1)
ax2_h2_0 = f2_h2_0.add_subplot(1,1,1)

ax1_h2_0.set_xlabel(r"True $\log\psi_\omega(\vec{\sigma})$")
ax1_h2_0.set_ylabel(r"Predicted $\log\psi_\omega(\vec{\sigma})$")




#f1.set_edgecolor("black")


ax1_h2_0.grid(True)
ax2_h2_0.set_xlabel("Epoch")
ax2_h2_0.set_ylabel(f"{loss_fn.__class__.__name__} (log scale)")
ax2_h2_0.set_title("(NQS-Bench-101): Training vs Validation Loss")


ax2_h2_0.grid(True)
ax2_h2_0.set_yscale("log")

graph_h2_0 = go.Figure()
graph2_h2_0 = go.Figure()

graph_h2_0.update_layout(
        xaxis_title="True " + "logψ_ω(vec{σ})", 
        yaxis_title="Predicted " + "logψ_ω(vec{σ})", 
        title="(NQS-Bench-101): True vs. Predicted " + "logψ_ω(vec{σ})", 
        showlegend=True,
        legend_title_text="Legend"
    )

graph2_h2_0.update_layout(
        xaxis_title="Epochs",
        yaxis_title=f"{loss_fn.__class__.__name__} (log scale)",
        title='(NQS-Bench-101): Train vs Valid Loss',
        showlegend=True,
        legend_title_text="Legend"
    )



if int(trained_regimes[0]):
    save_path_loss_curve_h10e6 = f"all_architectures/{timestamp}/h_1_0e6/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/loss_curve.png"
    save_path_pred_true_h10e6 = f"all_architectures/{timestamp}/h_1_0e6/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/pred_true_curve.png"

    save_path_loss_curve_html_h10e6 = f"all_architectures/{timestamp}/h_1_0e6/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/loss_curve.html"
    save_path_pred_true_html_h10e6 = f"all_architectures/{timestamp}/h_1_0e6/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/pred_true_curve.html"
    save_path_infidelity_regimes_h10e6 = f"all_architectures/{timestamp}/h_1_0e6/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/infidelity_regimes.png"

    csv_file_path_h10e6 = f'all_architectures/{timestamp}/h_1_0e6/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/metrics.csv'
    
    avg_test_loss_h1_0e6, amplitudes_h1_0e6, target_test_h1_0e6, pred_test_h1_0e6 = test(test_dataloader_h1_0e6, model_h1_0e6, loss_fn)
    avg_train_loss_h1_0e6, target_train_h1_0e6, pred_train_h1_0e6 = train(train_dataloader_h1_0e6, model_h1_0e6, loss_fn, optimizer_h1_0e6)
    avg_valid_loss_h1_0e6, target_valid_h1_0e6, pred_valid_h1_0e6 = valid(valid_dataloader_h1_0e6, model_h1_0e6, loss_fn)

    print(f"avg_train_loss_h1_0e6: {avg_train_loss_h1_0e6}")
    print(f"target_train_h1_0e6: {target_train_h1_0e6}")
    print(f"pred_train_h1_0e6: {pred_train_h1_0e6}")

    #perfect_prediction_x = [0,max(max(y_true_h1_0e6), max(y_pred_h1_0e6))]
    dict_optimizer_h1_0e6 = optimizer_h1_0e6.param_groups[0]
    dict_optimizer_h1_0e6.pop('params')
    
    total_params_h10e6 = sum(p_h1_0e6.numel() for p_h1_0e6 in model_h1_0e6.parameters())
    train_params_h10e6 = sum(p_h1_0e6.numel() for p_h1_0e6 in model_h1_0e6.parameters() if p_h1_0e6.requires_grad)
    
    with open(f'all_architectures/{timestamp}/h_1_0e6/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/pred_test_h1_0e6.txt', 'w') as file_h10e6:
        # Join the list elements into a single string with a newline character
        #print(f"pred_test_h1_0e6.tolist(): {pred_test_h1_0e6.tolist()}")
        print(f"len pred_test_h1_0e6: {len(pred_test_h1_0e6)}")
        all_hilbert = np.concatenate((pred_test_h1_0e6, pred_train_h1_0e6, pred_valid_h1_0e6), axis=None)
        print(f"The size of Hilbert space is: {len(all_hilbert)}")
        #all_hilbert_normalized = 
        all_hilbert = [(np.abs(np.exp(x))) for x in all_hilbert]
        all_hilbert = [x / np.sum(all_hilbert) for x in all_hilbert]
        all_hilbert = sum(all_hilbert)

        print(f"The sum of normalized pred amplitudes is: {all_hilbert}")
        data_to_write_h10e6 = '\n'.join([str(x_h1_0e6.tolist()) for x_h1_0e6 in pred_test_h1_0e6])
        # Write the data to the file
        file_h10e6.write(data_to_write_h10e6)
        
    with open(f'all_architectures/{timestamp}/h_1_0e6/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/target_test_h1_0e6.txt', 'w') as file_h10e6:
        # Join the list elements into a single string with a newline character
        data_to_write_h10e6 = '\n'.join([str(x_h1_0e6.tolist()) for x_h1_0e6 in target_test_h1_0e6])
        # Write the data to the file
        file_h10e6.write(data_to_write_h10e6)
    with open(f'all_architectures/{timestamp}/h_1_0e6/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/input_amplitudes_h1_0e6.txt', 'w') as file_h10e6:
        # Join the list elements into a single string with a newline character
        #print(f"amplitudes_h1_0e6 data type: {type(amplitudes_h1_0e6)}")
        str_confs_h10e6 = []
        for x_h10e6 in amplitudes_h1_0e6:
            for y_h10e6 in x_h10e6:
                y_h10e6 = y_h10e6.tolist()
                int_conf_h10e6 = [str(int(z_h10e6)) for z_h10e6 in y_h10e6]
                conf_str_h10e6 = "".join(int_conf_h10e6)
                str_confs_h10e6.append(conf_str_h10e6)
                #print(conf_str)
        data_to_write_h10e6 = '\n'.join(str_confs_h10e6)
        # Write the data to the file
        file_h10e6.write(data_to_write_h10e6)
    

    
    print(f"Target valid and pred valid: {target_valid_h1_0e6, pred_valid_h1_0e6}")
    #ax1 = f1.add_axes(train_losses_h1_0e6)
    df_metrics_h1_0e6 = metrics_data.copy()
    df_metrics_h1_0e6['regime'] = "h=1.0⁻⁶"
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
    df_metrics_h1_0e6['total_params'] = total_params_h10e6
    df_metrics_h1_0e6['train_params'] = train_params_h10e6
    df_metrics_h1_0e6['non_train_params'] = total_params_h10e6 - train_params_h10e6
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
    
    df_metrics_h1_0e6['infidelity'] = round(infidelity(np.concatenate((target_test_h1_0e6, target_train_h1_0e6, target_valid_h1_0e6), axis=None), np.concatenate((pred_test_h1_0e6, pred_train_h1_0e6, pred_valid_h1_0e6), axis=None)), DECIMAL_PLACES_METRICS)

        
    
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
    plot_metrics_h10e6 = "\n"+r"$R^2$"+f"={round(r2_score(target_test_h1_0e6, pred_test_h1_0e6),DECIMAL_PLACES_METRICS)}, MSE={round(mean_squared_error(target_test_h1_0e6, pred_test_h1_0e6),DECIMAL_PLACES_METRICS)}, MAE={round(mean_absolute_error(target_test_h1_0e6, pred_test_h1_0e6),DECIMAL_PLACES_METRICS)}"
    plot_title_h10e6 = r"(NQS-Bench-101): True vs. Predicted $\log\psi_\omega(\vec{\sigma})$"
    ax1_h10e6.set_title(plot_title_h10e6 + plot_metrics_h10e6)
    ax2_h10e6.plot(train_losses_h1_0e6, label="Train loss (h=10⁻⁶)")
    ax2_h10e6.plot(valid_losses_h1_0e6, label="Valid loss (h=10⁻⁶)")
    ax1_h10e6.plot(target_test_h1_0e6, pred_test_h1_0e6, 'o', markersize=5, label="h=10⁻⁶", alpha=0.5, mec='black')
    
    
    
    
    #ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
    #graph = go.Figure()
    graph_h10e6.add_trace(go.Scatter(x=target_test_h1_0e6, y=pred_test_h1_0e6, mode='markers', name="h=10⁻⁶", opacity=0.5))
    
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2_h10e6.add_trace(go.Scatter(x=[e_h1_0e6 for e_h1_0e6 in range(1,EPOCHS+1,1)], y=valid_losses_h1_0e6, mode='lines', name="Valid loss (h=10⁻⁶)"))
    graph2_h10e6.add_trace(go.Scatter(x=[e_h1_0e6 for e_h1_0e6 in range(1,EPOCHS+1,1)], y=train_losses_h1_0e6, mode='lines', name="Train loss (h=10⁻⁶)"))
    
    
    torch.save(model_h1_0e6.state_dict(), model_weights_h1_0e6_path) if SAVING_WEIGHTS else None
    df_metrics_all = pd.concat([df_metrics_all, pd.DataFrame([df_metrics_h1_0e6])], ignore_index=True)
    global_metrics_dataframe = pd.concat([global_metrics_dataframe, pd.DataFrame([df_metrics_h1_0e6])], ignore_index=True)
    
    all_values_h10e6.append(list(target_test_h1_0e6) + 
        list(pred_test_h1_0e6))
    
    df_metrics_h1_0e6 = pd.DataFrame([df_metrics_h1_0e6])
    df_metrics_h1_0e6.to_csv(csv_file_path_h10e6, index=False)
    
    

if int(trained_regimes[1]):
    
    save_path_loss_curve_h0_5 = f"all_architectures/{timestamp}/h_0_5/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/loss_curve.png"
    save_path_pred_true_h0_5 = f"all_architectures/{timestamp}/h_0_5/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/pred_true_curve.png"

    save_path_loss_curve_html_h0_5 = f"all_architectures/{timestamp}/h_0_5/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/loss_curve.html"
    save_path_pred_true_html_h0_5 = f"all_architectures/{timestamp}/h_0_5/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/pred_true_curve.html"
    save_path_infidelity_regimes_h0_5 = f"all_architectures/{timestamp}/h_0_5/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/infidelity_regimes.png"

    csv_file_path_h0_5 = f'all_architectures/{timestamp}/h_0_5/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/metrics.csv'
    
    avg_test_loss_h0_5, amplitudes_h0_5, target_test_h0_5, pred_test_h0_5 = test(test_dataloader_h0_5, model_h0_5, loss_fn)
    avg_train_loss_h0_5, target_train_h0_5, pred_train_h0_5 = train(train_dataloader_h0_5, model_h0_5, loss_fn, optimizer_h0_5)
    avg_valid_loss_h0_5, target_valid_h0_5, pred_valid_h0_5 = valid(valid_dataloader_h0_5, model_h0_5, loss_fn)

    print(f"avg_train_loss_h0_5: {avg_train_loss_h0_5}")
    print(f"target_train_h0_5: {target_train_h0_5}")
    print(f"pred_train_h0_5: {pred_train_h0_5}")

    #perfect_prediction_x = [0,max(max(y_true_h0_5), max(y_pred_h0_5))]
    dict_optimizer_h0_5 = optimizer_h0_5.param_groups[0]
    dict_optimizer_h0_5.pop('params')
    
    total_params_h0_5 = sum(p_h0_5.numel() for p_h0_5 in model_h0_5.parameters())
    train_params_h0_5 = sum(p_h0_5.numel() for p_h0_5 in model_h0_5.parameters() if p_h0_5.requires_grad)
    
    with open(f'all_architectures/{timestamp}/h_0_5/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/pred_test_h0_5.txt', 'w') as file_h0_5:
        # Join the list elements into a single string with a newline character
        #print(f"pred_test_h0_5.tolist(): {pred_test_h0_5.tolist()}")
        print(f"len pred_test_h0_5: {len(pred_test_h0_5)}")
        all_hilbert = np.concatenate((pred_test_h0_5, pred_train_h0_5, pred_valid_h0_5), axis=None)
        print(f"The size of Hilbert space is: {len(all_hilbert)}")
        #all_hilbert_normalized = 
        all_hilbert = [(np.abs(np.exp(x))) for x in all_hilbert]
        all_hilbert = [x / np.sum(all_hilbert) for x in all_hilbert]
        all_hilbert = sum(all_hilbert)

        print(f"The sum of normalized pred amplitudes is: {all_hilbert}")
        data_to_write_h0_5 = '\n'.join([str(x_h0_5.tolist()) for x_h0_5 in pred_test_h0_5])
        # Write the data to the file
        file_h0_5.write(data_to_write_h0_5)
        
    with open(f'all_architectures/{timestamp}/h_0_5/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/target_test_h0_5.txt', 'w') as file_h0_5:
        # Join the list elements into a single string with a newline character
        data_to_write_h0_5 = '\n'.join([str(x_h0_5.tolist()) for x_h0_5 in target_test_h0_5])
        # Write the data to the file
        file_h0_5.write(data_to_write_h0_5)
    with open(f'all_architectures/{timestamp}/h_0_5/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/input_amplitudes_h0_5.txt', 'w') as file_h0_5:
        # Join the list elements into a single string with a newline character
        #print(f"amplitudes_h0_5 data type: {type(amplitudes_h0_5)}")
        str_confs_h0_5 = []
        for x_h0_5 in amplitudes_h0_5:
            for y_h0_5 in x_h0_5:
                y_h0_5 = y_h0_5.tolist()
                int_conf_h0_5 = [str(int(z_h0_5)) for z_h0_5 in y_h0_5]
                conf_str_h0_5 = "".join(int_conf_h0_5)
                str_confs_h0_5.append(conf_str_h0_5)
                #print(conf_str)
        data_to_write_h0_5 = '\n'.join(str_confs_h0_5)
        # Write the data to the file
        file_h0_5.write(data_to_write_h0_5)
    

    
    print(f"Target valid and pred valid: {target_valid_h0_5, pred_valid_h0_5}")
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
    df_metrics_h0_5['total_params'] = total_params_h0_5
    df_metrics_h0_5['train_params'] = train_params_h0_5
    df_metrics_h0_5['non_train_params'] = total_params_h0_5 - train_params_h0_5
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
    
    df_metrics_h0_5['infidelity'] = round(infidelity(np.concatenate((target_test_h0_5, target_train_h0_5, target_valid_h0_5), axis=None), np.concatenate((pred_test_h0_5, pred_train_h0_5, pred_valid_h0_5), axis=None)), DECIMAL_PLACES_METRICS)

        
    
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
    plot_metrics_h0_5 = "\n"+r"$R^2$"+f"={round(r2_score(target_test_h0_5, pred_test_h0_5),DECIMAL_PLACES_METRICS)}, MSE={round(mean_squared_error(target_test_h0_5, pred_test_h0_5),DECIMAL_PLACES_METRICS)}, MAE={round(mean_absolute_error(target_test_h0_5, pred_test_h0_5),DECIMAL_PLACES_METRICS)}"
    plot_title_h0_5 = r"(NQS-Bench-101): True vs. Predicted $\log\psi_\omega(\vec{\sigma})$"
    ax1_h0_5.set_title(plot_title_h0_5 + plot_metrics_h0_5)
    ax2_h0_5.plot(train_losses_h0_5, label="Train loss (h=0.5)")
    ax2_h0_5.plot(valid_losses_h0_5, label="Valid loss (h=0.5)")
    ax1_h0_5.plot(target_test_h0_5, pred_test_h0_5, 'o', markersize=5, label="h=0.5", alpha=0.5, mec='black')
    
    
    
    
    #ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
    #graph = go.Figure()
    graph_h0_5.add_trace(go.Scatter(x=target_test_h0_5, y=pred_test_h0_5, mode='markers', name="h=0.5", opacity=0.5))
    
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2_h0_5.add_trace(go.Scatter(x=[e_h0_5 for e_h0_5 in range(1,EPOCHS+1,1)], y=valid_losses_h0_5, mode='lines', name="Valid loss (h=0.5)"))
    graph2_h0_5.add_trace(go.Scatter(x=[e_h0_5 for e_h0_5 in range(1,EPOCHS+1,1)], y=train_losses_h0_5, mode='lines', name="Train loss (h=0.5)"))
    
    
    torch.save(model_h0_5.state_dict(), model_weights_h0_5_path) if SAVING_WEIGHTS else None
    df_metrics_all = pd.concat([df_metrics_all, pd.DataFrame([df_metrics_h0_5])], ignore_index=True)
    global_metrics_dataframe = pd.concat([global_metrics_dataframe, pd.DataFrame([df_metrics_h0_5])], ignore_index=True)
    
    all_values_h0_5.append(list(target_test_h0_5) + 
        list(pred_test_h0_5))
    
    df_metrics_h0_5 = pd.DataFrame([df_metrics_h0_5])
    df_metrics_h0_5.to_csv(csv_file_path_h0_5, index=False)
    
    
    
if int(trained_regimes[2]):
    save_path_loss_curve_h1_0 = f"all_architectures/{timestamp}/h_1_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/loss_curve.png"
    save_path_pred_true_h1_0 = f"all_architectures/{timestamp}/h_1_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/pred_true_curve.png"

    save_path_loss_curve_html_h1_0 = f"all_architectures/{timestamp}/h_1_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/loss_curve.html"
    save_path_pred_true_html_h1_0 = f"all_architectures/{timestamp}/h_1_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/pred_true_curve.html"
    save_path_infidelity_regimes_h1_0 = f"all_architectures/{timestamp}/h_1_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/infidelity_regimes.png"

    csv_file_path_h1_0 = f'all_architectures/{timestamp}/h_1_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/metrics.csv'
    
    avg_test_loss_h1_0, amplitudes_h1_0, target_test_h1_0, pred_test_h1_0 = test(test_dataloader_h1_0, model_h1_0, loss_fn)
    avg_train_loss_h1_0, target_train_h1_0, pred_train_h1_0 = train(train_dataloader_h1_0, model_h1_0, loss_fn, optimizer_h1_0)
    avg_valid_loss_h1_0, target_valid_h1_0, pred_valid_h1_0 = valid(valid_dataloader_h1_0, model_h1_0, loss_fn)

    print(f"avg_train_loss_h1_0: {avg_train_loss_h1_0}")
    print(f"target_train_h1_0: {target_train_h1_0}")
    print(f"pred_train_h1_0: {pred_train_h1_0}")

    #perfect_prediction_x = [0,max(max(y_true_h1_0), max(y_pred_h1_0))]
    dict_optimizer_h1_0 = optimizer_h1_0.param_groups[0]
    dict_optimizer_h1_0.pop('params')
    
    total_params_h1_0 = sum(p_h1_0.numel() for p_h1_0 in model_h1_0.parameters())
    train_params_h1_0 = sum(p_h1_0.numel() for p_h1_0 in model_h1_0.parameters() if p_h1_0.requires_grad)
    
    with open(f'all_architectures/{timestamp}/h_1_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/pred_test_h1_0.txt', 'w') as file_h1_0:
        # Join the list elements into a single string with a newline character
        #print(f"pred_test_h1_0.tolist(): {pred_test_h1_0.tolist()}")
        print(f"len pred_test_h1_0: {len(pred_test_h1_0)}")
        all_hilbert = np.concatenate((pred_test_h1_0, pred_train_h1_0, pred_valid_h1_0), axis=None)
        print(f"The size of Hilbert space is: {len(all_hilbert)}")
        #all_hilbert_normalized = 
        all_hilbert = [(np.abs(np.exp(x))) for x in all_hilbert]
        all_hilbert = [x / np.sum(all_hilbert) for x in all_hilbert]
        all_hilbert = sum(all_hilbert)

        print(f"The sum of normalized pred amplitudes is: {all_hilbert}")
        data_to_write_h1_0 = '\n'.join([str(x_h1_0.tolist()) for x_h1_0 in pred_test_h1_0])
        # Write the data to the file
        file_h1_0.write(data_to_write_h1_0)
        
    with open(f'all_architectures/{timestamp}/h_1_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/target_test_h1_0.txt', 'w') as file_h1_0:
        # Join the list elements into a single string with a newline character
        data_to_write_h1_0 = '\n'.join([str(x_h1_0.tolist()) for x_h1_0 in target_test_h1_0])
        # Write the data to the file
        file_h1_0.write(data_to_write_h1_0)
    with open(f'all_architectures/{timestamp}/h_1_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/input_amplitudes_h1_0.txt', 'w') as file_h1_0:
        # Join the list elements into a single string with a newline character
        #print(f"amplitudes_h1_0 data type: {type(amplitudes_h1_0)}")
        str_confs_h1_0 = []
        for x_h1_0 in amplitudes_h1_0:
            for y_h1_0 in x_h1_0:
                y_h1_0 = y_h1_0.tolist()
                int_conf_h1_0 = [str(int(z_h1_0)) for z_h1_0 in y_h1_0]
                conf_str_h1_0 = "".join(int_conf_h1_0)
                str_confs_h1_0.append(conf_str_h1_0)
                #print(conf_str)
        data_to_write_h1_0 = '\n'.join(str_confs_h1_0)
        # Write the data to the file
        file_h1_0.write(data_to_write_h1_0)
    

    
    print(f"Target valid and pred valid: {target_valid_h1_0, pred_valid_h1_0}")
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
    df_metrics_h1_0['total_params'] = total_params_h1_0
    df_metrics_h1_0['train_params'] = train_params_h1_0
    df_metrics_h1_0['non_train_params'] = total_params_h1_0 - train_params_h1_0
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
    
    df_metrics_h1_0['infidelity'] = round(infidelity(np.concatenate((target_test_h1_0, target_train_h1_0, target_valid_h1_0), axis=None), np.concatenate((pred_test_h1_0, pred_train_h1_0, pred_valid_h1_0), axis=None)), DECIMAL_PLACES_METRICS)

        
    
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
    plot_metrics_h1_0 = "\n"+r"$R^2$"+f"={round(r2_score(target_test_h1_0, pred_test_h1_0),DECIMAL_PLACES_METRICS)}, MSE={round(mean_squared_error(target_test_h1_0, pred_test_h1_0),DECIMAL_PLACES_METRICS)}, MAE={round(mean_absolute_error(target_test_h1_0, pred_test_h1_0),DECIMAL_PLACES_METRICS)}"
    plot_title_h1_0 = r"(NQS-Bench-101): True vs. Predicted $\log\psi_\omega(\vec{\sigma})$"
    ax1_h1_0.set_title(plot_title_h1_0 + plot_metrics_h1_0)
    ax2_h1_0.plot(train_losses_h1_0, label="Train loss (h=1.0)")
    ax2_h1_0.plot(valid_losses_h1_0, label="Valid loss (h=1.0)")
    ax1_h1_0.plot(target_test_h1_0, pred_test_h1_0, 'o', markersize=5, label="h=1.0", alpha=0.5, mec='black')
    
    
    
    
    #ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
    #graph = go.Figure()
    graph_h1_0.add_trace(go.Scatter(x=target_test_h1_0, y=pred_test_h1_0, mode='markers', name="h=1.0", opacity=0.5))
    
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2_h1_0.add_trace(go.Scatter(x=[e_h1_0 for e_h1_0 in range(1,EPOCHS+1,1)], y=valid_losses_h1_0, mode='lines', name="Valid loss (h=1.0)"))
    graph2_h1_0.add_trace(go.Scatter(x=[e_h1_0 for e_h1_0 in range(1,EPOCHS+1,1)], y=train_losses_h1_0, mode='lines', name="Train loss (h=1.0)"))
    
    
    torch.save(model_h1_0.state_dict(), model_weights_h1_0_path) if SAVING_WEIGHTS else None
    df_metrics_all = pd.concat([df_metrics_all, pd.DataFrame([df_metrics_h1_0])], ignore_index=True)
    global_metrics_dataframe = pd.concat([global_metrics_dataframe, pd.DataFrame([df_metrics_h1_0])], ignore_index=True)
    
    all_values_h1_0.append(list(target_test_h1_0) + 
        list(pred_test_h1_0))
    
    df_metrics_h1_0 = pd.DataFrame([df_metrics_h1_0])
    df_metrics_h1_0.to_csv(csv_file_path_h1_0, index=False)
    
if int(trained_regimes[3]):
    save_path_loss_curve_h2_0 = f"all_architectures/{timestamp}/h_2_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/loss_curve.png"
    save_path_pred_true_h2_0 = f"all_architectures/{timestamp}/h_2_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/pred_true_curve.png"

    save_path_loss_curve_html_h2_0 = f"all_architectures/{timestamp}/h_2_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/loss_curve.html"
    save_path_pred_true_html_h2_0 = f"all_architectures/{timestamp}/h_2_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/pred_true_curve.html"
    save_path_infidelity_regimes_h2_0 = f"all_architectures/{timestamp}/h_2_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/curves/infidelity_regimes.png"

    csv_file_path_h2_0 = f'all_architectures/{timestamp}/h_2_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/metrics.csv'
    
    avg_test_loss_h2_0, amplitudes_h2_0, target_test_h2_0, pred_test_h2_0 = test(test_dataloader_h2_0, model_h2_0, loss_fn)
    avg_train_loss_h2_0, target_train_h2_0, pred_train_h2_0 = train(train_dataloader_h2_0, model_h2_0, loss_fn, optimizer_h2_0)
    avg_valid_loss_h2_0, target_valid_h2_0, pred_valid_h2_0 = valid(valid_dataloader_h2_0, model_h2_0, loss_fn)

    print(f"avg_train_loss_h2_0: {avg_train_loss_h2_0}")
    print(f"target_train_h2_0: {target_train_h2_0}")
    print(f"pred_train_h2_0: {pred_train_h2_0}")

    #perfect_prediction_x = [0,max(max(y_true_h2_0), max(y_pred_h2_0))]
    dict_optimizer_h2_0 = optimizer_h2_0.param_groups[0]
    dict_optimizer_h2_0.pop('params')
    
    total_params_h2_0 = sum(p_h2_0.numel() for p_h2_0 in model_h2_0.parameters())
    train_params_h2_0 = sum(p_h2_0.numel() for p_h2_0 in model_h2_0.parameters() if p_h2_0.requires_grad)
    
    with open(f'all_architectures/{timestamp}/h_2_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/pred_test_h2_0.txt', 'w') as file_h2_0:
        # Join the list elements into a single string with a newline character
        #print(f"pred_test_h2_0.tolist(): {pred_test_h2_0.tolist()}")
        print(f"len pred_test_h2_0: {len(pred_test_h2_0)}")
        all_hilbert = np.concatenate((pred_test_h2_0, pred_train_h2_0, pred_valid_h2_0), axis=None)
        print(f"The size of Hilbert space is: {len(all_hilbert)}")
        #all_hilbert_normalized = 
        all_hilbert = [(np.abs(np.exp(x))) for x in all_hilbert]
        all_hilbert = [x / np.sum(all_hilbert) for x in all_hilbert]
        all_hilbert = sum(all_hilbert)

        print(f"The sum of normalized pred amplitudes is: {all_hilbert}")
        data_to_write_h2_0 = '\n'.join([str(x_h2_0.tolist()) for x_h2_0 in pred_test_h2_0])
        # Write the data to the file
        file_h2_0.write(data_to_write_h2_0)
        
    with open(f'all_architectures/{timestamp}/h_2_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/target_test_h2_0.txt', 'w') as file_h2_0:
        # Join the list elements into a single string with a newline character
        data_to_write_h2_0 = '\n'.join([str(x_h2_0.tolist()) for x_h2_0 in target_test_h2_0])
        # Write the data to the file
        file_h2_0.write(data_to_write_h2_0)
    with open(f'all_architectures/{timestamp}/h_2_0/{HIDDEN_LAYERS}_hidden_layers/{activation}/width_{W}/{EPOCHS}_epochs/evaluation_metrics/input_amplitudes_h2_0.txt', 'w') as file_h2_0:
        # Join the list elements into a single string with a newline character
        #print(f"amplitudes_h2_0 data type: {type(amplitudes_h2_0)}")
        str_confs_h2_0 = []
        for x_h2_0 in amplitudes_h2_0:
            for y_h2_0 in x_h2_0:
                y_h2_0 = y_h2_0.tolist()
                int_conf_h2_0 = [str(int(z_h2_0)) for z_h2_0 in y_h2_0]
                conf_str_h2_0 = "".join(int_conf_h2_0)
                str_confs_h2_0.append(conf_str_h2_0)
                #print(conf_str)
        data_to_write_h2_0 = '\n'.join(str_confs_h2_0)
        # Write the data to the file
        file_h2_0.write(data_to_write_h2_0)
    

    
    print(f"Target valid and pred valid: {target_valid_h2_0, pred_valid_h2_0}")
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
    df_metrics_h2_0['total_params'] = total_params_h2_0
    df_metrics_h2_0['train_params'] = train_params_h2_0
    df_metrics_h2_0['non_train_params'] = total_params_h2_0 - train_params_h2_0
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
    
    df_metrics_h2_0['infidelity'] = round(infidelity(np.concatenate((target_test_h2_0, target_train_h2_0, target_valid_h2_0), axis=None), np.concatenate((pred_test_h2_0, pred_train_h2_0, pred_valid_h2_0), axis=None)), DECIMAL_PLACES_METRICS)

        
    
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
    plot_metrics_h2_0 = "\n"+r"$R^2$"+f"={round(r2_score(target_test_h2_0, pred_test_h2_0),DECIMAL_PLACES_METRICS)}, MSE={round(mean_squared_error(target_test_h2_0, pred_test_h2_0),DECIMAL_PLACES_METRICS)}, MAE={round(mean_absolute_error(target_test_h2_0, pred_test_h2_0),DECIMAL_PLACES_METRICS)}"
    plot_title_h2_0 = r"(NQS-Bench-101): True vs. Predicted $\log\psi_\omega(\vec{\sigma})$"
    ax1_h2_0.set_title(plot_title_h2_0 + plot_metrics_h2_0)
    ax2_h2_0.plot(train_losses_h2_0, label="Train loss (h=2.0)")
    ax2_h2_0.plot(valid_losses_h2_0, label="Valid loss (h=2.0)")
    ax1_h2_0.plot(target_test_h2_0, pred_test_h2_0, 'o', markersize=5, label="h=2.0", alpha=0.5, mec='black')
    
    
    
    
    #ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
    #graph = go.Figure()
    graph_h2_0.add_trace(go.Scatter(x=target_test_h2_0, y=pred_test_h2_0, mode='markers', name="h=2.0", opacity=0.5))
    
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2_h2_0.add_trace(go.Scatter(x=[e_h2_0 for e_h2_0 in range(1,EPOCHS+1,1)], y=valid_losses_h2_0, mode='lines', name="Valid loss (h=2.0)"))
    graph2_h2_0.add_trace(go.Scatter(x=[e_h2_0 for e_h2_0 in range(1,EPOCHS+1,1)], y=train_losses_h2_0, mode='lines', name="Train loss (h=2.0)"))
    
    
    torch.save(model_h2_0.state_dict(), model_weights_h2_0_path) if SAVING_WEIGHTS else None
    df_metrics_all = pd.concat([df_metrics_all, pd.DataFrame([df_metrics_h2_0])], ignore_index=True)
    global_metrics_dataframe = pd.concat([global_metrics_dataframe, pd.DataFrame([df_metrics_h2_0])], ignore_index=True)
    
    all_values_h2_0.append(list(target_test_h2_0) + 
        list(pred_test_h2_0))
    
    df_metrics_h2_0 = pd.DataFrame([df_metrics_h2_0])
    df_metrics_h2_0.to_csv(csv_file_path_h2_0, index=False)
    

all_values_h10e6 = list(itertools.chain.from_iterable(all_values_h10e6))
all_values_h0_5 = list(itertools.chain.from_iterable(all_values_h0_5))
all_values_h1_0 = list(itertools.chain.from_iterable(all_values_h1_0))
all_values_h2_0 = list(itertools.chain.from_iterable(all_values_h2_0))

if int(trained_regimes[0]):
    min_val_h10e6, max_val_h10e6 = min(all_values_h10e6), max(all_values_h10e6)
    perfect_prediction_x_h10e6 = [min_val_h10e6, max_val_h10e6]
    ax1_h10e6.plot(perfect_prediction_x_h10e6, perfect_prediction_x_h10e6, '-', label="y=x (perfect prediction)", linewidth=0.7, color='cyan', alpha=0.3)
    graph_h10e6.add_trace(go.Scatter(x=[min_val_h10e6, max_val_h10e6], y=[min_val_h10e6, max_val_h10e6], mode='lines', name="y=x (perfect prediction)", opacity=0.3, marker=dict(color='cyan')))
    ax1_h10e6.legend(title="Legend")
    ax2_h10e6.legend(title="Legend")
    f1_h10e6.savefig(save_path_pred_true_h10e6, dpi=300, bbox_inches="tight")
    f2_h10e6.savefig(save_path_loss_curve_h10e6, dpi=300, bbox_inches="tight")
    #graph.show()
    graph_h10e6.write_html(save_path_pred_true_html_h10e6)
    #graph2.show()
    graph2_h10e6.write_html(save_path_loss_curve_html_h10e6)
if int(trained_regimes[1]):    
    min_val_h0_5, max_val_h0_5 = min(all_values_h0_5), max(all_values_h0_5)
    perfect_prediction_x_h0_5 = [min_val_h0_5, max_val_h0_5]
    ax1_h0_5.plot(perfect_prediction_x_h0_5, perfect_prediction_x_h0_5, '-', label="y=x (perfect prediction)", linewidth=0.7, color='cyan', alpha=0.3)
    graph_h0_5.add_trace(go.Scatter(x=[min_val_h0_5, max_val_h0_5], y=[min_val_h0_5, max_val_h0_5], mode='lines', name="y=x (perfect prediction)", opacity=0.3, marker=dict(color='cyan')))
    ax1_h0_5.legend(title="Legend")
    ax2_h0_5.legend(title="Legend")
        
    f1_h0_5.savefig(save_path_pred_true_h0_5, dpi=300, bbox_inches="tight")
    f2_h0_5.savefig(save_path_loss_curve_h0_5, dpi=300, bbox_inches="tight")
    #graph.show()
    graph_h0_5.write_html(save_path_pred_true_html_h0_5)
    #graph2.show()
    graph2_h0_5.write_html(save_path_loss_curve_html_h0_5)
if int(trained_regimes[2]):
    min_val_h1_0, max_val_h1_0 = min(all_values_h1_0), max(all_values_h1_0)
    perfect_prediction_x_h1_0 = [min_val_h1_0, max_val_h1_0]
    ax1_h1_0.plot(perfect_prediction_x_h1_0, perfect_prediction_x_h1_0, '-', label="y=x (perfect prediction)", linewidth=0.7, color='cyan', alpha=0.3)
    graph_h1_0.add_trace(go.Scatter(x=[min_val_h1_0, max_val_h1_0], y=[min_val_h1_0, max_val_h1_0], mode='lines', name="y=x (perfect prediction)", opacity=0.3, marker=dict(color='cyan')))
    ax1_h1_0.legend(title="Legend")
    ax2_h1_0.legend(title="Legend")
        
    f1_h1_0.savefig(save_path_pred_true_h1_0, dpi=300, bbox_inches="tight")
    f2_h1_0.savefig(save_path_loss_curve_h1_0, dpi=300, bbox_inches="tight")
    #graph.show()
    graph_h1_0.write_html(save_path_pred_true_html_h1_0)
    #graph2.show()
    graph2_h1_0.write_html(save_path_loss_curve_html_h1_0)
if int(trained_regimes[3]):
    min_val_h2_0, max_val_h2_0 = min(all_values_h2_0), max(all_values_h2_0)
    perfect_prediction_x_h2_0 = [min_val_h2_0, max_val_h2_0]
    ax1_h2_0.plot(perfect_prediction_x_h2_0, perfect_prediction_x_h2_0, '-', label="y=x (perfect prediction)", linewidth=0.7, color='cyan', alpha=0.3)
    graph_h2_0.add_trace(go.Scatter(x=[min_val_h2_0, max_val_h2_0], y=[min_val_h2_0, max_val_h2_0], mode='lines', name="y=x (perfect prediction)", opacity=0.3, marker=dict(color='cyan')))
    ax1_h2_0.legend(title="Legend")
    ax2_h2_0.legend(title="Legend")
        
    f1_h2_0.savefig(save_path_pred_true_h2_0, dpi=300, bbox_inches="tight")
    f2_h2_0.savefig(save_path_loss_curve_h2_0, dpi=300, bbox_inches="tight")
    #graph.show()
    graph_h2_0.write_html(save_path_pred_true_html_h2_0)
    #graph2.show()
    graph2_h2_0.write_html(save_path_loss_curve_html_h2_0)




#print(f"Perferct prediction x: {perfect_prediction_x}")

#plt.show(block=False)


'''infidelities_graph(
    EPOCHS, 
    f"curves/infidelities_{timestamp}.png", 
    infidelity, 
    f'all_architectures/{timestamp}/h_1_0e6/{EPOCHS}_epochs/{HIDDEN_LAYERS}_hidden_layers/{activation}/evaluation_metrics/target_test_h1_0e6.txt', 
    f'all_architectures/{timestamp}/h_1_0e6/{EPOCHS}_epochs/{HIDDEN_LAYERS}_hidden_layers/{activation}/evaluation_metrics/pred_test_h1_0e6.txt', 
    f'all_architectures/{timestamp}/h_1_0/{EPOCHS}_epochs/{HIDDEN_LAYERS}_hidden_layers/{activation}/evaluation_metrics/target_test_h1_0.txt',
    f'all_architectures/{timestamp}/h_1_0/{EPOCHS}_epochs/{HIDDEN_LAYERS}_hidden_layers/{activation}/evaluation_metrics/pred_test_h1_0.txt',
    f'all_architectures/{timestamp}/h_0_5/{EPOCHS}_epochs/{HIDDEN_LAYERS}_hidden_layers/{activation}/evaluation_metrics/target_test_h0_5.txt',
    f'all_architectures/{timestamp}/h_0_5/{EPOCHS}_epochs/{HIDDEN_LAYERS}_hidden_layers/{activation}/evaluation_metrics/pred_test_h0_5.txt',
    f'all_architectures/{timestamp}/h_2_0/{EPOCHS}_epochs/{HIDDEN_LAYERS}_hidden_layers/{activation}/evaluation_metrics/target_test_h2_0.txt',
    f'all_architectures/{timestamp}/h_2_0/{EPOCHS}_epochs/{HIDDEN_LAYERS}_hidden_layers/{activation}/evaluation_metrics/pred_test_h2_0.txt'
    )'''

df_metrics_all.to_csv(csv_file_path, index=False)
global_metrics_dataframe.to_csv('all_architectures_metrics/all_architectures_metrics.csv', index=False)


