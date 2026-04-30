import matplotlib.pyplot as plt

from main import train_losses_h1_0e6, train_losses_h0_5, train_losses_h1_0, train_losses_h2_0
from main import model_h0_5, model_h1_0, model_h2_0, model_h1_0e6
from main import valid_losses_h1_0e6, valid_losses_h0_5, valid_losses_h1_0, valid_losses_h2_0
from main import loss_fn, y_true_h1_0e6, y_pred_h1_0e6, y_true_h0_5, y_pred_h0_5, y_true_h1_0, y_pred_h1_0, y_true_h2_0, y_pred_h2_0, optimizer_h0_5, total_training_time
from config import trained_regimes, EPOCHS, BATCH_SIZE, W, TEST_PROPORTION, TRAIN_PROPORTION, VALID_PROPORTION, HIDDEN_LAYERS, INPUT_SIZE, device, trained_regimes
from architecture import summary_str
import plotly.graph_objects as go
import plotly.express as px
import torch
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_log_error, mean_squared_log_error
import pandas as pd
from main import r2, var, rmse, mse, mae, mape, smape, rmse, rmsle, msle, hell_dist

from datetime import datetime

def smape(a, f):
    a_np, f_np = np.array(a), np.array(f)
    return 1/len(a) * np.sum(2 * np.abs(f_np-a_np) / (np.abs(a_np) + np.abs(f_np))*100)

def hellinger_distance(p,q):
    #Turning into probabilities
    p_prob, q_prob = [np.abs(a) for a in p], [np.abs(a) for a in q]
    p_prob, q_prob = [a/np.sum(p) for a in p_prob], [a/np.sum(q) for a in q_prob]
    #print(f"p_prob, q prob: {p_prob}, {q_prob}")
    final_result = 0   
    for i in range(len(p_prob)):
        diff = (p_prob[i])**(0.5) - (q_prob[i])**(0.5)
        final_result += diff**2
    final_result = (final_result**(0.5)) * (1/(2**(0.5)))
    final_result = round(final_result,3)
    
    return final_result

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

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join('training_logs', f"{timestamp}"))
os.makedirs(os.path.join(f'training_logs\\{timestamp}', "curves"))
os.makedirs(os.path.join(f'training_logs\\{timestamp}', "evaluation_metrics"))
os.makedirs(os.path.join(f'training_logs\\{timestamp}', "model_weights"))
save_path_loss_curve = f"training_logs\\{timestamp}\\curves\\loss_curve.png"
save_path_pred_true = f"training_logs\\{timestamp}\\curves\\pred_true_curve.png"

save_path_loss_curve_html = f"training_logs\\{timestamp}\\curves\\loss_curve.html"
save_path_pred_true_html = f"training_logs\\{timestamp}\\curves\\pred_true_curve.html"

model_weights_h0_5_path = f"training_logs\\{timestamp}\\model_weights\\model_weights_h0_5.pth"
model_weights_h1_0_path = f"training_logs\\{timestamp}\\model_weights\\model_weights_h1_0.pth"
model_weights_h2_0_path = f"training_logs\\{timestamp}\\model_weights\\model_weights_h2_0.pth"
model_weights_h1_0e6_path = f"training_logs\\{timestamp}\\model_weights\\model_weights_h1_0e6.pth"

csv_file_path = f'training_logs\\{timestamp}\\evaluation_metrics\\metrics.csv'

df_metrics = pd.DataFrame()

metrics_data = {
    'regime': None,
    'model_summary': None, # summary_str
    'training_time (s)': total_training_time,
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
    'bias': None,
    'avr_res': None,
    'MBE': None,
    'R2': r2,
    'RSS': None,
    'TSS': None,
    'adjusted_R2': None,
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'MAPE': mape,
    'wMAPE': None,
    'sMAPE': smape,
    'MSLE': msle,
    'RMSLE': rmsle,
    'AIC': None,
    'BIC': None,
    'ESS': None,
    'hellinger_dist': str(hell_dist)
    
}
r2 = round(r2_score(y_pred_h0_5, y_true_h0_5),4)
var = round(np.var(y_pred_h0_5),4)
rmse = round(root_mean_squared_error(y_true_h0_5, y_pred_h0_5),4)
mse = round(mean_squared_error(y_true_h0_5, y_pred_h0_5),4)
mae = round(mean_absolute_error(y_true_h0_5, y_pred_h0_5),4)
mape = round(mean_absolute_percentage_error(y_true_h0_5, y_pred_h0_5),4)
smape = round(smape(y_true_h0_5, y_pred_h0_5),4)
rmse = round(root_mean_squared_error(y_true_h0_5, y_pred_h0_5),4)
rmsle = round(root_mean_squared_log_error(y_true_h0_5, y_pred_h0_5),4)
msle = round(mean_squared_log_error(y_true_h0_5, y_pred_h0_5),4)
hell_dist = hellinger_distance(y_pred_h0_5, y_true_h0_5)


#print(f"Train loss length: {train_losses}")

f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111)
ax2 = f2.add_subplot(111)
ax1.set_xlabel("True Amplitudes")
ax1.set_ylabel("Predicted Amplitudes")
ax1.set_title("True vs Predicted Amplitudes")

ax1.grid(True)
ax2.set_xlabel("Epoch")
ax2.set_ylabel(f"Loss ({loss_fn.__class__.__name__})")
ax2.set_title("Training vs Validation Loss")

ax2.grid(True)
graph = go.Figure()
graph2 = go.Figure()

graph.update_layout(
        xaxis_title="True Amplitude",
        yaxis_title="Predicted Amplitude",
        title='True vs Predicted Amplitude',
        showlegend=True,
        legend_title_text="Legend"
    )

graph2.update_layout(
        xaxis_title="Epochs",
        yaxis_title=f"{loss_fn.__class__.__name__}",
        title='Train vs Valid Loss',
        showlegend=True,
        legend_title_text="Legend"
    )

perfect_prediction_x = [0,max(max(y_true_h1_0e6), max(y_true_h1_0e6))]
ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
graph.add_trace(go.Scatter(x=perfect_prediction_x, y=perfect_prediction_x, mode='lines', name="y = x (perfect prediction)", marker=dict(color='cyan')))


if trained_regimes["h=1.0e-6"]:
    
    metrics_h1_0e6 = metrics_data
    
    ax2.plot(train_losses_h1_0e6, label="Train loss (h=1.0e-6)")
    ax2.plot(valid_losses_h1_0e6, label="Valid loss (h=1.0e-6)")
    ax1.plot(y_true_h1_0e6, y_pred_h1_0e6, 's', markersize=1, label="h=1.0e-6")
    
    
    
    graph.add_trace(go.Scatter(x=y_true_h1_0e6, y=y_pred_h1_0e6, mode='markers', name="h=1.0e-6"))
    

    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=valid_losses_h1_0e6, mode='lines', name="Valid loss (h=1.0e-6)"))
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=train_losses_h1_0e6, mode='lines', name="Train loss (h=1.0e-6)"))
    
    #graph2.show()
    #graph2.write_html(save_path_loss_curve_html)
    torch.save(model_h1_0e6.state_dict(), model_weights_h1_0e6_path)
    

if trained_regimes["h=0.5"]:
    #perfect_prediction_x = [0,max(max(y_true_h0_5), max(y_pred_h0_5))]
    
    #ax1 = f1.add_axes(train_losses_h0_5)
    ax2.plot(train_losses_h0_5, label="Train loss (h=0.5)")
    ax2.plot(valid_losses_h0_5, label="Valid loss (h=0.5)")
    ax1.plot(y_true_h0_5, y_pred_h0_5, 's', markersize=1, label="h=0.5")
    
    #ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
    #graph = go.Figure()
    graph.add_trace(go.Scatter(x=y_true_h0_5, y=y_pred_h0_5, mode='markers', name="h=0.5"))
    #graph.add_trace(go.Scatter(x=perfect_prediction_x, y=perfect_prediction_x, mode='lines', name="y = x (perfect prediction)", marker=dict(color='cyan')))
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=valid_losses_h0_5, mode='lines', name="Valid loss (h=0.5)"))
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=train_losses_h0_5, mode='lines', name="Train loss (h=0.5)"))
    
    
    torch.save(model_h0_5.state_dict(), model_weights_h0_5_path)
    
    
    #f1.savefig(save_path_pred_true, dpi=300, bbox_inches="tight")
    
    
    
if trained_regimes["h=1.0"]:
    #perfect_prediction_x = [0,max(max(y_true_h1_0), max(y_pred_h1_0))]
    ax2.plot(train_losses_h1_0, label="Train Loss (h=1.0)")
    ax2.plot(valid_losses_h1_0, label="Valid Loss (h=1.0)")
    ax1.plot(y_true_h1_0, y_pred_h1_0, 's', markersize=1, label="h=1.0")
    
    
    #ax1.plot(perfect_prediction_x, perfect_prediction_x, '-', label="y = x (perfect prediction)", linewidth=0.5, color='cyan')
    #graph = go.Figure()
    graph.add_trace(go.Scatter(x=y_true_h1_0, y=y_pred_h1_0, mode='markers', name="h=1.0"))
    #graph.add_trace(go.Scatter(x=perfect_prediction_x, y=perfect_prediction_x, mode='lines', name="y = x (perfect prediction)", marker=dict(color='cyan')))
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=valid_losses_h1_0, mode='lines', name="Valid loss (h=1.0)"))
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=train_losses_h1_0, mode='lines', name="Train loss (h=1.0)"))
    
    
    torch.save(model_h1_0.state_dict(), model_weights_h1_0_path)
    
if trained_regimes["h=2.0"]:
    #perfect_prediction_x = [0,max(max(y_true_h2_0), max(y_pred_h2_0))]
    ax2.plot(train_losses_h2_0, label="Train Loss (h=2.0)")
    ax2.plot(valid_losses_h2_0, label="Valid Loss (h=2.0)")
    ax1.plot(y_true_h2_0, y_pred_h2_0, 's', markersize=1, label="h=2.0")
    
    graph.add_trace(go.Scatter(x=y_true_h2_0, y=y_pred_h2_0, mode='markers', name="h=2.0"))
    #graph.add_trace(go.Scatter(x=perfect_prediction_x, y=perfect_prediction_x, mode='lines', name="y = x (perfect prediction)", marker=dict(color='cyan')))
    
    #graph.show()
    #graph.write_html(save_path_pred_true_html)
    
    #graph2 = go.Figure()
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=valid_losses_h2_0, mode='lines', name="Valid loss (h=2.0)"))
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=train_losses_h2_0, mode='lines', name="Train loss (h=2.0)"))
    
    
    torch.save(model_h2_0.state_dict(), model_weights_h2_0_path)
    
ax1.legend(title="Legend")
ax2.legend(title="Legend")
    
f1.savefig(save_path_pred_true, dpi=300, bbox_inches="tight")
f2.savefig(save_path_loss_curve, dpi=300, bbox_inches="tight")
plt.show()

dict_optimizer = optimizer_h0_5.param_groups[0]
dict_optimizer.pop('params')
dict_optimizer = optimizer_h0_5.param_groups[0]



graph.show()
graph.write_html(save_path_pred_true_html)
graph2.show()
graph2.write_html(save_path_loss_curve_html)


df_metrics.to_csv(csv_file_path, index=False)





