import matplotlib.pyplot as plt

from main import train_losses_h1_0e6, train_losses_h0_5, train_losses_h1_0, train_losses_h2_0
from main import valid_losses_h1_0e6, valid_losses_h0_5, valid_losses_h1_0, valid_losses_h2_0
from main import loss_fn, y_true_h1_0e6, y_pred_h1_0e6, y_true_h0_5, y_pred_h0_5, y_true_h1_0, y_pred_h1_0, y_true_h2_0, y_pred_h2_0, optimizer_h0_5, total_training_time
from config import trained_regimes, EPOCHS, BATCH_SIZE, W, TEST_PROPORTION, TRAIN_PROPORTION, VALID_PROPORTION, HIDDEN_LAYERS, INPUT_SIZE, device, trained_regimes
from architecture import summary_str
import plotly.graph_objects as go
import plotly.express as px
import os
import pandas as pd
from main import r2, var, rmse, mse, mae, mape, smape, rmse, rmsle, msle, hell_dist

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(os.path.join('training_logs', f"{timestamp}"))
os.makedirs(os.path.join(f'training_logs\\{timestamp}', "curves"))
os.makedirs(os.path.join(f'training_logs\\{timestamp}', "evaluation_metrics"))
save_path_loss_curve = f"training_logs\\{timestamp}\\curves\\loss_curve_{timestamp}.png"
save_path_pred_true = f"training_logs\\{timestamp}\\curves\\pred_true_curve_{timestamp}.png"

save_path_loss_curve_html = f"training_logs\\{timestamp}\\curves\\loss_curve_{timestamp}.html"
save_path_pred_true_html = f"training_logs\\{timestamp}\\curves\\pred_true_curve_{timestamp}.html"





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


if trained_regimes["h=1.0e-6"]:
    ax2.plot(train_losses_h1_0e6, label="Train loss (h=1.0e-6)")
    ax2.plot(valid_losses_h1_0e6, label="Valid loss (h=1.0e-6)")
    ax1.plot(y_true_h1_0e6, y_pred_h1_0e6, 'ms', markersize=1, label="h=1.0e-6")
    

if trained_regimes["h=0.5"]:
    
    #ax1 = f1.add_axes(train_losses_h0_5)
    ax2.plot(train_losses_h0_5, label="Train loss (h=0.5)")
    ax2.plot(valid_losses_h0_5, label="Valid loss (h=0.5)")
    ax1.plot(y_true_h0_5, y_pred_h0_5, 'bs', markersize=1, label="h=0.5")
    graph = go.Figure()
    graph.add_trace(go.Scatter(x=y_true_h0_5, y=y_pred_h0_5, mode='markers', name="h=0.5", marker=dict(color='red')))
    graph.update_layout(
        xaxis_title="True Amplitude",
        yaxis_title="Predicted Amplitude",
        title='True vs Predicted Amplitude',
        showlegend=True
    )
    graph.show()
    graph.write_html(save_path_pred_true_html)
    
    graph2 = go.Figure()
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=valid_losses_h0_5, mode='lines', name="Valid loss (h=0.5)", line=dict(color='magenta')))
    graph2.add_trace(go.Scatter(x=[e for e in range(1,EPOCHS+1,1)], y=train_losses_h0_5, mode='lines', name="Train loss (h=0.5)", line=dict(color='blue')))
    graph2.update_layout(
        xaxis_title="Epochs",
        yaxis_title=f"{loss_fn.__class__.__name__}",
        title='Train vs Valid Loss',
        showlegend=True
    )
    graph2.show()
    graph2.write_html(save_path_loss_curve_html)
    
    
    
    
    
    
    #f1.savefig(save_path_pred_true, dpi=300, bbox_inches="tight")
    
    
    
if trained_regimes["h=1.0"]:
    ax2.plot(train_losses_h1_0, label="Train Loss (h=1.0)")
    ax2.plot(valid_losses_h1_0, label="Valid Loss (h=1.0)")
    ax1.plot(y_true_h1_0, y_pred_h1_0, 'rs', markersize=1, label="h=1.0")
    
if trained_regimes["h=2.0"]:
    ax2.plot(train_losses_h2_0, label="Train Loss (h=2.0)")
    ax2.plot(valid_losses_h2_0, label="Valid Loss (h=2.0)")
    ax1.plot(y_true_h2_0, y_pred_h2_0, 'gs', markersize=1, label="h=2.0")
    
ax1.legend()
ax2.legend()
    
f1.savefig(save_path_pred_true, dpi=300, bbox_inches="tight")
f2.savefig(save_path_loss_curve, dpi=300, bbox_inches="tight")
plt.show()

dict_optimizer = optimizer_h0_5.param_groups[0]
dict_optimizer.pop('params')
dict_optimizer = optimizer_h0_5.param_groups[0]

metrics_data = {
    'model_summary':summary_str,
    'training_time':total_training_time,
    'epochs':EPOCHS,
    'input_size':INPUT_SIZE,
    'batch_size': BATCH_SIZE,
    'network_width': W,
    'hidden_layers': HIDDEN_LAYERS,
    'train_proportion':TRAIN_PROPORTION,
    'test_proportion':TEST_PROPORTION,
    'valid_proportion':VALID_PROPORTION,
    'device':str(device),
    'optimizer_name':optimizer_h0_5.__class__.__name__,
    'optimizer_params':str(dict_optimizer),
    'loss_fn':str(loss_fn.__class__.__name__),
    'bias':None,
    'avr_res':None,
    'MBE':None,
    'R2':r2,
    'RSS':None,
    'TSS':None,
    'adjusted_R2':None,
    'MSE':mse,
    'RMSE':rmse,
    'MAE':mae,
    'MAPE':mape,
    'wMAPE':None,
    'sMAPE':smape,
    'MSLE':msle,
    'RMSLE':rmsle,
    'AIC':None,
    'BIC':None,
    'ESS':None,
    'hellinger_dist':str(hell_dist)
    
}

df = pd.DataFrame(metrics_data)
csv_file_path = f'training_logs\\{timestamp}\\evaluation_metrics\\metrics_h0_5.csv'
df.to_csv(csv_file_path, index=False)





