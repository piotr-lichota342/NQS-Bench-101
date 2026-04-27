import matplotlib.pyplot as plt

from main import train_losses_h1_0e6, train_losses_h0_5, train_losses_h1_0, train_losses_h2_0
from main import valid_losses_h1_0e6, valid_losses_h0_5, valid_losses_h1_0, valid_losses_h2_0
from main import loss_fn, y_true_h1_0e6, y_pred_h1_0e6, y_true_h0_5, y_pred_h0_5, y_true_h1_0, y_pred_h1_0, y_true_h2_0, y_pred_h2_0
from config import trained_regimes, EPOCHS, BATCH_SIZE, W, TEST_PROPORTION, TRAIN_PROPORTION, VALID_PROPORTION
import plotly.graph_objects as go

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path_loss_curve = f"curves\\loss_curve_{timestamp}.png"
save_path_pred_true = f"curves\\pred_true_curve_{timestamp}.png"

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
    graph = go.Figure(data=go.Scatter(x=y_true_h0_5, y=y_pred_h0_5, mode='markers'))
    graph.show()
    graph.write_html(f"curves\\plot_true_pred_{timestamp}.html")
    
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





