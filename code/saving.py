import matplotlib.pyplot as plt

from main import train_losses_h1_0e6, train_losses_h0_5, train_losses_h1_0, train_losses_h2_0
from main import valid_losses_h1_0e6, valid_losses_h0_5, valid_losses_h1_0, valid_losses_h2_0
from main import loss_fn, y_true, y_pred
from config import trained_regimes, EPOCHS, BATCH_SIZE, W, TEST_PROPORTION, TRAIN_PROPORTION, VALID_PROPORTION

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"curves\\loss_curve_{timestamp}.png"

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

if trained_regimes["h=1.0e-6"]:
    plt.plot(train_losses_h1_0e6, label="Train Loss (h=1.0e-6)")
    plt.plot(valid_losses_h1_0e6, label="Valid Loss (h=1.0e-6)")

if trained_regimes["h=0.5"]:
    #plt.plot(train_losses_h0_5, label="Train Loss (h=0.5)")
    #plt.plot(valid_losses_h0_5, label="Valid Loss (h=0.5)")
    plt.plot(y_true, y_pred, 'rs', markersize=1)
    plt.show()
    
if trained_regimes["h=1.0"]:
    plt.plot(train_losses_h1_0, label="Train Loss (h=1.0)")
    plt.plot(valid_losses_h1_0, label="Valid Loss (h=1.0)")
    
if trained_regimes["h=2.0"]:
    plt.plot(train_losses_h2_0, label="Train Loss (h=2.0)")
    plt.plot(valid_losses_h2_0, label="Valid Loss (h=2.0)")
    
'''
plt.xlabel("Epoch")
plt.ylabel(f"Loss ({loss_fn.__class__.__name__})")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

'''
plt.savefig(save_path, dpi=300, bbox_inches="tight")

