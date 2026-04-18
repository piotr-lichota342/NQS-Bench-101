import matplotlib.pyplot as plt
from main import train_losses, valid_losses, loss_fn
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"curves/loss_curve_{timestamp}.png"

print(f"Train loss length: {train_losses}")

plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel(f"Loss ({loss_fn.__class__.__name__})")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

