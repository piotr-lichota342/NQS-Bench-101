import torch
import os
from datetime import datetime

N_spins = 12
BATCH_SIZE = 40
W = 4096
EPOCHS = int(os.environ.get("EPOCHS", 3))
HIDDEN_LAYERS = int(os.environ.get("HIDDEN_LAYERS", 2))
INPUT_SIZE = (32,12)
ACT_FUNCTION = int(os.environ.get("ACT_FUNCTION", 0))
DECIMAL_PLACES_METRICS = 4
SAVING_WEIGHTS = False
TRAIN_PROPORTION = 0.7
TEST_PROPORTION = 0.2
VALID_PROPORTION = 0.1
DATASET_SIZE = 4096
TIMESTAMP = str(os.environ.get("TIMESTAMP", None))

trained_regimes = os.environ.get("trained_regimes", None)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)


