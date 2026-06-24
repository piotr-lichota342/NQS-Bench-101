import torch
import os
from datetime import datetime

N_spins = 12
BATCH_SIZE = 40

EPOCHS = int(os.environ.get("EPOCHS", 3))

INPUT_SIZE = (32,12)
ACT_FUNCTION = int(os.environ.get("ACT_FUNCTION", 0))
DECIMAL_PLACES_METRICS = 4
SAVING_WEIGHTS = False
TRAIN_PROPORTION = 0.7
TEST_PROPORTION = 0.2
VALID_PROPORTION = 0.1
DATASET_SIZE = 4096
TIMESTAMP = str(os.environ.get("TIMESTAMP", None))
EXPONENTIAL_LR = True
CUSTOM_ARCH = True
WIDTH_SEQUENCE = str(os.environ.get("WIDTH_SEQUENCE", None))

trained_regimes = os.environ.get("trained_regimes", None)
if CUSTOM_ARCH:
    W = os.environ.get("W", 16)
    HIDDEN_LAYERS = "custom"
else:
    W = int(os.environ.get("W", 16))
    HIDDEN_LAYERS = int(os.environ.get("HIDDEN_LAYERS", 2))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)


