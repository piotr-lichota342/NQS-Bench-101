import torch

N_spins = 12
BATCH_SIZE = 32
W = 512
EPOCHS = 10
HIDDEN_LAYERS = 4
INPUT_SIZE = (32,12)
DECIMAL_PLACES_METRICS = 4

TRAIN_PROPORTION = 0.8
TEST_PROPORTION = 0.05
VALID_PROPORTION = 0.15

trained_regimes = {
    "h=0.5":True,
    "h=1.0":True,
    "h=2.0":True,
    "h=1.0e-6":True
}

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)

