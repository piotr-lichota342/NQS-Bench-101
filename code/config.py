import torch

N_spins = 12
BATCH_SIZE = 32
W = 512
EPOCHS = 5
HIDDEN_LAYERS = 4
INPUT_SIZE = (32,12)

TRAIN_PROPORTION = 0.8
TEST_PROPORTION = 0.05
VALID_PROPORTION = 0.15

trained_regimes = {
    "h=0.5":True,
    "h=1.0":False,
    "h=2.0":False,
    "h=1.0e-6":False
}

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)

