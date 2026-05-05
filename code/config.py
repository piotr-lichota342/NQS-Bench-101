import torch

N_spins = 12
BATCH_SIZE = 32
W = 512
EPOCHS = 3
HIDDEN_LAYERS = 4
INPUT_SIZE = (32,12)
DECIMAL_PLACES_METRICS = 4
SAVING_WEIGHTS = False
TRAIN_PROPORTION = 0.75
TEST_PROPORTION = 0.25
VALID_PROPORTION = 0.0
DATASET_SIZE = 4096

trained_regimes = {
    "h=0.5":True,
    "h=1.0":True,
    "h=2.0":True,
    "h=10⁻⁶":True
}

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)

