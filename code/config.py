import torch

N_spins = 12
BATCH_SIZE = 32
W = 512
EPOCHS = 50

TRAIN_PROPORTION = 0.8
TEST_PROPORTION = 0.05
VALID_PROPORTION = 0.15

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"