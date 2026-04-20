# Python 3.12.3

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split

from dataset_loading import train_dataloader, test_dataloader, valid_dataloader
from architecture import model
from train import train
from valid import valid
from test import test
from config import EPOCHS, device

'''
Appropriate loss functions for this task available in Pytorch:
MSELoss, L1Loss, SmoothL1Loss, HuberLoss

Things to modify:
- optimizer (and set of hyperparameters)
- loss function
- learning rate
- scheduler
- train/test/valid proportion
- batch size
- weight decay
- dropout 
- momentum

Architecture:
- normalization layers
- layer width
- number of layers
- activation functions


'''


print(f"Using {device} device")

# Define model

print(model)




loss_fn = nn.HuberLoss()
optimizer = torch.optim.Rprop(model.parameters(), lr=1e-3)

train_losses = []
valid_losses = []

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    valid_loss = valid(valid_dataloader, model, loss_fn)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
print("Done!")


print(f"Loss on the test set: {test(test_dataloader, model, loss_fn)}.\n")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")