# Python 3.12.3

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split

from dataset_loading import train_dataloader_h0_5, train_dataloader_h1_0, train_dataloader_h2_0, train_dataloader_h1_0e6
from dataset_loading import test_dataloader_h0_5, test_dataloader_h1_0, test_dataloader_h2_0, test_dataloader_h1_0e6
from dataset_loading import valid_dataloader_h0_5, valid_dataloader_h1_0, valid_dataloader_h2_0, valid_dataloader_h1_0e6

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

train_losses_h0_5, valid_losses_h0_5 = [], []
train_losses_h1_0, valid_losses_h1_0 = [], []
train_losses_h2_0, valid_losses_h2_0 = [], []
train_losses_h1_0e6, valid_losses_h1_0e6 = [], []

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    
    train_loss_h0_5, valid_loss_h0_5 = train(train_dataloader_h0_5, model, loss_fn, optimizer), valid(valid_dataloader_h0_5, model, loss_fn)
    train_loss_h1_0, valid_loss_h1_0 = train(train_dataloader_h1_0, model, loss_fn, optimizer), valid(valid_dataloader_h1_0, model, loss_fn)
    train_loss_h2_0, valid_loss_h2_0 = train(train_dataloader_h2_0, model, loss_fn, optimizer), valid(valid_dataloader_h2_0, model, loss_fn)
    train_loss_h1_0e6, valid_loss_h1_0e6 = train(train_dataloader_h1_0e6, model, loss_fn, optimizer), valid(valid_dataloader_h1_0e6, model, loss_fn)
    
    train_losses_h0_5.append(train_loss_h0_5)
    valid_losses_h0_5.append(valid_loss_h0_5)
    
    train_losses_h1_0.append(train_loss_h1_0)
    valid_losses_h1_0.append(valid_loss_h1_0)
    
    train_losses_h2_0.append(train_loss_h2_0)
    valid_losses_h2_0.append(valid_loss_h2_0)
    
    train_losses_h1_0e6.append(train_loss_h1_0e6)
    valid_losses_h1_0e6.append(valid_loss_h1_0e6)
    
print("Done!")


print(f"Loss on the test set (h=0.5): {test(test_dataloader_h0_5, model, loss_fn)}.\n")
print(f"Loss on the test set (h=1.0): {test(test_dataloader_h1_0, model, loss_fn)}.\n")
print(f"Loss on the test set (h=2.0): {test(test_dataloader_h2_0, model, loss_fn)}.\n")
print(f"Loss on the test set (h=1.0e6): {test(test_dataloader_h1_0e6, model, loss_fn)}.\n")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")