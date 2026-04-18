import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split

from config import N_spins, W, device

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_gelu_stack = nn.Sequential(
            nn.Linear(N_spins, W),
            nn.LayerNorm(W),
            nn.GELU(),
            nn.Linear(W, W),
            nn.LayerNorm(W),
            nn.GELU(),
            nn.Linear(W, W),
            nn.LayerNorm(W),
            #nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(W, 1)
        )

    def forward(self, x):
        #print("x is: ",x)
        #x = self.flatten(x)
        logits = self.linear_gelu_stack(x)
        return logits

model = NeuralNetwork().to(device)