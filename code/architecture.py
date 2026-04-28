import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split
from torch.nn import Parameter
from torchinfo import summary

from config import N_spins, W, device, HIDDEN_LAYERS, INPUT_SIZE

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(N_spins, W))
        layers.append(nn.LayerNorm(W))
        layers.append(nn.GELU())
        
        for hl in range(HIDDEN_LAYERS):
            layers.append(nn.Linear(W, W))
            layers.append(nn.LayerNorm(W))
            layers.append(nn.GELU())
            
        layers.append(nn.Linear(W, 1))
        
        self.linear_gelu_stack = nn.Sequential(
            *layers
        )
        

    def forward(self, x):
        #print("x is: ",x)
        #x = self.flatten(x)
        logits = self.linear_gelu_stack(x)
        return logits

model_h0_5 = NeuralNetwork().to(device)
model_h1_0 = NeuralNetwork().to(device)
model_h2_0 = NeuralNetwork().to(device)
model_h1_0e6 = NeuralNetwork().to(device)


summary_str = str(summary(model_h0_5, INPUT_SIZE))
print(summary_str)
#print(model_h0_5.input_shape())
#print(model_h0_5.Torch)