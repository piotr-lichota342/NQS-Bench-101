import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split
from torch.nn import Parameter
from torchinfo import summary
from config import WIDTH_SEQUENCE

from config import N_spins, W, device, HIDDEN_LAYERS, INPUT_SIZE, ACT_FUNCTION, CUSTOM_ARCH, WIDTH_SEQUENCE



activation_f = None

match ACT_FUNCTION:
    case 0:
        activation_f = nn.GELU()
    case 1:
        activation_f = nn.Tanh()
    case 2:
        activation_f = nn.ReLU()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        if not CUSTOM_ARCH:
            layers = []
            layers.append(nn.Linear(N_spins, W))
            layers.append(nn.LayerNorm(W))
            layers.append(activation_f)
            
            '''layers.append(nn.Linear(W, 512))
            layers.append(nn.LayerNorm(512))
            layers.append(activation_f)'''
            
            
            for hl in range(HIDDEN_LAYERS):
                layers.append(nn.Linear(W, W))
                layers.append(nn.LayerNorm(W))
                layers.append(activation_f)
                
            layers.append(nn.Linear(W, 1))
            
            self.linear_gelu_stack = nn.Sequential(
                *layers
            )
        else:
            layers = []
            width_sequence = WIDTH_SEQUENCE.split(",")
            width_sequence = [int(w) for w in width_sequence]
            layers.append(nn.Linear(N_spins, width_sequence[0]))
            layers.append(nn.LayerNorm(width_sequence[0]))
            layers.append(activation_f)
            
            for hl in range(0,len(width_sequence)-1):
                layers.append(nn.Linear(width_sequence[hl], width_sequence[hl+1]))
                layers.append(nn.LayerNorm(width_sequence[hl+1]))
                layers.append(activation_f)
                
            layers.append(nn.Linear(width_sequence[-1], 1))
            
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


#summary_str = str(summary(model_h0_5, INPUT_SIZE))
#print(summary_str)
#print(model_h0_5.input_shape())
#print(model_h0_5.Torch)
