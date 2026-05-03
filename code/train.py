import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split

from config import device

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    target_points, pred_points, losses = [], [], []
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        #print("shape:  ", X.shape)
        

        # Compute prediction error
        pred = model(X)
        target_points.append(y)
        pred_points.append(pred)
        y = y.unsqueeze(1)
        #print("The prediction is: ", pred)
        loss = loss_fn(pred, y)
        losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    target_points, pred_points = torch.cat(target_points).flatten().numpy(), torch.cat(pred_points).detach().numpy()
    losses = [loss.item() for loss in losses]
    print(f"Train losses: {losses}")
            
    return total_loss / len(dataloader), target_points, pred_points