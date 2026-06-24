import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split
import numpy as np

from config import device

def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    if num_batches==0:
        return np.nan, [], []
    model.eval()
    test_loss = 0
    target_points, pred_points, losses = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            #y = y.view(-1, 1)

            pred = model(X)
            target_points.append(y)
            pred_points.append(pred)
            y = y.unsqueeze(1)
            test_loss += loss_fn(pred, y).item()
            losses.append(loss_fn(pred, y).item())
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_loss = test_loss / num_batches
    #correct /= size
    #print(f"Valid loss: {avg_loss:>8f} \n")
    target_points, pred_points = torch.cat(target_points).flatten().cpu().numpy(), torch.cat(pred_points).flatten().cpu().numpy()
    return avg_loss, target_points, pred_points
