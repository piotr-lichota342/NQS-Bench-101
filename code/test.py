import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split

from config import device

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    target_points, pred_points = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            #y = y.view(-1, 1)
            target_points.append(y)

            pred = model(X)
            pred_points.append(pred)
            y = y.unsqueeze(1)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    #correct /= size
    #print(f"Avg loss: {test_loss:>8f} \n")
    #print(f"Test target points: {torch.cat(pred_points).flatten().numpy()}")
    target_points, pred_points = torch.cat(target_points).flatten().numpy(), torch.cat(pred_points).flatten().numpy()
    return test_loss, target_points, pred_points