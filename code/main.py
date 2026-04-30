# Python 3.12.3

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split
import time
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
import numpy as np
from keras import backend as K
import sklearn
import astropy

from dataset_loading import train_dataloader_h0_5, train_dataloader_h1_0, train_dataloader_h2_0, train_dataloader_h1_0e6
from dataset_loading import test_dataloader_h0_5, test_dataloader_h1_0, test_dataloader_h2_0, test_dataloader_h1_0e6
from dataset_loading import valid_dataloader_h0_5, valid_dataloader_h1_0, valid_dataloader_h2_0, valid_dataloader_h1_0e6

from dataset_loading import dataloader_h0_5, dataloader_h1_0, dataloader_h1_0e6, dataloader_h2_0

from architecture import model_h0_5, model_h1_0, model_h2_0, model_h1_0e6
from train import train
from valid import valid
from test import test
from config import EPOCHS, device, trained_regimes
#from distancia import Hellinger

"""
Evaluation metrics (https://developer.nvidia.com/blog/a-comprehensive-overview-of-regression-evaluation-metrics/):
- sum of residuals (bias)
- the average residual
- mean bias error (MBE)
- R-squared (coefficient of determination)
- The residual sum of squares (RSS)
- Total sum of squares (TSS)
- adjusted R² (penalizes adding features that are not useful for predicting the target)
- Mean squared error (MSE)
- Root mean squared error (RMSE)
- Mean absolute error (MAE)
- Mean absolute percentage error (MAPE)
- Weighted mean absolute percentage error (wMAPE)
- Symmetric mean absolute percentage error (sMAPE)
- Mean squared log error (MSLE)
- Root mean squared log error (RMSLE)
- Akaike information criterion (AIC)
- Bayesian optimization criterion (BIC)
- Explained sum of squares (ESS)

"""


    




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

print(model_h0_5)




loss_fn = nn.MSELoss()

optimizer_h0_5 = torch.optim.Adam(model_h0_5.parameters(), lr=1e-3) if trained_regimes["h=0.5"] else None
optimizer_h1_0 = torch.optim.Adam(model_h1_0.parameters(), lr=1e-3) if trained_regimes["h=1.0"] else None
optimizer_h2_0 = torch.optim.Adam(model_h2_0.parameters(), lr=1e-3) if trained_regimes["h=2.0"] else None
optimizer_h1_0e6 = torch.optim.Adam(model_h1_0e6.parameters(), lr=1e-3) if trained_regimes["h=1.0e-6"] else None

#print(optimizer_h0_5.get_config())



train_losses_h0_5, valid_losses_h0_5 = [], []
train_losses_h1_0, valid_losses_h1_0 = [], []
train_losses_h2_0, valid_losses_h2_0 = [], []
train_losses_h1_0e6, valid_losses_h1_0e6 = [], []

start_training_time = time.time()

for t in range(EPOCHS):
    print(f"Epoch {t+1}/{EPOCHS}\n===============================")
    
    if trained_regimes["h=0.5"]:
        print("Losses for h=0.5:\n")
        train_loss_h0_5, valid_loss_h0_5 = train(train_dataloader_h0_5, model_h0_5, loss_fn, optimizer_h0_5), valid(valid_dataloader_h0_5, model_h0_5, loss_fn)
        train_losses_h0_5.append(train_loss_h0_5)
        valid_losses_h0_5.append(valid_loss_h0_5)
        
    if trained_regimes["h=1.0"]:
        print("Losses for h=1.0:\n")
        train_loss_h1_0, valid_loss_h1_0 = train(train_dataloader_h1_0, model_h1_0, loss_fn, optimizer_h1_0), valid(valid_dataloader_h1_0, model_h1_0, loss_fn)
        train_losses_h1_0.append(train_loss_h1_0)
        valid_losses_h1_0.append(valid_loss_h1_0)
    if trained_regimes["h=2.0"]:
        print("Losses for h=2.0:\n")
        train_loss_h2_0, valid_loss_h2_0 = train(train_dataloader_h2_0, model_h2_0, loss_fn, optimizer_h2_0), valid(valid_dataloader_h2_0, model_h2_0, loss_fn)
        train_losses_h2_0.append(train_loss_h2_0)
        valid_losses_h2_0.append(valid_loss_h2_0)
    if trained_regimes["h=1.0e-6"]:
        print("Losses for h=1.0e-6:\n")
        train_loss_h1_0e6, valid_loss_h1_0e6 = train(train_dataloader_h1_0e6, model_h1_0e6, loss_fn, optimizer_h1_0e6), valid(valid_dataloader_h1_0e6, model_h1_0e6, loss_fn)
        train_losses_h1_0e6.append(train_loss_h1_0e6)
        valid_losses_h1_0e6.append(valid_loss_h1_0e6)    
 
end_training_time = time.time() 
total_training_time = end_training_time - start_training_time

   
print("Done!")

y_pred_h1_0e6, y_true_h1_0e6 = [], []
y_pred_h0_5, y_true_h0_5 = [], []
y_pred_h1_0, y_true_h1_0 = [], []
y_pred_h2_0, y_true_h2_0 = [], []
   

if trained_regimes["h=0.5"]:
    print(f"Loss on the test set (h=0.5): {test(test_dataloader_h0_5, model_h0_5, loss_fn)}.\n")
    #torch.save(model_h0_5.state_dict(), "saved_models\\model_h0_5.pth")
    #print("Saved PyTorch Model State to saved_models\\model_h0_5.pth")
    size = len(dataloader_h0_5.dataset)
    num_batches = len(dataloader_h0_5)
    
    with torch.no_grad():
        for X, y in dataloader_h0_5.dataset:
            #X, y = X.to(device), y.to(device)
            #y = y.view(-1, 1)

            pred = model_h0_5(X)
            #y = y.unsqueeze(1)
            y_pred_h0_5.append(pred)
            y_true_h0_5.append(y)
    y_pred_h0_5 = [x.item() for x in y_pred_h0_5]
    y_true_h0_5 = [x.item() for x in y_true_h0_5]
    #print("types y_pred, y_true: ",(y_pred), (y_true))
    mse = round(mean_squared_error(y_true_h0_5, y_pred_h0_5),4)
    mae = round(mean_absolute_error(y_true_h0_5, y_pred_h0_5),4)
    

    # Create an instance of the Hellinger class
    #hellinger_dist = Hellinger()

    # Calculate the Hellinger distance between the two distributions
    #distance = hellinger_dist.calculate(y_pred_h0_5, y_true_h0_5)

    # Print the result
    #print(f"The Hellinger distance between the two distributions is (h=0.5): {hellinger_distance(y_pred_h0_5, y_true_h0_5)}")
if trained_regimes["h=1.0"]:
    print(f"Loss on the test set (h=1.0): {test(test_dataloader_h1_0, model_h1_0, loss_fn)}.\n")
    torch.save(model_h1_0.state_dict(), "saved_models\\model_h1_0.pth")
    print("Saved PyTorch Model State to saved_models\\model_h1_0.pth")
    size = len(dataloader_h1_0.dataset)
    num_batches = len(dataloader_h1_0)
    
    with torch.no_grad():
        for X, y in dataloader_h1_0.dataset:
            #X, y = X.to(device), y.to(device)
            #y = y.view(-1, 1)

            pred = model_h1_0(X)
            #y = y.unsqueeze(1)
            y_pred_h1_0.append(pred)
            y_true_h1_0.append(y)
    y_pred_h1_0 = [x.item() for x in y_pred_h1_0]
    y_true_h1_0 = [x.item() for x in y_true_h1_0]
    #print("types y_pred, y_true: ",(y_pred), (y_true))
    print(f"R square (h=1.0): {round(r2_score(y_pred_h1_0, y_true_h1_0),3)}")
if trained_regimes["h=2.0"]:
    print(f"Loss on the test set (h=2.0): {test(test_dataloader_h2_0, model_h2_0, loss_fn)}.\n")
    torch.save(model_h2_0.state_dict(), "saved_models\\model_h2_0.pth")
    size = len(dataloader_h2_0.dataset)
    num_batches = len(dataloader_h2_0)
    
    with torch.no_grad():
        for X, y in dataloader_h2_0.dataset:
            #X, y = X.to(device), y.to(device)
            #y = y.view(-1, 1)

            pred = model_h2_0(X)
            #y = y.unsqueeze(1)
            y_pred_h2_0.append(pred)
            y_true_h2_0.append(y)
    y_pred_h2_0 = [x.item() for x in y_pred_h2_0]
    y_true_h2_0 = [x.item() for x in y_true_h2_0]
    #print("types y_pred, y_true: ",(y_pred), (y_true))
    print(f"R square (h=2.0): {round(r2_score(y_pred_h2_0, y_true_h2_0),3)}")
    print("Saved PyTorch Model State to saved_models\\model_h2_0.pth")
if trained_regimes["h=1.0e-6"]:
    print(f"Loss on the test set (h=1.0e-6): {test(test_dataloader_h1_0e6, model_h1_0e6, loss_fn)}.\n")
    torch.save(model_h1_0e6.state_dict(), "saved_models\\model_h1_0e6.pth")
    print("Saved PyTorch Model State to saved_models\\ model_h1_0e6.pth")
    size = len(dataloader_h1_0e6.dataset)
    num_batches = len(dataloader_h1_0e6)
    
    with torch.no_grad():
        for X, y in dataloader_h1_0e6.dataset:
            #X, y = X.to(device), y.to(device)
            #y = y.view(-1, 1)

            pred = model_h1_0e6(X)
            #y = y.unsqueeze(1)
            y_pred_h1_0e6.append(pred)
            y_true_h1_0e6.append(y)
    y_pred_h1_0e6 = [x.item() for x in y_pred_h1_0e6]
    y_true_h1_0e6 = [x.item() for x in y_true_h1_0e6]
    #print("types y_pred, y_true: ",(y_pred), (y_true))
    print(f"R square (h=1.0e-6): {round(r2_score(y_pred_h1_0e6, y_true_h1_0e6),3)}")
    

print("Train losses h=0.5: ",train_losses_h0_5)
print("Valid losses h=0.5: ",valid_losses_h0_5)
print(f"Total training time: {round(total_training_time,2)} s")
