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
import torch.nn as nn

import sklearn
from tqdm import tqdm
#import astropy

from dataset_loading import train_dataloader_h0_5, train_dataloader_h1_0, train_dataloader_h2_0, train_dataloader_h1_0e6
from dataset_loading import test_dataloader_h0_5, test_dataloader_h1_0, test_dataloader_h2_0, test_dataloader_h1_0e6
from dataset_loading import valid_dataloader_h0_5, valid_dataloader_h1_0, valid_dataloader_h2_0, valid_dataloader_h1_0e6

#from dataset_loading import dataloader_h0_5, dataloader_h1_0, dataloader_h1_0e6, dataloader_h2_0

from architecture import model_h0_5, model_h1_0, model_h2_0, model_h1_0e6
from train import train
from valid import valid

from test import test
from config import EPOCHS, device, trained_regimes, DATASET_SIZE, TEST_PROPORTION, EXPONENTIAL_LR, W

from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
#from distancia import Hellinger


# Source: https://github.com/mschuylermoss/DoubleDescentNQS/blob/main/cost_functions.py
def hellinger_distance(true, pred):

    def distance():
    
        log_true = true # model output
        exp_true = torch.exp(log_true) 
        abs_true = torch.abs(exp_true)
        norm_prob_true = abs_true / torch.sum(abs_true)
        
        log_pred = pred
        exp_pred = torch.exp(log_pred)
        abs_pred = torch.abs(exp_pred)
        norm_prob_pred = abs_pred / torch.sum(abs_pred)

        
        return (torch.sqrt(norm_prob_pred) - torch.sqrt(norm_prob_true)) ** 2
    
    hellinger_distance = (1/np.sqrt(2)) * torch.sqrt(torch.sum(distance()))
    
    return hellinger_distance


class HellingerLoss(nn.Module):
    def __init__(self):
        super(HellingerLoss, self).__init__()
        #self.weight = weight

    def forward(self, true, pred):
        # Compute the loss
        loss = hellinger_distance(true, pred)
        return loss.mean()
    
def get_optimizer_and_scheduler(model, width=W, threshold=432):
    """
    Mirrors the optax setup from the paper:
    - small networks (width < 432): Adam + exponential decay
    - large networks (width >= 432): Adam + linear warmup + exponential
    decay
    Args:
    model: your nn.Module
    width: hidden layer width (used to decide schedule type)
    threshold: width cutoff from the paper (432 for N=12)
    """
    LR_PEAK = 1e-3 # peak / initial learning rate
    LR_INIT = 1e-6 # warmup starting lr (large networks only)
    DECAY_RATE = 0.99 # multiply lr by this every transition_steps
   
    TRANS_STEPS = 50 # how often decay is applied (in optimizer
  
    WARMUP_STEPS = 200 # warmup duration (large networks only)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_PEAK)
    if width < threshold:
        # --- small network: plain exponential decay ---
        # gamma is the per-step multiplier; since we call scheduler.step()
        # every TRANS_STEPS optimizer steps, gamma = DECAY_RATE directly.
        scheduler = ExponentialLR(
            optimizer,
            gamma=DECAY_RATE, # lr *= 0.99 each time
            )
    else:
        # --- large network: linear warmup then exponential decay ---
        # Phase 1: linear warmup from LR_INIT to LR_PEAK over WARMUP_STEPS
    
        warmup = LinearLR(
            optimizer,
            start_factor=LR_INIT / LR_PEAK, # 1e-6 / 1e-3 = 0.001
            end_factor=1.0,
            total_iters=WARMUP_STEPS // TRANS_STEPS # in scheduler steps,
        )
        # Phase 2: exponential decay from LR_PEAK onward
        decay = ExponentialLR(
            optimizer,
            gamma=DECAY_RATE
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[WARMUP_STEPS // TRANS_STEPS]
        )
    return optimizer, scheduler

TRANS_STEPS = 50

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




loss_fn = nn.MSELoss() # nn.MSELoss

optimizer_h0_5, scheduler_h0_5 = get_optimizer_and_scheduler(model_h0_5)
optimizer_h1_0, scheduler_h1_0 = get_optimizer_and_scheduler(model_h1_0)
optimizer_h2_0, scheduler_h2_0 = get_optimizer_and_scheduler(model_h2_0)
optimizer_h1_0e6, scheduler_h1_0e6 = get_optimizer_and_scheduler(model_h1_0e6)


'''optimizer_h0_5 = torch.optim.Adam(model_h0_5.parameters(), lr=1e-3, weight_decay=1e-4) 
optimizer_h1_0 = torch.optim.Adam(model_h1_0.parameters(), lr=1e-3) 
optimizer_h2_0 = torch.optim.AdamW(model_h2_0.parameters(), lr=3.5e-5, weight_decay=1e-1) 
optimizer_h1_0e6 = torch.optim.AdamW(model_h1_0e6.parameters(), lr=1e-3) '''

#print(optimizer_h0_5.get_config())



train_losses_h0_5, valid_losses_h0_5 = [], []
train_losses_h1_0, valid_losses_h1_0 = [], []
train_losses_h2_0, valid_losses_h2_0 = [], []
train_losses_h1_0e6, valid_losses_h1_0e6 = [], []

start_training_time = time.time()

'''scheduler_h0_5 = torch.optim.lr_scheduler.ExponentialLR(optimizer_h0_5, gamma=0.95) if EXPONENTIAL_LR else None
scheduler_h1_0 = torch.optim.lr_scheduler.ExponentialLR(optimizer_h1_0, gamma=0.95) if EXPONENTIAL_LR else None
scheduler_h2_0 = torch.optim.lr_scheduler.ExponentialLR(optimizer_h2_0, gamma=0.95) if EXPONENTIAL_LR else None
scheduler_h1_0e6 = torch.optim.lr_scheduler.ExponentialLR(optimizer_h1_0e6, gamma=0.95) if EXPONENTIAL_LR else None'''

for t in tqdm(range(EPOCHS)):
    print(f"Epoch {t+1}/{EPOCHS}\n===============================")
    
    if int(trained_regimes[1]):
        print("Losses for h=0.5:\n")
        train_loss_h0_5, target_train_h0_5, pred_train_h0_5 = train(train_dataloader_h0_5, model_h0_5, loss_fn, optimizer_h0_5)
        valid_loss_h0_5, target_valid_h0_5, pred_valid_h0_5 = valid(valid_dataloader_h0_5, model_h0_5, loss_fn)
        train_losses_h0_5.append(train_loss_h0_5)
        valid_losses_h0_5.append(valid_loss_h0_5)

        if (t + 1) % TRANS_STEPS == 0 and EXPONENTIAL_LR:
            scheduler_h0_5.step()
            print(f"Current LR (h=0.5): {scheduler_h0_5.get_last_lr()}")
        #scheduler_h0_5.step() if EXPONENTIAL_LR else None
        

        
        
    if int(trained_regimes[2]):
        print("Losses for h=1.0:\n")
        train_loss_h1_0, target_train_h1_0, pred_train_h1_0 = train(train_dataloader_h1_0, model_h1_0, loss_fn, optimizer_h1_0)
        valid_loss_h1_0, target_valid_h1_0, pred_valid_h1_0 = valid(valid_dataloader_h1_0, model_h1_0, loss_fn)
        train_losses_h1_0.append(train_loss_h1_0)
        valid_losses_h1_0.append(valid_loss_h1_0)
        if (t + 1) % TRANS_STEPS == 0 and EXPONENTIAL_LR:
            scheduler_h1_0.step()
            print(f"Current LR (h=1.0): {scheduler_h1_0.get_last_lr()}")
        #scheduler_h1_0.step() if EXPONENTIAL_LR else None
        #print(f"Last learning rate (h=1.0): {scheduler_h1_0.get_last_lr()}") if EXPONENTIAL_LR else None
        
    if int(trained_regimes[3]):
        print("Losses for h=2.0:\n")
        train_loss_h2_0, target_train_h2_0, pred_train_h2_0 = train(train_dataloader_h2_0, model_h2_0, loss_fn, optimizer_h2_0)
        valid_loss_h2_0, target_valid_h2_0, pred_valid_h2_0 = valid(valid_dataloader_h2_0, model_h2_0, loss_fn)
        train_losses_h2_0.append(train_loss_h2_0)
        valid_losses_h2_0.append(valid_loss_h2_0)
        if (t + 1) % TRANS_STEPS == 0 and EXPONENTIAL_LR:
            scheduler_h2_0.step()
            print(f"Current LR (h=2.0): {scheduler_h2_0.get_last_lr()}")
        #scheduler_h2_0.step() if EXPONENTIAL_LR else None
        #print(f"Last learning rate (h=2.0): {scheduler_h2_0.get_last_lr()}") if EXPONENTIAL_LR else None
    if int(trained_regimes[0]):
        print("Losses for h=10⁻⁶\n")
        train_loss_h1_0e6, target_train_h1_0e6, pred_train_h1_0e6 = train(train_dataloader_h1_0e6, model_h1_0e6, loss_fn, optimizer_h1_0e6)
        valid_loss_h1_0e6, target_valid_h1_0e6, pred_valid_h1_0e6 = valid(valid_dataloader_h1_0e6, model_h1_0e6, loss_fn)
        train_losses_h1_0e6.append(train_loss_h1_0e6)
        valid_losses_h1_0e6.append(valid_loss_h1_0e6) 
        if (t + 1) % TRANS_STEPS == 0 and EXPONENTIAL_LR:
            scheduler_h1_0e6.step()
            print(f"Current LR (h=10⁻⁶): {scheduler_h1_0e6.get_last_lr()}")
        #scheduler_h1_0e6.step() if EXPONENTIAL_LR else None
        #print(f"Last learning rate (h=1.0⁻⁶): {scheduler_h1_0e6.get_last_lr()}") if EXPONENTIAL_LR else None
 
end_training_time = time.time() 
total_training_time = round(end_training_time - start_training_time, 2)

   
print("Done!")

y_pred_h1_0e6, y_true_h1_0e6 = [], []
y_pred_h0_5, y_true_h0_5 = [], []
y_pred_h1_0, y_true_h1_0 = [], []
y_pred_h2_0, y_true_h2_0 = [], []
   

if int(trained_regimes[1]):
    avg_test_loss_h0_5, _, target_test_h0_5, pred_test_h0_5 = test(test_dataloader_h0_5, model_h0_5, loss_fn)
    avg_train_loss_h0_5, target_train_h0_5, pred_train_h0_5 = train(train_dataloader_h0_5, model_h0_5, loss_fn, optimizer_h0_5)
    avg_valid_loss_h0_5, target_valid_h0_5, pred_valid_h0_5 = valid(valid_dataloader_h0_5, model_h0_5, loss_fn)
    
    #print(f"Loss on the test set (h=0.5): {avg_test_loss_h0_5}.\n")
    #torch.save(model_h0_5.state_dict(), "saved_models\\model_h0_5.pth")
    #print("Saved PyTorch Model State to saved_models\\model_h0_5.pth")
    #size = len(dataloader_h0_5.dataset)
    #num_batches = len(dataloader_h0_5)
    '''
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
    '''
    #print("types y_pred, y_true: ",(y_pred), (y_true))
    #mse = round(mean_squared_error(y_true_h0_5, y_pred_h0_5),4)
    #mae = round(mean_absolute_error(y_true_h0_5, y_pred_h0_5),4)
    

    # Create an instance of the Hellinger class
    #hellinger_dist = Hellinger()

    # Calculate the Hellinger distance between the two distributions
    #distance = hellinger_dist.calculate(y_pred_h0_5, y_true_h0_5)

    # Print the result
    #print(f"The Hellinger distance between the two distributions is (h=0.5): {hellinger_distance(y_pred_h0_5, y_true_h0_5)}")
if int(trained_regimes[2]):
    avg_test_loss_h1_0, _, target_test_h1_0, pred_test_h1_0 = test(test_dataloader_h1_0, model_h1_0, loss_fn)
    avg_train_loss_h1_0, target_train_h1_0, pred_train_h1_0 = train(train_dataloader_h1_0, model_h1_0, loss_fn, optimizer_h1_0)
    avg_valid_loss_h1_0, target_valid_h1_0, pred_valid_h1_0 = valid(valid_dataloader_h1_0, model_h1_0, loss_fn)
    #size = len(dataloader_h1_0.dataset)
    #num_batches = len(dataloader_h1_0)
    '''
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
    '''
if int(trained_regimes[3]):
    avg_test_loss_h2_0, _, target_test_h2_0, pred_test_h2_0 = test(test_dataloader_h2_0, model_h2_0, loss_fn)
    avg_train_loss_h2_0, target_train_h2_0, pred_train_h2_0 = train(train_dataloader_h2_0, model_h2_0, loss_fn, optimizer_h2_0)
    avg_valid_loss_h2_0, target_valid_h2_0, pred_valid_h2_0 = valid(valid_dataloader_h2_0, model_h2_0, loss_fn)
    #print(f"Loss on the test set (h=2.0): {avg_test_loss_h2_0}.\n")
    #torch.save(model_h2_0.state_dict(), "saved_models\\model_h2_0.pth")
    #size = len(dataloader_h2_0.dataset)
    #num_batches = len(dataloader_h2_0)
    '''
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
    '''
if int(trained_regimes[0]):
    avg_test_loss_h1_0e6, _, target_test_h1_0e6, pred_test_h1_0e6 = test(test_dataloader_h1_0e6, model_h1_0e6, loss_fn)
    avg_train_loss_h1_0e6, target_train_h1_0e6, pred_train_h1_0e6 = train(train_dataloader_h1_0e6, model_h1_0e6, loss_fn, optimizer_h1_0e6)
    avg_valid_loss_h1_0e6, target_valid_h1_0e6, pred_valid_h1_0e6 = valid(valid_dataloader_h1_0e6, model_h1_0e6, loss_fn)

    #print(f"Loss on the test set (h=1.0⁻⁶): {avg_test_loss_h1_0e6}.\n")
    #torch.save(model_h1_0e6.state_dict(), "saved_models\\model_h1_0e6.pth")
    #print("Saved PyTorch Model State to saved_models\\ model_h1_0e6.pth")
    #size = len(dataloader_h1_0e6.dataset)
    #num_batches = len(dataloader_h1_0e6)
    '''
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
    print(f"R square (h=1.0⁻⁶): {round(r2_score(y_pred_h1_0e6, y_true_h1_0e6),3)}")
    '''

#print("Train losses h=0.5: ",train_losses_h0_5)
#print("Valid losses h=0.5: ",valid_losses_h0_5)
#print(f"Total training time: {round(total_training_time,2)} s")
