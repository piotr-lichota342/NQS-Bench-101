import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split

from config import BATCH_SIZE, TRAIN_PROPORTION, TEST_PROPORTION


# Initialize the dataset


csv_filenames_dict = {
    'h=0.5':'1d_tfim_N12_h0.5_full_dataset.csv',
    'h=1.0':'1d_tfim_N12_h1.0_full_dataset.csv',
    'h=1.0e-6':'1d_tfim_N12_h1.0e-6_full_dataset.csv',
    'h=2.0':'1d_tfim_N12_h2.0_full_dataset.csv'  
}


class CSVDataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename, dtype={"config": str})
       
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Assume the semi last column is the target (amplitude) and the spin configuration is the feature 
        config_numeric_list = [int(x) for x in row["config"]]
        #print(config_numeric_list)
        
        features = torch.tensor(config_numeric_list, dtype=torch.float32)
    
        target = torch.tensor(row["amplitude"], dtype=torch.float32)
        
        return features, target
    
    


#csv_file = '1d_tfim_N12_h0.5_full_dataset.csv'

#merged_df = pd.concat(map(pd.read_csv, csv_filenames), ignore_index=True)
dataset_h0_5 = CSVDataset(csv_filenames_dict['h=0.5'])
dataset_h1_0 = CSVDataset(csv_filenames_dict['h=1.0'])
dataset_h1_0e6 = CSVDataset(csv_filenames_dict['h=1.0e-6'])
dataset_h2_0 = CSVDataset(csv_filenames_dict['h=2.0'])


print(f"Train size for h=0.5: {int(TRAIN_PROPORTION*len(dataset_h0_5))}")
print(f"Test size h=0.5: {int(TEST_PROPORTION*len(dataset_h0_5))}")
print(f"Valid size h=0.5: {int(len(dataset_h0_5) - ((TRAIN_PROPORTION+TEST_PROPORTION)*len(dataset_h0_5)))}")
print(f"Dataset length h=0.5: {len(dataset_h0_5)}")
 
# Create DataLoaders
dataloader_h0_5 = DataLoader(dataset_h0_5, batch_size=BATCH_SIZE, shuffle=True)
dataloader_h1_0 = DataLoader(dataset_h1_0, batch_size=BATCH_SIZE, shuffle=True)
dataloader_h1_0e6 = DataLoader(dataset_h1_0e6, batch_size=BATCH_SIZE, shuffle=True)
dataloader_h2_0 = DataLoader(dataset_h2_0, batch_size=BATCH_SIZE, shuffle=True)

'''
for features, targets in dataloader:
    # Here you can perform operations on features and targets
    print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
'''



n = len(dataset_h0_5)

train_size = int(TRAIN_PROPORTION * n)
test_size = int(TEST_PROPORTION * n)
valid_size = n - train_size - test_size

train_dataset_h0_5, test_dataset_h0_5, valid_dataset_h0_5 = random_split(dataset_h0_5, [train_size, test_size, valid_size])
train_dataset_h1_0, test_dataset_h1_0, valid_dataset_h1_0 = random_split(dataset_h1_0, [train_size, test_size, valid_size])
train_dataset_h2_0, test_dataset_h2_0, valid_dataset_h2_0 = random_split(dataset_h2_0, [train_size, test_size, valid_size])
train_dataset_h1_0e6, test_dataset_h1_0e6, valid_dataset_h1_0e6 = random_split(dataset_h1_0e6, [train_size, test_size, valid_size])
 
train_dataloader_h0_5, test_dataloader_h0_5, valid_dataloader_h0_5 = DataLoader(train_dataset_h0_5, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset_h0_5, batch_size=BATCH_SIZE, shuffle=False), DataLoader(valid_dataset_h0_5, batch_size=BATCH_SIZE, shuffle=False)
train_dataloader_h1_0, test_dataloader_h1_0, valid_dataloader_h1_0 = DataLoader(train_dataset_h1_0, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset_h1_0, batch_size=BATCH_SIZE, shuffle=False), DataLoader(valid_dataset_h1_0, batch_size=BATCH_SIZE, shuffle=False)
train_dataloader_h2_0, test_dataloader_h2_0, valid_dataloader_h2_0 = DataLoader(train_dataset_h2_0, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset_h2_0, batch_size=BATCH_SIZE, shuffle=False), DataLoader(valid_dataset_h2_0, batch_size=BATCH_SIZE, shuffle=False)
train_dataloader_h1_0e6, test_dataloader_h1_0e6, valid_dataloader_h1_0e6 = DataLoader(train_dataset_h1_0e6, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset_h1_0e6, batch_size=BATCH_SIZE, shuffle=False), DataLoader(valid_dataset_h1_0e6, batch_size=BATCH_SIZE, shuffle=False)