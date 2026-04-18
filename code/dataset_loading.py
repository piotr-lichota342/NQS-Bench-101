import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from torch.utils.data import random_split

from config import BATCH_SIZE, TRAIN_PROPORTION, TEST_PROPORTION


# Initialize the dataset
csv_filenames = [
    'NQS-Bench-101/1d_tfim_N12_h1.0e-6_full_dataset.csv' 
]

csv_filenames2 = [
    'NQS-Bench-101/1d_tfim_N12_h0.5_full_dataset.csv',
    'NQS-Bench-101/1d_tfim_N12_h1.0_full_dataset.csv',
    'NQS-Bench-101/1d_tfim_N12_h1.0e-6_full_dataset.csv',
    'NQS-Bench-101/1d_tfim_N12_h2.0_full_dataset.csv'  
] # zostawiam na razie inne regimy


class CSVDataset(Dataset):
    def __init__(self, csv_filenames):
        self.data = pd.concat(
            [pd.read_csv(f, dtype={"config": str}) for f in csv_filenames],
            ignore_index=True
        )
 
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
dataset = CSVDataset(csv_filenames)

print(f"Train size: {int(TRAIN_PROPORTION*len(dataset))}")
print(f"Test size: {int(TEST_PROPORTION*len(dataset))}")
print(f"Valid size: {int(len(dataset) - ((TRAIN_PROPORTION+TEST_PROPORTION)*len(dataset)))}")
print(f"Dataset length: {len(dataset)}")
 
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

'''
for features, targets in dataloader:
    # Here you can perform operations on features and targets
    print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
'''



n = len(dataset)

train_size = int(TRAIN_PROPORTION * n)
test_size = int(TEST_PROPORTION * n)
valid_size = n - train_size - test_size

train_dataset, test_dataset, valid_dataset = random_split(dataset, [train_size, test_size, valid_size])
 
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)