import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lib.dataset import *
from lib.model import *
from lib.train import *
import pickle

input_channels= 2
hidden_size= 64
num_layers= 4
seq_length= 128
batch_size = 4096
num_epochs = 20
num_inner_steps = 3
log_file = './rul_log.txt'
model_path="./models/rul_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BearingDataset(Dataset):
    def __init__(self, vibration, rul, seq_length, device):
        self.vibration = vibration
        self.rul = rul
        self.seq_length = seq_length
        self.device = device

    def __len__(self):
        return len(self.rul) - self.seq_length + 1

    def __getitem__(self, idx):
        x = self.vibration[idx:idx + self.seq_length]
        y = self.rul[idx + self.seq_length - 1]

        x = torch.tensor(x, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)

        return x, y
    

def loss_function(pred, target):
    loss = F.mse_loss(pred, target, reduction='mean')
    rmse_loss = torch.sqrt(loss)
    return rmse_loss

def main(train_tasks, val_tasks, input_channels, hidden_size, num_layers, seq_length, num_epochs, num_inner_steps, log_file, model_path, device):
    rul_model = DegradationModel(
        input_channels=input_channels, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        seq_length=seq_length, 
        device=device
        ).to(device)
    
    train_meta_model(
        train_tasks, 
        val_tasks,
        model=rul_model,
        criterion=loss_function,
        num_epochs=num_epochs,
        num_inner_steps=num_inner_steps,
        device=device
    )

if __name__ == "__main__":
    
    data_paths = get_bearing_paths(".\\datasets")

    train_tasks = []
    val_tasks = []
    for i in range(len(data_paths)):
        df = load_data(data_paths[i])
        df = preprocess_data(df)

        vibration = df[['Horizontal_vibration_signals', 'Vertical_vibration_signals']].values
        rul = df['Normalized_RUL'].values

        dataset = BearingDataset(vibration, rul, seq_length, device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if i in [0,5,10]:
            val_tasks.append(dataloader)
        else:
            train_tasks.append(dataloader)
    
    main(train_tasks, val_tasks, input_channels, hidden_size, num_layers, seq_length, num_epochs, num_inner_steps, log_file, model_path, device)
