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
from lib.engine import *
import pickle

extisted = True
input_channels= 2
hidden_size= 64
num_layers= 4
seq_length= 128
batch_size = 4096
num_epochs = 20
num_inner_steps = 3
lr=0.01
rul_log_file = './rul_log.txt'
rul_model_path="./models/rul_model.pth"
gan_log_file = "./gan_log.txt"
gan_model_ptah = "./models/gan_model.pth"
dis_model_path = "./models/dis_model.pth"
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

        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        return x, y
    

def rmse_loss_function(pred, target):
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
        )

    generator = Generator(
        input_channel=input_channels,
        hidden_channel=hidden_size,
        num_layers=num_layers
    )

    discriminator = Discriminator(
        input_channel=input_channels,
        hidden_channel=hidden_size,
        num_layers=num_layers
    )

    g_optimizer = optim.AdamW(generator.parameters(), lr=lr)   
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=lr)

    train_gan_model(
        train_tasks=train_tasks,
        generator=generator,
        discriminator=discriminator,
        criterion=nn.BCEWithLogitsLoss(),
        g_optimizer = g_optimizer,
        d_optimizer = d_optimizer,
        num_epochs=num_epochs,
        device = device,
        log_file = gan_log_file,
        gan_model_path  = gan_model_ptah,
        dis_model_path = dis_model_path
    )
    
    train_meta_model(
        train_tasks, 
        val_tasks,
        model=rul_model,
        criterion=rmse_loss_function,
        generator=generator,
        num_augmented_samples=batch_size,
        num_epochs=num_epochs,
        num_inner_steps=num_inner_steps,
        device=device,
        rul_log_file = rul_log_file,
        rul_model_path = rul_model_path,
    )

def data(extisted=False):
    if extisted == False:
        data_paths = get_bearing_paths(".\\datasets")

        train_tasks = []
        val_tasks = []
        for i in range(len(data_paths)):
            df = load_data(data_paths[i])
            df = preprocess_data(df)#.iloc[:12800,:]

            vibration = df[['Horizontal_vibration_signals', 'Vertical_vibration_signals']].values
            rul = df['Normalized_RUL'].values

            dataset = BearingDataset(vibration, rul, seq_length, device)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            if i in [0,5,10]:
                val_tasks.append(dataloader)
            else:
                train_tasks.append(dataloader)
    else:
        with open('trainset.pkl', 'rb') as f:
            train_tasks = pickle.load(f)
        f.close()

        with open('testset.pkl', 'rb') as f:
            val_tasks = pickle.load(f)
        f.close()

    return train_tasks, val_tasks


if __name__ == "__main__":
    train_tasks, val_tasks = data(extisted)

    main(train_tasks, val_tasks, input_channels, hidden_size, num_layers, seq_length, num_epochs, num_inner_steps, rul_log_file, rul_model_path, device)
