import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from lib.model import DegradationModel

def train_meta_task(epoch, num_epochs, step, num_inner_steps, model, dataloader, criterion, optimizer, device):
    total_loss = 0.0
    model.train()
    for x, y in tqdm(dataloader, desc=f"Training... Epoch: {epoch}/{num_epochs}, Inner Step: {step}/{num_inner_steps}"):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate_meta_task(epoch, num_epochs, i, num_tasks, model, dataloader, criterion, device):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"Validating... Epoch: {epoch}/{num_epochs}, Task: {i+1}/{num_tasks}"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y.view(-1, 1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_meta_model(train_tasks, val_tasks, model, criterion, num_epochs=100, num_inner_steps=10, device="cpu"):
    log_file = f'rul_log.txt'
    with open(log_file, 'w') as f:
        f.write("Training Log\n")

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        train_task_losses = []

        for i, train_task in enumerate(train_tasks):
            task_optimizer = optim.AdamW(model.parameters(), lr=0.001)
            train_loss = 0.0
            for j in range(1, num_inner_steps + 1):
                step = (i)*3 + j
                inner_loss = train_meta_task(epoch, num_epochs, step, num_inner_steps*len(train_tasks), model, train_task, criterion, task_optimizer, device)
                train_loss += inner_loss
            train_task_losses.append(train_loss / num_inner_steps)

        train_loss = sum(train_task_losses) / len(train_task_losses)

        val_task_losses = []
        for i, val_task in enumerate(val_tasks):
            val_loss = validate_meta_task(epoch, num_epochs, i, len(val_tasks), model, val_task, criterion, device)
            val_task_losses.append(val_loss)

        val_loss = sum(val_task_losses) / len(val_task_losses)

        end_time = time.time()
        log_message = f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Time: {end_time-start_time}\n'
        print(log_message)

        with open(log_file, 'a') as f:
            f.write(log_message)

        torch.save(model.state_dict(), f'./models/rul_model.pth')
