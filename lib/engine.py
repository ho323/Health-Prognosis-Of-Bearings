import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from lib.model import DegradationModel

def generate_augmented_data(generator, x, num_augmented_samples, device):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_augmented_samples, x.size(1), x.size(2)).to(device)
        augmented_data = generator(noise)
    return torch.cat([x, augmented_data], dim=0)

def train_meta_task(epoch, num_epochs, step, num_inner_steps, model, dataloader, criterion, optimizer, device, generator=None, num_augmented_samples=0):
    total_loss = 0.0
    model.train()
    for x, y in tqdm(dataloader, desc=f"Training... Epoch: {epoch}/{num_epochs}, Inner Step: {step}/{num_inner_steps}"):
        x, y = x.to(device), y.to(device)
        
        if generator:
            x = generate_augmented_data(generator, x, num_augmented_samples, device)
            y = y.repeat(1 + num_augmented_samples // len(y)) 
        
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

def train_meta_model(train_tasks, val_tasks, model, criterion, generator=None, num_augmented_samples=1024, num_epochs=100, num_inner_steps=10, device="cpu", log_file="rul.log", model_path="rul.pth"):
    with open(log_file, 'w') as f:
        f.write("Training Log\n")
    model = model.to(device)

    if generator:
        generator = generator.to(device)
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        train_task_losses = []
        for i, train_task in enumerate(train_tasks):
            task_optimizer = optim.AdamW(model.parameters(), lr=0.001)
            train_loss = 0.0
            for j in range(1, num_inner_steps + 1):
                step = (i)*3 + j
                inner_loss = train_meta_task(epoch, num_epochs, step, num_inner_steps*len(train_tasks), 
                                             model, train_task, criterion, task_optimizer, device, 
                                             generator=generator, num_augmented_samples=num_augmented_samples)
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

        torch.save(model.state_dict(), model_path)

def train_gan_model(train_tasks, generator, discriminator, criterion, g_optimizer, d_optimizer, num_epochs=100, device="cpu", log_file="gan.log", gan_model_path="gan.pth", dis_model_path="dis.pth"):
    with open(log_file, 'w') as f:
        f.write("Training Log\n")

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        generator.train()
        discriminator.train()
        for i, train_task in enumerate(train_tasks):
            for batch in tqdm(train_task, desc=f"Epoch: {epoch}, Task: {i+1}/{len(train_tasks)}"):
                d_optimizer.zero_grad()
                
                real_data, real_labels = batch
                real_data, real_labels = real_data, real_labels.view(-1,1)

                noise = torch.randn(real_data.size()).to(device)
                fake_data = generator(noise)
                fake_labels = torch.zeros(real_labels.size()).to(device)
                
                real_outputs = discriminator(real_data)
                fake_outputs = discriminator(fake_data.detach())

                d_loss_real = criterion(real_outputs, real_labels)
                d_loss_fake = criterion(fake_outputs, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                d_optimizer.step()
                
                g_optimizer.zero_grad()
                fake_outputs = discriminator(fake_data)
                g_loss = criterion(fake_outputs, real_labels)
                
                g_loss.backward()
                g_optimizer.step()
            
        end_time = time.time()
        log_message = f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f},  Time: {end_time-start_time}\n"
        print(log_message)

        with open(log_file, 'a') as f:
            f.write(log_message)

        torch.save(generator.state_dict(), gan_model_path)
        torch.save(discriminator.state_dict(), dis_model_path)