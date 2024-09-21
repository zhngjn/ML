#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gen_data

torch.manual_seed(100)

g_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        # set train mode
        model.train()
    
        y_pred = model(x)
        
        loss = loss_fn(y_pred, y)
        
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item()
    
    return train_step


def make_val_step(model, loss_fn):
    def val_step(x, y):
        # set eval mode
        model.eval()
    
        y_pred = model(x)
        
        loss = loss_fn(y_pred, y)
        
        return loss.item()
    
    return val_step


def mini_batch(device, data_loader, step):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
    
        mini_batch_loss = step(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    
    return np.mean(mini_batch_losses)


train_loader, val_loader = gen_data.make_dataloader()

model = nn.Sequential(nn.Linear(1, 3), nn.Linear(3, 3), nn.Linear(3, 1)).to(g_device)

lr = 0.001

loss_fn = nn.MSELoss(reduction='mean')

optimizer = optim.SGD(model.parameters(), lr=lr)

train_step = make_train_step(model, loss_fn, optimizer)
val_step = make_val_step(model, loss_fn)

writer = SummaryWriter('/home/zhangjun/tmp/tb/lg_torch/')
x_sample, _ = next(iter(train_loader))
writer.add_graph(model, x_sample.to(g_device))

# # init weights
# w = torch.randn(1, requires_grad=True, dtype=torch.float64, device=g_device)
# b = torch.randn(1, requires_grad=True, dtype=torch.float64, device=g_device)


epochs = 100
losses = []
val_losses = []

for epoch in range(epochs):
    print(f'epoch {epoch} -----------------------------------')
    
    loss = mini_batch(g_device, train_loader, train_step)
    print(f'loss={loss}')
    losses.append(loss)
    
    with torch.no_grad():
        val_loss = mini_batch(g_device, val_loader, val_step)
        print(f'val_loss={val_loss}')
        val_losses.append(val_loss)
        
    writer.add_scalars(
        main_tag='loss', 
        tag_scalar_dict={'train': loss, 'val': val_loss}, 
        global_step=epoch)
    
writer.close()