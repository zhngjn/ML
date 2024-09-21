import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

np.random.seed(100)

g_gt_w = 3
g_gt_b = 5

g_N = 1000

def gen_data():
    x = np.random.rand(g_N, 1)
    error = 0.1 * np.random.randn(g_N, 1)
    y = g_gt_w * x + g_gt_b + error
    return x, y

def make_dataloader():
    x, y = gen_data()
    
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    dataset = TensorDataset(x_tensor, y_tensor)
    
    num_train = int(len(dataset) * 0.8)
    num_val = len(dataset) - num_train
    train_data, val_data = random_split(dataset, [num_train, num_val])
    
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=16)
    
    return train_loader, val_loader