#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gen_data
import lg_model

np.set_printoptions(suppress=True)

g_device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = nn.Sequential(nn.Linear(1, 3), nn.Linear(3, 3), nn.Linear(3, 1)).to(g_device)
print(f'model------------------------------\n{model.state_dict()}')

lr = 0.001

loss_fn = nn.MSELoss(reduction='mean')

optimizer = optim.SGD(model.parameters(), lr=lr)

# create LGModel
lg_model = lg_model.LGModel(model, loss_fn, optimizer)

train_loader, val_loader = gen_data.make_dataloader()
lg_model.set_loaders(train_loader, val_loader)

writer = SummaryWriter('/home/zhangjun/tmp/tb/lg_torch_2/')
lg_model.set_writer(writer)

# train
lg_model.train(epochs=100, seed=100)

# predict
pred_x = np.random.rand(10, 1)
gt_y = gen_data.g_gt_w * pred_x + gen_data.g_gt_b
pred_y = lg_model.predict(pred_x)

results = np.hstack((pred_x, gt_y, pred_y))
print(results)
