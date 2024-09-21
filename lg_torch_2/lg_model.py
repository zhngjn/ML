import numpy as np
import torch

class LGModel:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.train_loader = None
        self.val_loader = None
        
        self.writer = None
        
        self.losses = []
        self.val_losses = []
        self.epochs = 0
        
        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()
        
        
    def to(self, device):
        self.device = device
        self.model.to(self.device)
        
        
    def set_loaders(self, train_loader, val_loader = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        
    def set_writer(self, writer):
        self.writer = writer
        
        
    def train(self, epochs, seed):
        # set train mode
        self.model.train()
        
        self._set_seeds(seed)
        
        for epoch in range(epochs):
            print(f'epoch {epoch} -----------------------------------')
        
            self.epochs += 1
            
            loss = self._mini_batch('train')
            print(f'loss={loss}')
            self.losses.append(loss)
        
            if self.val_loader != None:
                with torch.no_grad():
                    val_loss = self._mini_batch('val')
                    print(f'val_loss={val_loss}')
                    self.val_losses.append(val_loss)
                    
            if self.writer:
                scalars = {'train': loss}
                if self.val_loader:
                    scalars.update({'val': val_loss})
                self.writer.add_scalars(
                    main_tag='loss', 
                    tag_scalar_dict=scalars, 
                    global_step=self.epochs)
                
        if self.writer:
            self.writer.flush()
        
        
    def predict(self, x):
        # set eval mode
        self.model.eval()
        
        x_tensor = torch.as_tensor(x).float()
        
        y_pred = self.model(x_tensor.to(self.device))
        
        return y_pred.detach().cpu().numpy()
    
    
    def _set_seeds(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        
    def _make_train_step(self):
        def train_step(x, y):
            # set train mode
            self.model.train()
        
            y_pred = self.model(x)
            
            loss = self.loss_fn(y_pred, y)
            
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return loss.item()
        
        return train_step
    
        
    def _make_val_step(self):
        def val_step(x, y):
            # set eval mode
            self.model.eval()
        
            y_pred = self.model(x)
            
            loss = self.loss_fn(y_pred, y)
            
            return loss.item()
        
        return val_step
    
    
    def _mini_batch(self, mode):
        if mode == 'train':
            data_loader = self.train_loader
            step = self.train_step
            assert data_loader != None
        elif mode == 'val':
            data_loader = self.val_loader
            step = self.val_step
        else:
            raise RuntimeError(f'Unknown mode {mode}')
        
        if data_loader is None:
            return None
        
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
    
            mini_batch_loss = step(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
    
        loss = np.mean(mini_batch_losses)
        
        return loss
    