import torch
import numpy as np

torch_inf = torch.tensor(np.Inf)

class Earlystopping_by_div():
    def __init__(self,
                 monitor='loss',
                 min_delta=0.0001,
                 patience=5,
                 mode='min'):
        self.wait = 0
        self.patience = patience
        self.stopped_epoch = 0
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.train_losses = []
        self.val_losses = []
        if self.monitor == 'acc':
            self.mode = 'max'
        else:
            self.mode = 'min'
        mode_dict = {'min': torch.lt,'max': torch.gt}
        self.monitor_op=mode_dict[self.mode]
        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        self.best = torch_inf if self.monitor_op == torch.lt else -torch_inf


    def early_stop(self, current_epoch,current_train,current_val):
        stop_training = False
        if not isinstance(current_val, torch.Tensor):
            current_val = torch.tensor(current_val)
        if not isinstance(current_train, torch.Tensor):
            current_train = torch.tensor(current_train)
        if (len(self.train_losses) > 0 and len(self.val_losses) > 0):
            # If training loss is decreasing while val loss is increasing
            # Loss (t-1) > Loss (t0), Loss_val (t0) > Loss_val (t0)
            if self.monitor_op(self.train_losses[-1], current_train-self.min_delta) and self.monitor_op(current_val-self.min_delta, self.val_losses[-1]):
                if self.wait >= self.patience:
                    self.stopped_epoch = current_epoch
                    stop_training = True
                self.wait += 1
            else:
                self.wait = 0
        
        self.train_losses.append(current_train)
        self.val_losses.append(current_val)
        return stop_training