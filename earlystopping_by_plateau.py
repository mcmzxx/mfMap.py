import torch
import numpy as np

torch_inf = torch.tensor(np.Inf)

class Earlystopping_by_plateau():


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
        mode_dict = {'min': torch.lt,'max': torch.gt}
        if self.monitor == 'acc':
            self.mode = 'max'
        else:
            self.mode = 'min'
        self.monitor_op=mode_dict[self.mode]

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        self.best = torch_inf if self.monitor_op == torch.lt else -torch_inf

    

    def early_stop(self, current_epoch, current_val):
        stop_training = False

        if not isinstance(current_val, torch.Tensor):
            current_val = torch.tensor(current_val)

        if self.monitor_op(current_val - self.min_delta, self.best):
            self.best = current_val
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = current_epoch
                stop_training = True
        return stop_training

    
