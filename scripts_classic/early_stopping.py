# https://blog.csdn.net/qq_37430422/article/details/103638681

import numpy as np
import torch
import os

class Loss_EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='outputs'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        mode_fname = os.path.join(self.path, 'checkpoint.pt')
        torch.save(model.state_dict(), mode_fname)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

        
class F1_EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, verbose=False,  path='outputs'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.f1_score_max = 0
        self.delta = delta
        self.path = path

    def __call__(self, f1_score, model):

        score = f1_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'F1_EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(f1_score, model)
            self.counter = 0

    def save_checkpoint(self, f1_score, model):
        '''Saves model when validation F1score crease.'''
        if self.verbose:
            print(f'Validation F1 score increased ({self.f1_score_max:.4f} --> {f1_score:.4f}).  Saving model ...')
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        mode_fname = os.path.join(self.path, 'checkpoint.pt')
        torch.save(model.state_dict(), mode_fname)	# 这里会存储迄今最优模型的参数
        self.f1_score_max = f1_score
        
        
        