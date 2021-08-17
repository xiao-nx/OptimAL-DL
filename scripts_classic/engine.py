# engine.py

import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np

'''
6994
0     992  0.14->3.49
1    2400  0.34-> 1.44
2     143  0.02-> 24.12
3    3459  0.49->
'''

def train_fn(data_loader, model, optimizer,device):
    """
    training function which trains for one epoch
    parameters:
    data_loader: torch dataloader object
    model: torch model,bert in our case
    optimizer:torch optimizer, e.g. adam,sgd, etc.
    device: "cuda" or "cpu"
    scheduler: learning rate scheduler
    """
    
    model.train()
    
    final_labels = []
    final_outputs = []
    total_loss = []
    
    #weights = torch.tensor([2, 1.2, 5, 1],dtype=torch.float).to(device)
    #weights = torch.tensor([1, 1.5],dtype=torch.float).to(device)
    
    # loop over all batches
    for data in data_loader:
        texts, labels = data.text, data.label

        # zero-grad the optimizer
        optimizer.zero_grad()
            
        # pass through the model
        logits = model(texts)
        
        # cross entropy loss for classifier
        #loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        loss = nn.CrossEntropyLoss()(logits, labels)

        # backward step the loss
        loss.backward()
        
        # step optimizer
        optimizer.step()
        
        # save training parameters
        outputs = logits
        final_labels.extend(labels.cpu().detach().numpy().tolist())
        final_outputs.extend(outputs.cpu().detach().numpy().tolist())
        total_loss.append(loss.cpu().detach().numpy().tolist())
        
    train_loss = np.average(total_loss)
        
    return final_outputs, final_labels, train_loss 
    
def evaluate_fn(data_loader, model, device,loss_flag=False):
    # initialize empty lists to store predictions and labels
    final_predictions = []
    final_labels = []
    total_loss =[]

    # put the model in eval mode
    model.eval()
    
    # disable gradient calculation
    with torch.no_grad():
        
        for data in data_loader:
            # fetch text and label from the dict
            texts, labels = data.text, data.label     

            # make predictions
            logits = model(texts)
            
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss.append(loss.cpu().detach().numpy().tolist())

            # move predictions and labels to list, move predictiona and labels to cpu.
            logits = logits.cpu().numpy().tolist()
            labels = data.label.cpu().numpy().tolist()
            final_predictions.extend(logits)
            final_labels.extend(labels)
    
    valid_loss = np.average(total_loss)

    # return final predictions and labels
    return final_predictions,final_labels, valid_loss

    





