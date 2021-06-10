# engine.py

from typing_extensions import final
import torch
from torch._C import dtype
import torch.nn as nn

def loss_fn(outputs,labels):
    """
    This function returns the loss.
    outputs: output from the model (real numbers)
    labels: input labels 
    """
    # loss = nn.BCEWithLogitsLoss(outputs,targets.view(-1,1))

    loss = nn.CrossEntropyLoss()
    return loss

def train_fn(data_loader, model, optimizer,device,scheduler):
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
    # loop over all batches
    for data in data_loader:
        # extract ids, token type ids and mask from current batch,
        # also extract targets
        ids = data["ids"]
        token_type_ids = data["token_type_ids"]
        mask = data["mask"]
        labels = data["labels"]

        # move everything to specified device
        ids = ids.to(device,dtype=torch.long)
        token_type_ids = token_type_ids.to(device,dtype=torch.long)
        mask = mask.to(device,dtype=torch.long)
        labels = labels.to(device,dtype=torch.long)

        # zero-grad the optimizer
        optimizer.zero_grad()
        # pass through the model
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        # calculate loss
        # loss = loss_fn(outputs,labels)
        criterion = nn.CrossEntropyLoss()
        #print("labels:",labels.shape)
        #print("outputs:",outputs.shape)
        loss = criterion(outputs, labels)
        # backward step the loss
        loss.backward()
        # step optimizer
        optimizer.step()
        # step scheduler
        scheduler.step()


def eval_fn(data_loader, model, device):
    """
    this is the validation function that generates predictions on validation data
    data_loader: torch dataloader object
    model: torch model, bert in our case
    device: can be cpu or gpu
    return: output and targets
    """

    # put model in eval mode
    model.eval()
    # initialize empty lists for labels and outputs
    fin_labels = []
    fin_outputs = []

    # use the no_grad scope
    # its very important else you might run out of gpu memory
    with torch.no_grad():
        # this part is same as training function except for the fact that there is 
        # no zero_grad of optimizer and there is no loss calculation or scheduler steps.
        for data in data_loader:
            ids = data["ids"]
            token_type_ids = data["token_type_ids"]
            mask = data["mask"]
            labels = data["labels"]

            ids = ids.to(device,dtype=torch.long)
            token_type_ids = token_type_ids.to(device,dtype=torch.long)
            mask = mask.to(device,dtype=torch.long)
            labels = labels.to(device,dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            # convert labels to cpu and extend the final list
            labels = labels.cpu().detach()
            fin_labels.extend(labels.numpy().tolist())
            # convert outputs to cpu and extend the final list
            outputs = torch.sigmoid(outputs).cpu().detach()
            fin_outputs.extend(outputs.numpy().tolist())

    return fin_outputs, fin_labels