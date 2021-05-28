# engine.py

import torch
from torch._C import dtype
import torch.nn as nn

def train(data_loader, model, optimizer,device):
    """
    train model for one epoch
    parameters:
    data_loader: torch dataloader
    model: model
    optimizer:torch optimizer, e.g. adam,sgd, etc.
    device: "cuda" or "cpu"
    """

    # set model to training mode
    model.train()

    # go through batches of data in data loader
    for data in data_loader:
        # fetch text and label from the dict
        # texts = data["text"]
        # labels = data["label"]
        # print("data:",data)
        texts, labels = data.text, data.label
        # print("texts:",texts.shape)
        # print("labels:",labels.shape)

        # move the data to device 
        # texts = texts.to(device,dtype=torch.long)
        # labels = labels.to(device,dtype=torch.float)

        # clear the gradients
        optimizer.zero_grad()

        # make predictions from the model
        predications = model(texts)

        # calculate the loss
        # loss = nn.BCEWithLogitsLoss()(
        #     predications,
        #     labels.view(-1,1)
        # )

        # print("predications:",predications.shape)
        # print("labels:",labels.shape)
        loss = nn.CrossEntropyLoss()(predications,labels)

        # compute gradient of loss w.r.t.
        # all parameters of the model that are trainable
        loss.backward()

        # single optimization step
        optimizer.step()

def evaluate(data_loader, model, device):
    # initialize empty lists to store predictions and labels
    final_predictions = []
    final_labels = []

    # put the model in eval mode
    model.eval()
    
    # disable gradient calculation
    with torch.no_grad():
        for data in data_loader:
            # texts = data["text"]
            # labels = data["label"]
            # texts = texts.to(device,dtype=torch.long)
            # labels = labels.to(device,dtyoe=torch.float)

            texts, labels = data.text, data.label

            # make predictions
            predictions = model(texts)

            # move predictions and labels to list
            # need to move predictiona and labels to cpu,too
            predictions = predictions.cpu().numpy().tolist()
            labels = data["label"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_labels.extend(labels)

    # return final predictions and labels
    return final_predictions,final_labels

    





