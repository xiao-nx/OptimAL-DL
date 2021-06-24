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
        # print(data)
        # fetch text and label from the dict
        entity_1, entity_2, labels, texts = data.drug, data.disease, data.label, data.text
        entity_1 = (entity_1.unsqueeze(0)).expand(texts.shape)
        entity_2 = (entity_2.unsqueeze(0)).expand(texts.shape)
        
        train_data = {'entity_1':entity_1,'entity_2':entity_2,'texts':texts}
        batch_sents = train_data['texts']
        # batch_pos1s = train_data['entity_1']
        # batch_pos2s = train_data['entity_2']
        # print('engine batch_sents: ',batch_sents.shape[0])
        # print('engine batch_pos1s: ',batch_pos1s.shape)
        # print('engine batch_pos2s: ',batch_pos2s.shape)
        # move the data to device 
        # texts = texts.to(device,dtype=torch.long)
        # labels = labels.to(device,dtype=torch.float)

        # clear the gradients
        optimizer.zero_grad()

        # make predictions from the model
        # redications = model(text)
        predications = model(train_data)

        # calculate the loss
        loss = nn.CrossEntropyLoss()(predications,labels)

        # compute gradient of loss w.r.t.
        # all parameters of the model that are trainable
        loss.backward()

        # single optimization step
        optimizer.step()
        
    return predications,labels    
    
def evaluate(data_loader, model, device):
    # initialize empty lists to store predictions and labels
    final_predictions = []
    final_labels = []

    # put the model in eval mode
    model.eval()
    
    # disable gradient calculation
    with torch.no_grad():
        
        for data in data_loader:
            # fetch text and label from the dict
            entity_1, entity_2, labels, texts = data.drug, data.disease, data.label, data.text
            entity_1 = (entity_1.unsqueeze(0)).expand(texts.shape)
            entity_2 = (entity_2.unsqueeze(0)).expand(texts.shape)
        
            val_data = {'entity_1':entity_1,'entity_2':entity_2,'texts':texts}            

            # make predictions
            predictions = model(val_data)

            # move predictions and labels to list, move predictiona and labels to cpu.
            predictions = predictions.cpu().numpy().tolist()
            labels = data.label.cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_labels.extend(labels)

    # return final predictions and labels
    return final_predictions,final_labels

    





