# train.py

import io
from unicodedata import name
import torch 
import torch.nn as nn
from torchtext import vocab
from torchtext.legacy import data

import numpy as np
import pandas as pd
import time
import os
import datetime
import random

import config 
import engine
import metrics_func

from importlib import import_module

import early_stopping
from early_stopping import Loss_EarlyStopping,F1_EarlyStopping
import learning_rate_scheduler
from learning_rate_scheduler import LRScheduler


## early stop paparameters
patience = 10
if config.SELECTED == 'loss':
    early_stopping = Loss_EarlyStopping(patience, verbose=True)
if config.SELECTED == 'F1score':
    early_stopping = F1_EarlyStopping(patience, verbose=True)


def load_vectors(fname):
    """
    load pretrained word vector
    parameters:
    fname: the path of pretrined vector
    """
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    vectors_data = vocab.Vectors(name=fname)

    return vectors_data


def run():
    """
    Run training and validation for a given fold and dataset.
    parameters:
    """
    
    LABEL = data.LabelField(use_vocab=True)
    TEXT = data.Field(sequential=True, tokenize=lambda x:x.split(), lower=True, fix_length=config.MAX_LENGTH)

### 1/5
    dataset  = data.TabularDataset(path=config.TRAIN_DATASET_FNAME, 
                                    format='csv', 
                                    fields=[('text', TEXT),('label', LABEL)], 
                                    skip_header=True)
    # split the dataset, 8:2
    train_dataset, valid_dataset = dataset.split(split_ratio=[0.8,0.2], random_state=random.getstate())
    
    test_data  = data.TabularDataset(path=config.TEST_DATASET_FNAME,
                                    format='csv', 
                                    fields=[('text', TEXT),('label', LABEL)], 
                                    skip_header=True)
    
### 2
#     train_dataset  = data.TabularDataset(path=config.TRAIN_DATASET_FNAME, 
#                                     format='csv', 
#                                     fields=[('text', TEXT),('label', LABEL)], 
#                                     skip_header=True)    
#     valid_dataset  = data.TabularDataset(path=config.VAL_DATASET_FNAME, 
#                                     format='csv', 
#                                     fields=[('text', TEXT),('label', LABEL)], 
#                                     skip_header=True) 
                                    
#     test_data  = data.TabularDataset(path=config.TEST_DATASET_FNAME,
#                                     format='csv', 
#                                     fields=[('text', TEXT),('label', LABEL)], 
#                                     skip_header=True)
  
### 3/4
#     train_dataset  = data.TabularDataset(path=config.TRAIN_DATASET_FNAME, 
#                                     format='csv', 
#                                     fields=[('text', TEXT),('label', LABEL)], 
#                                     skip_header=True)     
    
#     dataset  = data.TabularDataset(path=config.TEST_DATASET_FNAME, 
#                                     format='csv', 
#                                     fields=[('text', TEXT),('label', LABEL)], 
#                                     skip_header=True)
#     # split the dataset, 5:5
#     valid_dataset, test_data = dataset.split(split_ratio=[0.5,0.5], random_state=random.getstate())

### 5



    # load embeddings
    vectors_data = load_vectors(config.EMBEDDING_FNAME)

    TEXT.build_vocab(train_dataset, vectors=vectors_data)
    LABEL.build_vocab(train_dataset)
    print ('vector size:',TEXT.vocab.vectors.size())
    embedding_pretrained_matrix = TEXT.vocab.vectors
    
    # create torch device
    print("To device...")
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    train_it, valid_it = data.BucketIterator.splits((train_dataset, valid_dataset),
                                                    batch_sizes=(config.TRAIN_BATCH_SIZE,config.VAL_BATCH_SIZE), 
                                                    device=device, 
                                                    sort_key=lambda x: len(x.text),
                                                    sort_within_batch=False,
                                                    shuffle=True,
                                                    repeat=False)
    test_it = data.BucketIterator(test_data, 
                                  batch_size=config.TEST_BATCH_SIZE, 
                                  sort_key=lambda x: len(x.text), 
                                  shuffle=False,
                                  device=device)
    
                                                          
    # fetch model
    vocab_size = len(TEXT.vocab) # TEXT.vocab.vectors.size()
#     pretrained_vec = TEXT.vocab.vectors
    
    # selecte network  
    x = import_module('networks.'+config.NETWORK)
    model = x.Model(vocab_size,embedding_pretrained=embedding_pretrained_matrix)
        
    # send model to device
    model.to(device)

    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # if you have multiple GPUs, model model to DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    params_list = []
    # train and validate for all epochs
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()

        ###----Train--------
        train_outputs, train_labels, train_loss = engine.train_fn(train_it, model, optimizer, device)
        train_outputs = torch.Tensor(train_outputs)
        _, train_predicted = torch.max(train_outputs, dim=1)
        train_parameters_dict = metrics_func.performance_evaluation_func(train_predicted,train_labels,epoch=str(epoch))
        # save train paremeters
        params_list.append(train_parameters_dict)
        train_f1 = train_parameters_dict['f1_score_macro']
        train_prec = train_parameters_dict['precision_macro']
        train_recall = train_parameters_dict['precision_macro']
        print('\n')
        print(f" Train Epoch: {epoch}, F1 = {train_f1},precision = {train_prec},recall = {train_recall}")
        ###------------
        
        # validate
        val_outputs, val_labels, valid_loss = engine.evaluate_fn(valid_it, model, device)
        val_outputs = torch.Tensor(val_outputs)
        _, val_predicted = torch.max(val_outputs, dim=1)     
        # calculate evaluation paremeters
        val_parameters_dict = metrics_func.performance_evaluation_func(val_predicted, val_labels, epoch=str(epoch),flag='val')
        # save evaluation paremeters
        params_list.append(val_parameters_dict)
            
        val_f1 = val_parameters_dict['f1_score_macro']
        val_prec = val_parameters_dict['precision_macro']
        val_recall = val_parameters_dict['recall_macro']
        print(f"Val Epoch: {epoch},F1 = {val_f1},precision = {val_prec}, recall = {val_recall}")
        
        ###-------Test-----------------------
        test_outputs, test_labels, test_loss = engine.evaluate_fn(test_it, model, device)
        test_outputs = torch.Tensor(test_outputs)
        _, test_predicted = torch.max(test_outputs, dim=1)    
        # calculate evaluation paremeters
        test_parameters_dict = metrics_func.performance_evaluation_func(test_predicted, test_labels, epoch=str(epoch),flag='test')
        # save evaluation paremeters
        params_list.append(test_parameters_dict)
           
        test_f1 = test_parameters_dict['f1_score_macro']
        test_prec = test_parameters_dict['precision_macro']
        test_recall = test_parameters_dict['recall_macro']
        print(f"test Epoch: {epoch},F1 = {test_f1},precision = {test_prec}, recall = {test_recall}")
        
        lr_scheduler = LRScheduler(optimizer)
        lr_scheduler(valid_loss)
        
        
        # simple early stopping
#         val_f1 = float(val_f1)
        #f1 = (float(train_f1) + float(val_f1)) / 2
        val_loss = float(valid_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # 获得 early stopping 时的模型参数
#         model.load_state_dict(torch.load('checkpoint.pt'))

#         save_model_func(model, epoch, path='outputs')
    
    metrics_func.save_parameters_txt(params_list)
    
if __name__ == "__main__":

    run()
