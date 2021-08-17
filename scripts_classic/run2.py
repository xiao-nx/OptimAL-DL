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
delta = 0.005
if config.SELECTED == 'loss':
    early_stopping = Loss_EarlyStopping(patience, verbose=True)
if config.SELECTED == 'F1score':
    early_stopping = F1_EarlyStopping(patience, delta, verbose=True)


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

    train_dataset  = data.TabularDataset(path=config.TRAIN_DATASET_FNAME, 
                                    format='csv', 
                                    fields=[('text', TEXT),('label', LABEL)], 
                                    skip_header=True)
    
    test_data  = data.TabularDataset(path=config.TEST_DATASET_FNAME,
                                    format='csv', 
                                    fields=[('text', TEXT),('label', LABEL)], 
                                    skip_header=True)
    
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

    train_it = data.BucketIterator(train_dataset, 
                                  batch_size=config.TRAIN_BATCH_SIZE, 
                                  sort_key=lambda x: len(x.text), 
                                  shuffle=True,
                                  device=device)                                               
    
    test_it = data.BucketIterator(test_data, 
                                  batch_size=config.TEST_BATCH_SIZE, 
                                  sort_key=lambda x: len(x.text), 
                                  shuffle=False,
                                  device=device)
    
                                                          
    # fetch model
    vocab_size = len(TEXT.vocab) # TEXT.vocab.vectors.size()
    
    # selecte network  
    x = import_module('networks.'+config.NETWORK)
    model = x.Model(vocab_size,embedding_pretrained=embedding_pretrained_matrix)
        
    # send model to device
    model.to(device)

    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    lr_scheduler = LRScheduler(optimizer)

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
        lr_scheduler(train_loss)
        
        
        # simple early stopping
        train_f1 = float(train_f1)
        early_stopping(train_f1, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # 获得 early stopping 时的模型参数
#         model.load_state_dict(torch.load('checkpoint.pt'))

#         save_model_func(model, epoch, path='outputs')
    
    metrics_func.save_parameters_txt(params_list)
    
if __name__ == "__main__":

    run()
