# train.py

import io
from unicodedata import name
import torch 
from torchtext import vocab
from torchtext.legacy import data

import numpy as np
import pandas as pd
import time
import os
import datetime

from sklearn import metrics
# import dataset
import config 
import engine

import CNN
import LSTM
import BiLSTM


def load_vectors(fname):
    """
    load pretrained word vector
    parameters:
    fname: the path of pretrined vector
    """
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    vectors_data = vocab.Vectors(name=fname)

    return vectors_data

def save_model_func(model, epoch, path='outputs', **kwargs):
    """
    parameters:
    model: trained model
    path: the file path to save model
    loss: loss
    last_loss: the loss of beset epoch
    kwargs: every_epoch or best_epoch
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        name = cur_time + '_epoch:{}'.format(epoch)
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')
            
def performance_evaluation_func(y_predicted, y_label,epoch='epoch'):
    """
    parameters:
    y_predicted:
    y_label:
    """

    # define a dict to save parameters.
    parameters_dict = {'epoch':epoch}

    # calculate accuracy
    accuracy = metrics.accuracy_score(y_label, y_predicted)
    # calculate precision
    precision_macro = metrics.precision_score(y_label, y_predicted, average="macro")
    precision_micro = metrics.precision_score(y_label, y_predicted, average="micro")
    precision_weighted = metrics.precision_score(y_label, y_predicted, average="weighted")
    # Recall
    recall_macro = metrics.recall_score(y_label, y_predicted, average='macro')
    recall_micro = metrics.recall_score(y_label, y_predicted, average='micro')
    recall_weighted = metrics.recall_score(y_label, y_predicted, average='weighted')
    # F1 score
    f1_score_macro = metrics.f1_score(y_label, y_predicted, average='macro')
    f1_score_micro = metrics.f1_score(y_label, y_predicted, average='micro')
    f1_score_weighted = metrics.f1_score(y_label, y_predicted, average='weighted')

    # confusion matrix, return tn, fp, fn, tp.
    # classifier_labels = ['treatment','symptomatic relief','contradiction','effect']
    classifier_labels = [0,1,2,3]
    conf_matrix = metrics.confusion_matrix(y_label, y_predicted, labels=classifier_labels)


    # add parameters to dict
    parameters_dict.update({'accuracy':accuracy})
    parameters_dict.update({'precision_macro':precision_macro})
    parameters_dict.update({'precision_micro':precision_micro})
    parameters_dict.update({'precision_weighted':precision_weighted})
    parameters_dict.update({'recall_macro':recall_macro})
    parameters_dict.update({'recall_micro':recall_micro})
    parameters_dict.update({'recall_weighted':recall_weighted})
    parameters_dict.update({'f1_score_macro':f1_score_macro})
    parameters_dict.update({'f1_score_micro':f1_score_micro})
    parameters_dict.update({'f1_score_weighted':f1_score_weighted})
    parameters_dict.update({'confusion_matrix':conf_matrix})

    # save parameters
    # np.save('parameters_info.npy', parameters_dict) 

    return parameters_dict
    
def save_parameters(parameters_dict, path='outputs',epoch='epoch', **kwargs):
    """
    parameters:
    kwargs: every_epoch or best_epoch
    return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    save_fname = os.path.join(path, 'parameters_info'+'_'+ epoch + '.npy')
    np.save(save_fname, parameters_dict)


def run():
    """
    Run training and validation for a given fold and dataset.
    parameters:
    df: pandas dataframe with kfold column
    """
    
    TEXT = data.Field(sequential=True, tokenize=lambda x:x.split(), lower=True)
    LABEL = data.LabelField(use_vocab=True)
    
    train_data  = data.TabularDataset(path=config.TRAIN_DATASET_FNAME, 
                                    format='csv', 
                                    fields=[('text', TEXT), ('label', LABEL)], 
                                    skip_header=True)
                                    
    test_data  = data.TabularDataset(path=config.TEST_DATASET_FNAME,
                                    format='csv', 
                                    fields=[('text', TEXT), ('label', LABEL)], 
                                    skip_header=True)

    # load embeddings
    vectors_data = load_vectors(config.EMBEDDING_FNAME)
    
    TEXT.build_vocab(train_data, vectors=vectors_data)
    LABEL.build_vocab(train_data)
    print ('vector size:',TEXT.vocab.vectors.size())
    
    # create torch device
    print("To device...")
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    train_it, test_it = data.BucketIterator.splits((train_data, test_data),
                                                    batch_sizes=(config.TRAIN_BATCH_SIZE,config.VAL_BATCH_SIZE), 
                                                    device=device, 
                                                    sort_key=lambda x: len(x.text), 
                                                    repeat=False)
                                                          
    # fetch model
    vocab_size = len(TEXT.vocab) # TEXT.vocab.vectors.size()
    pretrained_vec = TEXT.vocab.vectors
    #rint('len(TEXT.vocab): ', len(TEXT.vocab))
    
    # selecte network  
    # Maxpool
    if config.NETWORK == 'CNN':
        model = CNN.CNN(vocab_size, vec=pretrained_vec)
    elif config.NETWORK == 'LSTM':
        model = LSTM.LSTM(vocab_size, vec=pretrained_vec)
    elif config.NETWORK == 'BiLSTM':
        model = BiLSTM.BiLSTM(vocab_size, vec=pretrained_vec)
    else:
        print('Wrong input network!!!')
    
 
    # send model to device
    model.to(device)

    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # train and validate for all epochs
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        # train one epoch
        train_outputs, train_labels = engine.train(train_it, model, optimizer, device)
        ###----Train--------
        if True:
            train_outputs = train_outputs.cpu().detach()
            train_labels = train_labels.cpu().detach()
        train_outputs = torch.Tensor(train_outputs)
        _, train_predicted = torch.max(train_outputs, dim=1)
        train_parameters_dict = performance_evaluation_func(train_predicted,train_labels,epoch=str(epoch))
        train_f1 = train_parameters_dict['f1_score_weighted']
        train_prec = train_parameters_dict['precision_weighted']
        train_recall = train_parameters_dict['recall_weighted']
        print('\n') 
        print(f"Train Epoch: {epoch}, F1 = {train_f1},Precision = {train_prec}, Recall = {train_recall}, ")
        ###------------
        
        # validate
        val_outputs, val_labels = engine.evaluate(test_it, model, device)
        val_outputs = torch.Tensor(val_outputs)
        _, val_predicted = torch.max(val_outputs, dim=1)   
        
        # calculate accuracy
        val_parameters_dict = performance_evaluation_func(val_predicted, val_labels, epoch=str(epoch))
        # save evaluation paremeters
        save_parameters(val_parameters_dict, path='outputs', epoch=str(epoch))
        
        val_f1 = val_parameters_dict['f1_score_weighted']
        val_prec = val_parameters_dict['precision_weighted']
        val_recall = val_parameters_dict['recall_weighted']
        print(f"Val Epoch: {epoch}, F1 = {val_f1},Precision = {val_prec}, Recall = {val_recall}, ")
        print('\n') 
        # print('train_parameters_dict:\n',train_parameters_dict)
        # print('val_parameters_dict:\n',val_parameters_dict)
        save_model_func(model, epoch, path='outputs')

if __name__ == "__main__":

    run()
