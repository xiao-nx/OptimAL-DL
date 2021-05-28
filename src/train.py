# train.py

import io
from unicodedata import name
import torch 
from torchtext import vocab
from torchtext.legacy import data

import numpy as np
import pandas as pd
import time

from sklearn import metrics

import dataset
import config 
import engine

# out networks
import lstm
import textcnn
# import bilstm_attention

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
    df: pandas dataframe with kfold column
    """

    TEXT = data.Field(sequential=True, tokenize=lambda x:x.split(), lower=True)
    LABEL = data.LabelField(use_vocab=True)
    
        
    train_data  = data.TabularDataset(path=config.TRAIN_DATASET_FNAME, 
                                    format='csv', 
                                    fields=[('text', TEXT), ('label', LABEL)], 
                                    skip_header=True)
    
    validation_data  = data.TabularDataset(path=config.VALIDATION_DATASET_FNAME,
                                    format='csv', 
                                    fields=[('text', TEXT), ('label', LABEL)], 
                                    skip_header=True)
                                    
    test_data  = data.TabularDataset(path=config.TEST_DATASET_FNAME,
                                    format='csv', 
                                    fields=[('text', TEXT), ('label', LABEL)], 
                                    skip_header=True)


    print("Loading embeddings start......")
    # load embeddings
    vectors_data = load_vectors("../inputs/crawl-300d-2M.vec")
    print("Loading embeddings end......")

    # embedding_matrix = create_embedding_matrix(
    #         tokenizer.word_index, embedding_dict
    # )
    
    TEXT.build_vocab(train_data, vectors=vectors_data)
    LABEL.build_vocab(train_data)
    print ('vector size:',TEXT.vocab.vectors.size())
    
    

    # create torch device
    print("To device...")
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    train_it, valid_it, test_it = data.BucketIterator.splits((train_data, validation_data, test_data),
                                                            batch_sizes=(config.BATCH_SIZE,config.BATCH_SIZE,config.BATCH_SIZE), 
                                                            device=device, 
                                                            sort_key=lambda x: len(x.text), 
                                                            repeat=False)

    # fetch our LSTM model
    vocab_size = len(TEXT.vocab)
    pretrained_vec = TEXT.vocab.vectors
    
    # selecte network
    # model = lstm.LSTM(vocab_size, vec=pretrained_vec)
    model = textcnn.textCNN(vocab_size, vec=pretrained_vec)
    # model = bilstm_attention.BiLstmAttention(vocab_size, vec=pretrained_vec)

    # send model to device
    model.to(device)

    # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("Training Model......")
    # set best accuracy to zero
    best_accuracy = 0
    # set early stopping counter to zero
    early_stopping_counter = 0
    # train and validate for all epochs
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        # train one epoch
        # engine.train(train_data_loader, model, optimizer, device)
        engine.train(train_it, model, optimizer, device)
        # validate
        # outputs, labels = engine.evaluate(valid_data_loader, model, device)
        # outputs, labels = engine.evaluate(valid_it, model, device)
        outputs, labels = engine.evaluate(test_it, model, device)
        outputs = torch.Tensor(outputs)
        _, predicted = torch.max(outputs, dim=1)
        
        # use threshold of 0.5
        # using linear layer and no sigmoid
        # you should do this 0.5 threshold after sigmoid
        # outputs = np.array(outputs) >= 0.5

        # calculate accuracy
        accuracy = metrics.accuracy_score(labels, predicted)
        print(f"Epoch: {epoch}, Accuracy Score = {accuracy}")
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accuracy))
        print('-' * 59)

        # # simple early stopping
        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        # else:
        #     early_stopping_counter += 1

        # if early_stopping_counter > 2:
        #     break


if __name__ == "__main__":

    run()
