# train.py

import io
import torch
# import torch.utils.data as Data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd.grad_mode import no_grad 
import torch.nn as nn
import numpy as np
import pandas as pd
import time

from sklearn import metrics
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import dataset
import config 
import engine
from model import BertBaseUncased


def train():
    """
    This function trains the model
    read the training file and fill NaN values with "none"
    can also choose to drop NaN values in this specific dataset
    """

    train_df = pd.read_csv(config.TRAIN_DATASET_PATH)#.fillna("none")
    validation_df = pd.read_csv(config.TEST_DATASET_PATH)#.fillna("none")

    # initialize BERTDataset from dataset.py
    train_dataset = dataset.BertDataset(
        text=train_df.text.values,
        label=train_df.label.values
    )

    # create training dataloader
    train_data_loader = DataLoader(
        train_dataset, # the training samples
        sampler = RandomSampler(train_dataset), # Select batches randomly
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    # initialize BERTDataset from dataset.py for validaton dataset
    validation_dataset = dataset.BertDataset(
        text=validation_df.text.values,
        label=validation_df.label.values
    )

    # create validation data loader
    validation_data_loader = DataLoader(
        validation_dataset, # the validation samples.
        batch_size= config.VALIDATION_BATCH_SIZE, # Pull out batches sequentially.
        num_workers=1
    )

    # initialize the cuda device
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print("device:",device)
    # load model and send it to the device
    model = BertBaseUncased()
    model.to(device)

    # create parameters we want to optimize
    # we generally do not use any decay for bias and weight layers
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay":0.001,
        },
        {
            "params": [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay":0.0,
        },
    ]

    # calculate the number of training steps
    # this is used by scheduler
    num_train_steps = int(len(train_df) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    # AdamW optimizer
    # AdamW is the most widely used optimzer for transformer based networks
    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)

    # fetch a scheduler
    # you can also try using reduce lr on plateau
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # if you have multiple GPUs
    # model model to DataParallel to use multiple GPUs
    # model = nn.DataParallel(model)

    # start training the epochs
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        # train one epoch
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        # validation epoch
        outputs, labels = engine.eval_fn(validation_data_loader, model, device)
        
        outputs = torch.Tensor(outputs)
        _, predicted = torch.max(outputs, dim=1)

        # outputs = np.array(outputs) >= 0.5
        
        # calculate accuracy
        accuracy = metrics.accuracy_score(predicted,labels)
        # print(f"Accuracy Score = {accuracy}")
        print(f"Epoch: {epoch}, Accuracy Score = {accuracy}")
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accuracy))
        print('-' * 59)

        if accuracy > best_accuracy:
            torch.save(model.state_dict(),config.MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    
    # train_df = pd.read_csv("../inputs/train_cleaned_dataset.csv",encoding = "ISO-8859-1")
    # valid_df = pd.read_csv("../inputs/expert_dataset.csv",encoding = "ISO-8859-1")
    
    train()
