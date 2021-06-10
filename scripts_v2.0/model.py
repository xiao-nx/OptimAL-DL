# model.py

import config 
import transformers
import torch
import torch.nn as nn 

class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        # we fetch the model from the BERT_PATH defined in config.py
        self.bert_embedding = transformers.BertModel.from_pretrained(
            config.BERT_PATH
        )
        # add a lstm layer
        embed_dim = 768   # bert embedding dim is 768
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            bidirectional=True,
            hidden_size=768//4,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        
        # add a dropout for regularization
        self.lstm_drop = nn.Dropout(0.3)
        # add some network layer
        
        # a simple linear layer for output
        num_class = 4
        self.fc = nn.Linear(768, num_class)


    def forward(self, ids, mask, token_type_ids):
        # BERT in its default settings returns two outputs
        # last hidden state and output of bert pooler layer
        # output of the pooler size (batch_size, hidden_size)
        # hidden size can be 768 or 1024 depending on if using bert base or large respectively
        
        bert_outputs, _ = self.bert_embedding(
            ids,
            attention_mask = mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        #bert_outputs = bert_outputs[-1]
        #print("bert_outputs:",bert_outputs.shape) # o2: a tensot
        # lstm layer
        lstm_outputs, _ = self.lstm(bert_outputs) # [32, 512, 384]
        #print("lstm_outputs1:",lstm_outputs.shape)
        #lstm_outputs = lstm_outputs.permute(1,0,2)
        #print("lstm_outputs2:",lstm_outputs.shape)
        
        # apply mean and max pooling on lstm output 
        avg_pool = torch.mean(lstm_outputs,1) # [32, 384]
        max_pool,_ = torch.max(lstm_outputs,1) # [32, 384]
        #print("avg_pool: ",avg_pool.shape)
        #print("max_pool: ",max_pool.shape)
        
        # concatenate mean and max pooling
        lstm_outputs = torch.cat((avg_pool,max_pool),1)
        #print("lstm_outputs3:",lstm_outputs.shape)
        
        # pass through dropout layer
        lstm_outputs = self.lstm_drop(lstm_outputs)
        
        # pass through linear layer
        logits = self.fc(lstm_outputs) # [batch_size, num_class] = [32, 4]
        #print("logits",logits.shape)
        
        return logits