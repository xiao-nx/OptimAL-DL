# model.py

import config 
import transformers
import torch.nn as nn 

class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased, self).__init__()
        # we fetch the model from the BERT_PATH defined in config.py
        self.bert = transformers.BertModel.from_pretrained(
            config.BERT_PATH
        )
        # add a dropout for regularization
        self.bert_drop = nn.Dropout(0.3)
        # a simple linear layer for output
        num_class = 4
        self.out = nn.Linear(768, num_class)


    def forward(self, ids, mask, token_type_ids):
        # BERT in its default settings returns two outputs
        # last hidden state and output of bert pooler layer
        # output of the pooler size (batch_size, hidden_size)
        # hidden size can be 768 or 1024 depending on if using bert base or large respectively
        
        _, o2 = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        # print("o2:",o2) # o2: a tensot
        # pass through dropout layer
        bo = self.bert_drop(o2)
        # pass through linear layer
        output = self.out(bo)
        
        # return output
        return output