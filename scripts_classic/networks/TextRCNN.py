# TextRCNN
'''Recurrent Convolutional Neural Networks for Text Classification'''
# https://blog.csdn.net/MrR1ght/article/details/105516472

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_pretrained=None):
        super(Model, self).__init__()
        
        self.embed_size = embedding_pretrained.size(1) if embedding_pretrained is not None else 300
        self.hidden_size = 256
        self.num_layers = 1
        self.dropout_p = 0.3
        self.pad_size = 32
        self.num_class = 4
        
        self.embedding = nn.Embedding(
                                num_embeddings=vocab_size,
                                embedding_dim=self.embed_size)
        
        if embedding_pretrained is not None:
            self.embedding.weight.data.copy_(embedding_pretrained) #load pretrained
        
        # whether train the pretrained embeddings
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.dropout_p)
        self.maxpool = nn.MaxPool1d(self.pad_size)
        self.fc = nn.Linear(self.hidden_size * 2 + self.embed_size, self.num_class)

    def forward(self, x):
        """
        x: (sentence_length, batch)
        """
        x = x.permute(1,0) # x: [batch, sentence_length]

        embed = self.embedding(x)  # x: [batch_size, seq_len, embed_size]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out) # x: [batch_size, seq_len, hidden_size * 2 + embed_size]
        out = out.permute(0, 2, 1)
        out = self.maxpool(out)
        out = squeeze(out)
        fc = self.fc(out)
        
        logit = F.log_softmax(fc, dim=1)
        
        return logit