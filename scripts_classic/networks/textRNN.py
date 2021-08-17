# lstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_pretrained=None):
        """
        paprameter:
        embedding_matrix: numpy array with vectors for all words
        """
        super(Model,self).__init__()
        # number of words = number of rows in embedding matrix
        # num_words = embedding_matrix.shape[0]
        
        self.embed_size = embedding_pretrained.size(1) if embedding_pretrained is not None else 300
        self.hidden_size = 128
        self.num_layers = 2
        self.num_class = 4

        # Embedding        
        self.embedding = nn.Embedding(
                                num_embeddings=vocab_size,
                                embedding_dim=self.embed_size)
        if embedding_pretrained is not None:
            self.embedding.weight.data.copy_(embedding_pretrained) #load pretrained
        
        # whether train the pretrained embeddings
        self.embedding.weight.requires_grad = False

        # a simple bi-directional LSTM with hidden size of 128
        self.lstm = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            bidirectional=True,
            batch_first=True,
        )

        # output layer is a linear layer
        # input (512)=128+128 for mean and same for max-pooling
        self.fc = nn.Linear(4*self.hidden_size,self.num_class)

    def forward(self,x):
        # pass data through embedding layer
        # the input is just the tokens
        x = x.permute(1,0) # x: [batch, sentence_length]
        x = self.embedding(x)

        # move embedding output to lstm
        x,_ = self.lstm(x)

        # apply mean and max pooling on lstm output
        avg_pool = torch.mean(x,1)
        max_pool,_ = torch.max(x,1)

        # concatenate mean and max pooling
        # this is why size is 512
        # 128 for each direction = 256
        # avg_pool=256 and max_pool=256
        out = torch.cat((avg_pool,max_pool),1)
        
        # pass through the output layer and return the output
        fc = self.fc(out)
        
        logits = F.log_softmax(fc, dim=1)
        return logits

    
        


