# lstm.py

import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, vec=None):
        super(BiLSTM, self).__init__()
        """
        paprameter:
        embedding_matrix: numpy array with vectors for all words
        """
        sentence_length = vocab_size
        num_classes = 4
        embed_dim = 300

        # define an input emdedding layer
        self.word_embedding = nn.Embedding(
                                num_embeddings=vocab_size,
                                embedding_dim=embed_dim)
        
        # load pretrained embedding vectors
        if vec is not None:
            self.word_embedding.weight.data.copy_(vec)
        
        # whether train the pretrained embeddings
        self.word_embedding.weight.requires_grad = False   

        # a simple bi-directional LSTM with hidden size of 128
        self.bilstm = nn.LSTM(
                            input_size=embed_dim,
                            hidden_size=128,
                            num_layers=1,
                            bias=True,
                            bidirectional=True,
                            batch_first=True,
                            )
        # bn layer                    
        self.bn = nn.BatchNorm1d(256)
        # maxpool layer
        # self.maxpool = nn.MaxPool1d(3, stride=2)
                        
        # output layer is a linear layer
        # only one output
        # input (512)=128+128 for mean and same for max-pooling
        out_dim = 4
        self.fc = nn.Linear(256,out_dim)

    def forward(self,x):

        word_embs = self.word_embedding(x) # [batch_size,seq_len,feature_dim]
        
        # move embedding output to lstm
        x,_ = self.bilstm(word_embs)
        x = x.permute(1,0,2)

        # apply max pooling on lstm output
        max_pool,_ = torch.max(x,1)
        
        # pass through the output layer and return the output
        logits = self.fc(max_pool)
        # return linear output
        return logits

        


