# lstm.py

import torch
import torch.nn as nn
 
class LSTM(nn.Module):
    def __init__(self, vocab_size, vec=None):
        super(LSTM, self).__init__()
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
             
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=128,
            num_layers=1,
            bias=True,
            bidirectional=False,
            batch_first=True,
        )
        
        # add BN layer
        self.bn = nn.BatchNorm1d(128)

        # output layer is a linear layer
        # only one output
        # input (512)=128+128 for mean and same for max-pooling
        out_dim = 4
        self.fc = nn.Linear(128,out_dim)

    def forward(self,x):
        
        word_embs = self.word_embedding(x) # [batch_size,seq_len,feature_dim]
        # print('word_embs: ',word_embs.shape) # [seq_len, batch_size, embs_dim]
        
        x,_ = self.lstm(word_embs)
        x = x.permute(1,0,2) # [batch_size, seq_len, filter_dim]

        
        # applymax pooling on lstm output
        max_pool,_ = torch.max(x,1)
        # pass through the output layer and return the output
        logits = self.fc(max_pool)
        # print("logits:",logits.shape)
        # return linear output
        return logits
        


