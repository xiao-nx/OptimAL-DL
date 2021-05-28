# lstm.py

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, vocab_size, vec=None):
        """
        paprameter:
        embedding_matrix: numpy array with vectors for all words
        """
        super(LSTM,self).__init__()
        # number of words = number of rows in embedding matrix
        # num_words = embedding_matrix.shape[0]
        num_words = vocab_size

        # dimension of embedding is num of columns in the matrix
        # embed_dim = embedding_matrix.shape[1]
        embed_dim = 300

        # define an input embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embed_dim
        )

        # # embeddinf matrix is used as weights of the embedding layer
        # self.embedding.weight = nn.Parameter(
        #     torch.tensor(
        #         embedding_matrix,
        #         dtype=torch.float32
        #     )
        # )

        if vec is not None:
            self.embedding.weight.data.copy_(vec) #load pretrained

        # whether train the pretrained embeddings
        self.embedding.weight.requires_grad = False

        # a simple bi-directional LSTM with hidden size of 128
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=128,
            num_layers=1,
            bias=True,
            bidirectional=True,
            batch_first=True,
        )

        # output layer is a linear layer
        # only one output
        # input (512)=128+128 for mean and same for max-pooling
        out_dim = 4
        self.fc = nn.Linear(512,out_dim)

    def forward(self,x):
        # pass data through embedding layer
        # the input is just the tokens
        x = self.embedding(x)

        # move embedding output to lstm
        x,_ = self.lstm(x)
        x = x.permute(1,0,2)

        # apply mean and max pooling on lstm output
        avg_pool = torch.mean(x,1)
        max_pool,_ = torch.max(x,1)

        # concatenate mean and max pooling
        # this is why size is 512
        # 128 for each direction = 256
        # avg_pool=256 and max_pool=256
        out = torch.cat((avg_pool,max_pool),1)
        # print("out_1:",out.shape)
        
        # pass through the output layer and return the output
        logits = self.fc(out)
        # print("logits:",logits.shape)
        # return linear output
        return logits

    
        


