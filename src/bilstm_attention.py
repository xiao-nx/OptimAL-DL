# bilstm_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLstmAttention(nn.Module):
    def __init__(self, vocab_size, vec=None):
        """
        paprameter:
        embedding_matrix: numpy array with vectors for all words
        """
        super(BiLstmAttention,self).__init__()
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
        
        if vec is not None:
            self.embedding.weight.data.copy_(vec) #load pretrained

            # whether train the pretrained embeddings
            self.embedding.weight.requires_grad = False

        # a simple bi-directional LSTM attention with hidden size of 128
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
        # 256 = 2 * hidden_size
        self.fc = nn.Linear(256,out_dim)
        # self.dropout = nn.Dropout(0.3)
        
    def attention_network(self,lstm_out,final_hidden):

        hidden = torch.cat((final_hidden[0],final_hidden[1]),dim=1).unsqueeze(2)
        # print("lstm_out:",lstm_out.shape)
        # print("hidden",hidden.shape)
        attention_weights = torch.bmm(lstm_out, hidden).squeeze(2)
        soft_attention_weights = F.softmax(attention_weights, 1) # attention_weights:[batch_size,n_step]
        # print("attention_weights:",attention_weights.shape)
        # [batch_size, n_hidden * num_directions(=2)]
        new_hidden = torch.bmm(lstm_out.transpose(1,2), soft_attention_weights.unsqueeze(2)).squeeze(2)
        # print("new_hidden:",new_hidden.shape)
        
        return new_hidden

    def forward(self,x):
        # pass data through embedding layer
        # the input is just the tokens
        # x:[batch_size,sequence_length]
        input = self.embedding(x) # embedding_data: [batch_size, seq_len, embedding_dim]
        input = input.transpose(0, 1) # input : [sequence_length, batch_size, embedding_dim]

        # move embedding output to lstm
        output, (h_n, c_n) = self.lstm(input)
        # output = output.transpose(0, 1) #output : [batch_size, sequence_length, n_hidden * num_directions(=2)]
        # print("output:",output.shape)
        
        attention_output = self.attention_network(output,h_n) # # attention_output: [batch_size, num_classes]
        # print("attention_output:",attention_output.shape)
        
        # pass through the output layer and return the output
        logits = self.fc(attention_output)
        # print("logits:",logits.shape)
        
        # return linear output
        return logits

    
        


