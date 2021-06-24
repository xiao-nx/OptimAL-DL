# textcnn.py 

import torch
import torch.nn as nn
import torch.nn.functional as F
import config 

class CNN(nn.Module):
    def __init__(self, vocab_size, vec=None):
        super(CNN, self).__init__()
        
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
        
        # params
        filter_num = 128
        filters = [3]
        feature_dim = embed_dim
        
        # encode sentence level features via cnn
        self.convs = nn.Sequential(nn.Conv1d(in_channels=feature_dim,
                                            out_channels=filter_num,
                                            kernel_size=3),
                                   nn.ReLU())
        filter_dim = filter_num * len(filters)
        labels_num = 4
        # output layer
        self.fc = nn.Linear(filter_dim, labels_num)

    def forward(self, x):
        """
        x: (sentence_length, batch)
        """
        
        # embedding
#         print('x input: ', x.shape)
        x = x.permute(1,0)
        word_embs = self.word_embedding(x) # word_embs [batch_size,seq_len,feature_dim]
        
#         print('word_embs: ',word_embs.shape)
        input_feature = word_embs.permute(0,2,1)
#         print('input_feature: ',input_feature.shape)
        
        conv_out = self.convs(input_feature) #input_feature shoule be [batch_size, filter_num, seq_len]
        
        conv_out = conv_out.permute(0,2,1)
        
        # apply max pooling on conv output # 
        max_pool,_ = torch.max(conv_out,1)

        # pass through the output layer and return the output
        logits = self.fc(max_pool)
        # return linear output
        return logits
    
    