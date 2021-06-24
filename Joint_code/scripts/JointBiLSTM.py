# lstm.py

import torch
import torch.nn as nn

class JointBiLSTM(nn.Module):
    def __init__(self, vocab_size, vec=None):
        super(JointBiLSTM, self).__init__()
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
        self.pos1_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos2_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
        # load pretrained embedding vectors
        if vec is not None:
            self.word_embedding.weight.data.copy_(vec)
            self.pos1_embedding.weight.data.copy_(vec) 
            self.pos2_embedding.weight.data.copy_(vec) 
        
        # whether train the pretrained embeddings
        self.word_embedding.weight.requires_grad = False
        self.pos1_embedding.weight.requires_grad = False
        self.pos2_embedding.weight.requires_grad = False
        

        # a simple bi-directional LSTM with hidden size of 128
        self.bilstm = nn.LSTM(
                            input_size=embed_dim * 3,
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
        self.fc = nn.Linear(512,out_dim)

    def forward(self,x):
        batch_sents = x['texts']
        batch_pos1s = x['entity_1']
        batch_pos2s = x['entity_2']

        word_embs = self.word_embedding(batch_sents) # [batch_size,seq_len,feature_dim]
        pos1_embs = self.pos1_embedding(batch_pos1s)
        pos2_embs = self.pos2_embedding(batch_pos2s)

        input_feature = torch.cat([word_embs,pos1_embs,pos2_embs],dim=2)  # batch_size x seq_len x feature_dim

        # move embedding output to lstm
        x,_ = self.bilstm(input_feature)
        x = x.permute(1,0,2)
        # print('x: ',x.shape) # ([4, 276, 256])

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

#### 
class JointBiLSTM2(nn.Module):
    def __init__(self, vocab_size, vec=None):
        super(JointBiLSTM2, self).__init__()
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
        self.pos1_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pos2_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
        # load pretrained embedding vectors
        if vec is not None:
            self.word_embedding.weight.data.copy_(vec)
            self.pos1_embedding.weight.data.copy_(vec) 
            self.pos2_embedding.weight.data.copy_(vec) 
        
        # whether train the pretrained embeddings
        self.word_embedding.weight.requires_grad = False
        self.pos1_embedding.weight.requires_grad = False
        self.pos2_embedding.weight.requires_grad = False
        

        # a simple bi-directional LSTM with hidden size of 128
        self.bilstm = nn.LSTM(
                            input_size=embed_dim * 3,
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
        batch_sents = x['texts']
        batch_pos1s = x['entity_1']
        batch_pos2s = x['entity_2']

        word_embs = self.word_embedding(batch_sents) # [batch_size,seq_len,feature_dim]
        pos1_embs = self.pos1_embedding(batch_pos1s)
        pos2_embs = self.pos2_embedding(batch_pos2s)

        input_feature = torch.cat([word_embs,pos1_embs,pos2_embs],dim=2)  # batch_size x seq_len x feature_dim

        # move embedding output to lstm
        x,_ = self.bilstm(input_feature)
        x = x.permute(1,0,2)

        # apply max pooling on lstm output
        max_pool,_ = torch.max(x,1)
        
        # pass through the output layer and return the output
        logits = self.fc(max_pool)
        # return linear output
        return logits

        


