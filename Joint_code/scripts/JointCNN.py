# textcnn.py 

import torch
import torch.nn as nn
import torch.nn.functional as F
import config 

class JointCNN(nn.Module):
    def __init__(self, vocab_size, vec=None):
        super(JointCNN, self).__init__()
        
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
        
        # params
        filter_num = 128
#         filters = [2,3,4,5]
        filters = [3]
        feature_dim = embed_dim * 3
        max_len = 500
        
        # encode sentence level features via cnn
        self.convs = nn.Sequential(nn.Conv1d(in_channels=feature_dim,
                                            out_channels=filter_num,
                                            kernel_size=3),
                                   nn.ReLU())
        filter_dim = filter_num * len(filters)
        labels_num = 4
        # output layer
        self.fc = nn.Linear(filter_dim*2, labels_num)
        
    def forward(self, x):
        """
        x: (batch, sentence_length)
        """
        # bacth_size = x.shape[0]
        # print('x:  ',x)
        batch_sents = x['texts']
        batch_pos1s = x['entity_1']
        batch_pos2s = x['entity_2']
        
        # embedding
        word_embs = self.word_embedding(batch_sents) # [batch_size,seq_len,feature_dim]
        pos1_embs = self.pos1_embedding(batch_pos1s)
        pos2_embs = self.pos2_embedding(batch_pos2s)
        
        # concat
        input_feature = torch.cat([word_embs,pos1_embs,pos2_embs],dim=2)  # batch_size x seq_len x feature_dim
#         print('cat: ',input_feature.shape) # cat:  torch.Size([809, 64, 900])
        input_feature = input_feature.permute(1,2,0)
#         print('input_feature before:  ',input_feature.shape) # input_feature before:   torch.Size([809, 900, 64])
        
        conv_out = self.convs(input_feature) #input_feature shoule be [batch_size, filter_num, seq_len]
#         print('conv_out:', conv_out.shape) # conv_out: torch.Size([64, 128, 857])
        
        conv_out = conv_out.permute(0,2,1)
        
        # poo input size should be x:  torch.Size([64, 582, 128])
        
        # apply mean and max pooling on conv output # 
        avg_pool = torch.mean(conv_out,1) # conv_out shoud be torch.Size [batch_size, seq_len, filter_num]
        max_pool,_ = torch.max(conv_out,1)

        # concatenate mean and max pooling
        # this is why size is 512
        # 128 for each direction = 256
        # avg_pool=256 and max_pool=256
        out = torch.cat((avg_pool,max_pool),1)
#         print("out_1:",out.shape)
        
        # pass through the output layer and return the output
        logits = self.fc(out)
        # print("logits:",logits.shape)
        # return linear output
        return logits
    
### 
class JointCNN2(nn.Module):
    def __init__(self, vocab_size, vec=None):
        super(JointCNN2, self).__init__()
        
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
        
        # params
        filter_num = 128
#         filters = [2,3,4,5]
        filters = [3]
        feature_dim = embed_dim * 3
        max_len = 500
        
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
        x: (batch, sentence_length)
        """
        # bacth_size = x.shape[0]
        # print('x:  ',x)
        batch_sents = x['texts']
        batch_pos1s = x['entity_1']
        batch_pos2s = x['entity_2']
        
        # embedding
        word_embs = self.word_embedding(batch_sents) # [batch_size,seq_len,feature_dim]
        pos1_embs = self.pos1_embedding(batch_pos1s)
        pos2_embs = self.pos2_embedding(batch_pos2s)
        
        # concat
        input_feature = torch.cat([word_embs,pos1_embs,pos2_embs],dim=2)  # batch_size x seq_len x feature_dim
        input_feature = input_feature.permute(1,2,0)
        
        conv_out = self.convs(input_feature) #input_feature shoule be [batch_size, filter_num, seq_len]
        
        conv_out = conv_out.permute(0,2,1)
        
        # apply max pooling on conv output # 
        max_pool,_ = torch.max(conv_out,1)

        # pass through the output layer and return the output
        logits = self.fc(max_pool)
        # return linear output
        return logits