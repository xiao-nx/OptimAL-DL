# DPCNN.py Deep Pyramid Convolutional Neural Networksfor Text Categorization
# https://zhuanlan.zhihu.com/p/376903408 

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        
        self.kernel_num = 250
        
        #https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d
        self.half_max_pooling = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),#用0填充
            nn.MaxPool1d(kernel_size=3, stride=2)#1/2池化
        )
        #两个等长卷积层：步长=1，卷积核大小=k,两端补0数量p为(k-1)/2时，卷积后序列长度不变
        #卷积核大小k=3,因此p=1
        self.equal_width_conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.kernel_num),
            nn.ReLU(),
            #padding-->卷积
            nn.Conv1d(self.kernel_num, self.kernel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=self.kernel_num),
            nn.ReLU(),
            nn.Conv1d(self.kernel_num, self.kernel_num, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        half_pooling_x = self.half_max_pooling(x)
        conv_x = self.equal_width_conv(half_pooling_x)
        final_x = half_pooling_x + conv_x
        return final_x
        

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_pretrained=None):
        super(Model, self).__init__()
        self.embed_size = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300
        self.kernel_num = 250
        self.dropout_p = 0.3
        self.num_class = 4
        # Embedding        
        self.embedding = nn.Embedding(
                                num_embeddings=vocab_size,
                                embedding_dim=self.embed_size)
        if embedding_pretrained is not None:
            self.embedding.weight.data.copy_(embedding_pretrained) #load pretrained
        
        # whether train the pretrained embeddings
        self.embedding.weight.requires_grad = False

        #region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(self.embed_size, self.kernel_num, kernel_size=3, stride=1),
            #BatchNormalization
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )

        #两个等长卷积层：步长=1，卷积核大小=k,两端补0数量p为(k-1)/2时，卷积后序列长度不变
        #卷积核大小k=3,因此p=1
        self.equal_width_conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.kernel_num),
            nn.ReLU(),
            #padding-->卷积
            nn.Conv1d(self.kernel_num, self.kernel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=self.kernel_num),
            nn.ReLU(),
            nn.Conv1d(self.kernel_num, self.kernel_num, kernel_size=3, stride=1, padding=1)
        )

        #ResNet_Block
        self.resnet_block = ResnetBlock()

        self.fc = nn.Linear(self.kernel_num, self.num_class)

    def forward(self, x):
        """
        x: (sentence_length, batch)
        """
        x = x.permute(1,0) # x: [batch, sentence_length]
        batch = x.shape[0]
        x = self.embedding(x) # x: (batch_size, max_seq_len, embed_size)
        x = x.permute(0, 2, 1) # x: (batch_size, embed_size, max_seq_len)
        
        x = self.region_embedding(x) #x: (batch_size, 250, max_seq_len-3+1)
        x = self.equal_width_conv(x) #x: (batch_size, 250, max_seq_len-3+1)
        
        while x.size()[2] > 1:#当序列长度大于2时，一直迭代 # 存疑
            x = self.resnet_block(x) #x.shape: (batch_size, 250, 1)
            
        x = x.view(batch, self.kernel_num) # x: (batch_size, 250)
        fc = self.fc(x) #output.shape: (batch_size, 4)
        logit = F.softmax(fc, dim=1)
        
        return logit

    
        


