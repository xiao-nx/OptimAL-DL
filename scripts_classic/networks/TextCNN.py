# textcnn.py 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_pretrained=None):
        '''
        A CNN for text classification.
        Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
        '''
        super(Model, self).__init__()
        
        self.max_length = 512
        self.embed_size = embedding_pretrained.size(1) if embedding_pretrained is not None else 300
        self.kernel_size = [3, 4, 5]
        self.kernel_num = 16
        self.dropout_p = 0.3
        self.num_class = 2

        # define an input emdedding layer
        self.embedding = nn.Embedding(
                                num_embeddings=vocab_size,
                                embedding_dim=self.embed_size)
        if embedding_pretrained is not None:
            print('start load pretrain vector...')
            self.embedding.weight.data.copy_(embedding_pretrained) #load pretrained
        
        # whether train the pretrained embeddings
        self.embedding.weight.requires_grad = False
        
        # a simple CNN
        self.conv1 = nn.Conv2d(1, self.kernel_num, (self.kernel_size[0], self.embed_size))
        self.conv2 = nn.Conv2d(1, self.kernel_num, (self.kernel_size[1], self.embed_size))
        self.conv3 = nn.Conv2d(1, self.kernel_num, (self.kernel_size[2], self.embed_size))
        
        self.dropout = nn.Dropout(self.dropout_p)

        self.fc = nn.Linear(len(self.kernel_size) * self.kernel_num, self.num_class)
        
    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        """
        x: (batch, sentence_length)
        """
        x = x.permute(1,0) # x: [batch, sentence_length]
        x = self.embedding(x) # x: (batch, sentence_length, embed_size)
        x = x.unsqueeze(1) # x: (batch, 1, sentence_length, embed_size)
        
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc(x), dim=1)
        
        return logit       
        

# class Model(nn.Module):
#     def __init__(self, vocab_size, embedding_pretrained=None):
#         '''
#         A CNN for text classification.
#         Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
#         '''
#         super(Model, self).__init__()

# #         sentence_length = vocab_size
#         num_classes = 4
#         embed_dim = 300
#         print(vocab_size)
#         sentence_length = 512
        
#         self.embed_size = embedding_pretrained.size(1) if embedding_pretrained is not None else 300
#         self.num_class = 4

#         # define an input emdedding layer
#         self.embedding = nn.Embedding(
#                                 num_embeddings=vocab_size,
#                                 embedding_dim=self.embed_size)
#         if embedding_pretrained is not None:
#             print('start load pretrain vector...')
#             self.embedding.weight.data.copy_(embedding_pretrained) #load pretrained
        
#         # whether train the pretrained embeddings
#         self.embedding.weight.requires_grad = False
        
#         # a simple CNN
#         self.conv1 = nn.Conv2d(1, 1, (3, self.embed_size))
#         self.conv2 = nn.Conv2d(1, 1, (4, self.embed_size))
#         self.conv3 = nn.Conv2d(1, 1, (5, self.embed_size))

#         self.pool1 = nn.MaxPool2d((sentence_length - 3 + 1, 1))
#         self.pool2 = nn.MaxPool2d((sentence_length - 4 + 1, 1))
#         self.pool3 = nn.MaxPool2d((sentence_length - 5 + 1, 1))

#         self.fc = nn.Linear(3, num_classes)

#     def forward(self, x):
#         """
#         x: (batch, sentence_length)
#         """
#         # bacth_size = x.shape[0]
#         x = self.embedding(x) # x: (sentence_length, batch, embed_size)
#         x = x.transpose(0,1) # x: (batch, sentence_length, embed_size)
#         x = x.unsqueeze(1) # x: (batch, 1, sentence_length, embed_size)
        
#         # Convolution
#         # x: (batch, 1, sentence_length, embed_size)
#         x1 = F.relu((self.conv1(x)).squeeze(3)) # x1: (batch, kernel_num, H_out)
#         x2 = F.relu(self.conv2(x).squeeze(3))
#         x3 = F.relu(self.conv3(x).squeeze(3))

#         # Pooling
#         x1 = F.max_pool1d(x1,x1.size(2)).squeeze(2) # (batch, kernel_num)
#         x2 = F.max_pool1d(x2,x2.size(2)).squeeze(2)
#         x3 = F.max_pool1d(x3,x3.size(2)).squeeze(2)

#         # capture and concatenate the feature
#         x = torch.cat((x1, x2, x3), 1) # (batch, 3 * kernel_num)
        
#         #logits = F.log_softmax(self.fc(x),dim=1)
#         logits = self.fc(x)
        
#         return logits