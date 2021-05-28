# textcnn.py 

import torch
import torch.nn as nn
import torch.nn.functional as F

class textCNN(nn.Module):
    def __init__(self, vocab_size, vec=None):
        super(textCNN, self).__init__()

        sentence_length = vocab_size
        num_classes = 4
        embed_dim = 300

        # define an input emdedding layer
        self.embedding = nn.Embedding(
                                num_embeddings=vocab_size,
                                embedding_dim=embed_dim)
        if vec is not None:
            self.embedding.weight.data.copy_(vec) #load pretrained
        
        # whether train the pretrained embeddings
        self.embedding.weight.requires_grad = False
        
        # a simple CNN
        self.conv1 = nn.Conv2d(1, 1, (3, embed_dim))
        self.conv2 = nn.Conv2d(1, 1, (4, embed_dim))
        self.conv3 = nn.Conv2d(1, 1, (5, embed_dim))

        self.pool1 = nn.MaxPool2d((sentence_length - 3 + 1, 1))
        self.pool2 = nn.MaxPool2d((sentence_length - 4 + 1, 1))
        self.pool3 = nn.MaxPool2d((sentence_length - 5 + 1, 1))

        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        """
        x: (batch, sentence_length)
        """
        bacth_size = x.shape[0]
        x = self.embedding(x) # x: (sentence_length, batch, embed_dim)
        x = x.transpose(0,1) # x: (batch, sentence_length, embed_dim)
        x = x.unsqueeze(1) # x: (batch, 1, sentence_length, embed_dim)
        
        # Convolution
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = F.relu((self.conv1(x)).squeeze(3))
        # print("x1 before",x1.shape) # x1: (batch, kernel_num, H_out)
        x2 = F.relu(self.conv2(x).squeeze(3))
        x3 = F.relu(self.conv3(x).squeeze(3))

        # Pooling
        x1 = F.max_pool1d(x1,x1.size(2)).squeeze(2) # (batch, kernel_num)
        # print("x1 after",x1.shape)
        x2 = F.max_pool1d(x2,x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3,x3.size(2)).squeeze(2)

        # capture and concatenate the feature
        x = torch.cat((x1, x2, x3), 1) # (batch, 3 * kernel_num)
        
        logits = F.log_softmax(self.fc(x),dim=1)

        return logits