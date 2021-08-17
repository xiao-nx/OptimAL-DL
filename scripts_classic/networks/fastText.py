# https://zhuanlan.zhihu.com/p/73176084

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_pretrained=None):
        super(Model, self).__init__()
        
        self.n_gram_vocab = 250499 ## 自定义大小
        self.embed_size = embedding_pretrained.size(1) if embedding_pretrained is not None else 300
        self.dropout_p = 0.3 
        self.hidden_size = 256  
        self.num_classes = 4
        
        self.embedding = nn.Embedding(
                                num_embeddings=vocab_size,
                                embedding_dim=self.embed_size)
        
        if embedding_pretrained is not None:
            self.embedding.weight.data.copy_(embedding_pretrained) #load pretrained
        
        # whether train the pretrained embeddings
        self.embedding.weight.requires_grad = False
        
        self.embedding_ngram2 = nn.Embedding(self.n_gram_vocab, self.embed_size)
        self.embedding_ngram3 = nn.Embedding(self.n_gram_vocab, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc1 = nn.Linear(self.embed_size * 3, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        """
        x: (sentence_length, batch)
        """
        x = x.permute(1,0) # x: [batch, sentence_length]

        out_word = self.embedding(x) # x: [batch_size, seq_len, embed_size]
        out_bigram = self.embedding_ngram2(x) # x: [batch_size, seq_len, embed_size]
        out_trigram = self.embedding_ngram3(x) # x: [batch_size, seq_len, embed_size]
        
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        fc2 = self.fc2(out)
        
        logit = F.log_softmax(fc2, dim=1)
        
        return logit
    
    