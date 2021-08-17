import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

'''Attention Is All You Need'''


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_pretrained=None):
        super(Transformer, self).__init__()
        self.embed_size = 300
        self.pad_size = 512 # # 每句话处理成的长度(短填长切)
        self.dropout_p = 0.3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim_model = 300
        self.num_head = 5
        self.hidden = 1024
        self.num_encoder = 2
        self.num_classes = 4

        self.embedding = nn.Embedding(
                                num_embeddings=vocab_size,
                                embedding_dim=self.embed_size)
        
        if embedding_pretrained is not None:
            self.embedding.weight.data.copy_(embedding_pretrained) #load pretrained
        
        # whether train the pretrained embeddings
        self.embedding.weight.requires_grad = False
        

        self.postion_embedding = Positional_Encoding(self.embed_size, self.pad_size, self.dropout_p, self.device)
        self.encoder = Encoder(self.dim_model,self.num_head, self.hidden, self.dropout_p)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(self.num_encoder)])

        self.fc1 = nn.Linear(self.pad_size * self.dim_model, self.num_classes)

    def forward(self, x):
        #print('Transformer: ',x.shape)
        x = x.permute(1,0) # x: [batch, sentence_length]
        out = self.embedding(x)
        
#         out = self.embedding(x[0])
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        #print('Encoder: ',x.shape)
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print('Positional_Encoding: ',x.shape)
        xx = nn.Parameter(self.pe, requires_grad=False)
        #print('xx: ',xx.shape)
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        # dim_model 必须可以正确分为各个头
        assert dim_model % num_head == 0
        # 分头后的维度
        self.dim_head = dim_model // self.num_head
        
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        #print('Multi_Head_Attention: ',x.shape)
        batch_size = x.size(0)
        
        # 分头前的前向网络，获取q、k、v语义
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        
        # 分头
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        
        # 通过缩放点积注意力层
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        '''
        x: (batch_size, seq_len, d_model)
        '''
        #print('Position_wise_Feed_Forward: ',x.shape)
        out = self.fc1(x) 
        out = F.relu(out) # (batch_size, seq_len, d_ff)
        out = self.fc2(out) # (batch_size, seq_len, d_model)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
    
    