import torch.nn as nn
import math
import torch

def get_activation(activation):
    if activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'silu':
        return nn.SiLU()
    else:
        print('None activation')
        return nn.Identity()

class MHTVI(nn.Module):
    def __init__(self, seq_len, enc_in, dropout):
        super(MHTVI, self).__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.d_model = seq_len
        self.n_heads = 8
        self.head_dim = self.d_model // self.n_heads
        self.enc_in = enc_in

        self.mlp1 = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim, bias=True),
            get_activation("relu"),
            nn.Linear(self.head_dim, 1, bias=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(enc_in, 1, bias=True),
            get_activation("relu"),
        )

        self.linear1 = nn.Linear(enc_in, enc_in)
        self.linear2 = nn.Linear(self.head_dim,self.head_dim)
        self.in_projection = nn.Linear(seq_len, 3 * self.d_model, bias=True)
        self.out_projection = nn.Linear(2 * self.d_model, seq_len, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [bsz, nvar, seq_len]
        B, N, _ = x.shape
        H = self.n_heads
        scale = 1. / math.sqrt(self.head_dim * self.enc_in)
        src = x

        # 计算 qkv
        qkv = self.in_projection(self.norm(x))
        q, k, v = torch.split(qkv, self.d_model, dim=-1)
        q = q.contiguous().view(B, H, N, -1)
        k = k.contiguous().view(B, H, N, -1)
        v = v.contiguous().view(B, H, N, -1)

        score1 = self.mlp1(q)                                       # score1: [bsz, n_heads, nvar, 1]
        score2 = self.mlp2(k.permute(0,1,3,2)).permute(0,1,3,2)     # score2: [bsz, n_heads, 1, head_dim]
        score = (score1 @ score2) * scale                           # score: [bsz, n_heads, nvar, head_dim]  
        v1 = score * v                                              # v: [bsz, n_heads, nvar, head_dim]
        v1 = v1.reshape(B, N, -1)                                   # v: [bsz, nvar, seq_len]
        
        score1 = self.mlp1(k)                                       # score1: [bsz, n_heads, nvar, 1]
        score2 = self.mlp2(q.permute(0,1,3,2)).permute(0,1,3,2)     # score2: [bsz, n_heads, 1, head_dim]
        score = (score1 @ score2) * scale                           # score: [bsz, n_heads, nvar, head_dim]
        v2 = score * v                                              # v: [bsz, n_heads, nvar, head_dim]
        v2 = v2.reshape(B, N, -1)                                   # v: [bsz, nvar, seq_len]
        
        v = torch.concat([v1,v2],dim=-1)
        out = self.out_projection(v)
        x = self.dropout(out) + src

        return x

# Patch混合（建模时间维度的关系）    
class FeedForward(nn.Module):
    def __init__(self, seq_len, dropout):
        super(FeedForward, self).__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(seq_len, 4*seq_len, bias=True),
            get_activation("relu"),
            nn.Linear(4*seq_len, seq_len, bias=True),
        )
    def forward(self, x):                                                       # x: [bsz, nvar, seq_len]
        src = x
        x = self.ff(self.norm(x))
        x = self.dropout(x) + src
        return x

class TVI_Module(nn.Module):
    def __init__(self, seq_len, enc_in, dropout, activation):
        super(TVI_Module, self).__init__()
        self.mhtvi = MHTVI(seq_len=seq_len, enc_in=enc_in, dropout=dropout)
        self.ff = FeedForward(seq_len=seq_len, dropout=dropout )
    def forward(self, x):
        x = self.mhtvi(x)
        x = self.ff(x)
        return x