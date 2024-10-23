import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.RevIN import RevIN
import math
from layers.Embed import DataEmbedding_inverted
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
        self.d_model = 2*seq_len
        self.n_heads = 8
        self.head_dim = self.d_model // self.n_heads
        self.enc_in = enc_in

        self.mlp1 = nn.Sequential(
            nn.Linear(self.head_dim, 1, bias=True),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(enc_in, 1, bias=True),
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
    def __init__(self, seq_len, enc_in, dropout, activation):
        super(FeedForward, self).__init__()
        self.norm = nn.LayerNorm(seq_len)
        # self.norm = nn.LayerNorm([enc_in,seq_len])
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
        self.ff = FeedForward(seq_len=seq_len, enc_in=enc_in, dropout=dropout, activation=activation)
    def forward(self, x):
        x = self.mhtvi(x)
        x = self.ff(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        self.pred_len = pred_len
        self.ablation = configs.ablation

        self.revin_layer = RevIN(configs.enc_in, affine=False, subtract_last=False)
        n1, n2, n3 = configs.n1, configs.n2, configs.n3

        self.tvi1 = TVI_Module(seq_len=n1, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")
        self.tvi2 = TVI_Module(seq_len=n2, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")
        self.tvi3 = TVI_Module(seq_len=n3, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")
        self.tvi4 = TVI_Module(seq_len=n2, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")
        self.tvi5 = TVI_Module(seq_len=n1, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")

        self.embed_linear = nn.Linear(seq_len, n1)
        self.pred_linear = nn.Linear(n1, pred_len)

        self.sample_linear1 = nn.Linear(n1, n2)
        self.sample_linear2 = nn.Linear(n2, n3)
        self.upsample_linear1 = nn.Linear(n2, n1)
        self.upsample_linear2 = nn.Linear(n3, n2)

        self.down_linear1 = nn.Linear(n2, n2)
        self.down_linear2 = nn.Linear(n2, n2)
        self.down_linear3 = nn.Linear(n1, n1)
        self.down_linear4 = nn.Linear(n1, n1)

        self.dropout = nn.Dropout(configs.dropout)

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, n1, configs.embed, configs.freq,
                                                    configs.dropout)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        self.pred_len = pred_len
        self.ablation = configs.ablation

        self.revin_layer = RevIN(configs.enc_in, affine=False, subtract_last=False)
        n1, n2, n3 = configs.n1, configs.n2, configs.n3

        self.tvi1 = TVI_Module(seq_len=n1, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")
        self.tvi2 = TVI_Module(seq_len=n2, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")
        self.tvi3 = TVI_Module(seq_len=n3, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")
        self.tvi4 = TVI_Module(seq_len=n2, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")
        self.tvi5 = TVI_Module(seq_len=n1, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")

        self.embed_linear = nn.Linear(seq_len, n1)
        self.pred_linear = nn.Linear(n1, pred_len)

        self.sample_linear1 = nn.Linear(n1, n2)
        self.sample_linear2 = nn.Linear(n2, n3)
        self.upsample_linear1 = nn.Linear(n2, n1)
        self.upsample_linear2 = nn.Linear(n3, n2)

        self.down_linear1 = nn.Linear(n2, n2)
        self.down_linear2 = nn.Linear(n2, n2)
        self.down_linear3 = nn.Linear(n1, n1)
        self.down_linear4 = nn.Linear(n1, n1)

        self.dropout = nn.Dropout(configs.dropout)

    def forecast(self, x, x_mark_enc):
        # norm
        x = self.revin_layer(x, 'norm')
        _, _, N = x.shape
        x = x.permute(0, 2, 1)  # x: [bsz, nvar, seq_len]

        down_x1 = self.embed_linear(x)  # x: [bsz, nvar, n1]
        down_x1_, variance_loss1, ortho_loss1 = self.tvi1(down_x1)

        down_x2 = self.sample_linear1(down_x1_)  # x: [bsz, nvar, n2]
        down_x2_, variance_loss2, ortho_loss2 = self.tvi2(down_x2)

        down_x3 = self.sample_linear2(down_x2_)  # x: [bsz, nvar, n2]
        up_x3, variance_loss3, ortho_loss3 = self.tvi3(down_x3)

        up_x2_ = self.upsample_linear2(up_x3) + self.down_linear1(down_x2_)

        tvi4, variance_loss4, ortho_loss4 = self.tvi4(up_x2_)
        up_x2 = tvi4 + self.down_linear2(down_x2)

        up_x1_ = self.upsample_linear1(up_x2) + self.down_linear3(down_x1_)
        tvi5, variance_loss5, ortho_loss5 = self.tvi5(up_x1_)
        up_x1 = tvi5 + self.down_linear4(down_x1)

        x = self.pred_linear(up_x1)  # x: [bsz, nvar, pred_len]
        x = x.permute(0, 2, 1)[:, :, :N]  # x: [bsz, pred_len, nvar]

        x = self.revin_layer(x, 'denorm')

        # 总方差损失和正交性损失
        variance_loss = variance_loss1 + variance_loss2 + variance_loss3 + variance_loss4 + variance_loss5
        ortho_loss = ortho_loss1 + ortho_loss2 + ortho_loss3 + ortho_loss4 + ortho_loss5
        return x, variance_loss, ortho_loss

    ## TODO:待完善
    def imputation(self, x, x_mark_enc, x_dec, x_mark_dec, mask):
        
        return x 


    # TODO:待完善
    def anomaly_detection(self, x):
        return x

    
    # TODO:待完善
    def classification(self, x):
        return x 
 

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :] # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
