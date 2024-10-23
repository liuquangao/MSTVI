import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.RevIN import RevIN
import math
from layers.Embed import DataEmbedding_inverted
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GRU_Moudle(nn.Module):
    def __init__(self, seq_len, enc_in, dropout):
        super(GRU_Moudle, self).__init__()
        self.gru = nn.GRU(input_size=enc_in, hidden_size=enc_in, batch_first=True)
        self.norm = nn.LayerNorm([enc_in, seq_len])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        src = x
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out.permute(0, 2, 1)
        return self.dropout(out) + src

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

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        self.pred_len = pred_len

        self.revin_layer = RevIN(configs.enc_in, affine=False, subtract_last=False)
        n1, n2, n3 = configs.n1, configs.n2, configs.n3

        self.gru1 = GRU_Moudle(seq_len=n1, enc_in=configs.enc_in , dropout=configs.dropout)
        self.gru2 = GRU_Moudle(seq_len=n2, enc_in=configs.enc_in , dropout=configs.dropout)
        self.gru3 = GRU_Moudle(seq_len=n3, enc_in=configs.enc_in , dropout=configs.dropout)
        self.gru4 = GRU_Moudle(seq_len=n2, enc_in=configs.enc_in , dropout=configs.dropout)
        self.gru5 = GRU_Moudle(seq_len=n1, enc_in=configs.enc_in , dropout=configs.dropout)

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
        x = x.permute(0,2,1)                            # x: [bsz, nvar, seq_len] 
        # 96--> 512 --> 256 --> 128                            # x: [bsz, nvar, n1]
        down_x1 = self.embed_linear(x)                       # x: [bsz, nvar, n1]

        down_x1_= self.gru1(down_x1)                            # x: [bsz, nvar, n1]
        down_x2 = self.sample_linear1(down_x1_)                          # x: [bsz, nvar, n2]
        down_x2_ = self.gru2(down_x2)                             # x: [bsz, nvar, n2]
        down_x3 = self.sample_linear2(down_x2_)                     # x: [bsz, nvar, n2]

        up_x3 = self.gru3(down_x3)                             # x: [bsz, nvar, n3]

        up_x2_ = self.upsample_linear2(up_x3) + self.down_linear1(down_x2_)
        up_x2 = self.gru4(up_x2_) + self.down_linear2(down_x2)
        up_x1_ = self.upsample_linear1(up_x2) + self.down_linear3(down_x1_)
        up_x1 = self.gru5(up_x1_) + self.down_linear4(down_x1)

        x =  self.pred_linear(up_x1)                         # x: [bsz, nvar, pred_len]
        x = x.permute(0,2,1)[:, :, :N]                        # x: [bsz, pred_len, nvar]
        # denorm
        x = self.revin_layer(x, 'denorm')
        return x 

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
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
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
