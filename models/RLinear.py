# 效果巨好的
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.RevIN import RevIN
import math
from layers.TVI import TVI_Module

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.task_name = configs.task_name
#         seq_len = configs.seq_len
#         pred_len = configs.pred_len
#         self.pred_len = pred_len

#         self.revin_layer = RevIN(configs.enc_in, affine=False, subtract_last=False)
#         n1, n2, n3 = configs.n1, configs.n2, configs.n3

#         self.embed_linear = nn.Linear(seq_len, n1)
#         self.pred_linear1 = nn.Linear(n1, pred_len)

#         self.sample_linear1 = nn.Linear(n1, n2)
#         self.sample_linear2 = nn.Linear(n2, n3)
#         self.upsample_linear1 = nn.Linear(n2, n1)
#         self.upsample_linear2 = nn.Linear(n3, n2)

#         self.norm1 = nn.LayerNorm(n1)
#         self.norm2 = nn.LayerNorm(n2)

#     def forecast(self, x):

#         # norm
#         x = self.revin_layer(x, 'norm')
#         x = x.permute(0,2,1)                            # x: [bsz, nvar, seq_len] 
#         # 96--> 512 --> 256 --> 128
#         x1 = self.embed_linear(x)                       # x: [bsz, nvar, n1]
#         # 下采样
#         x2 = self.sample_linear1(x1)                   # x: [bsz, nvar, n2]
#         x3 = self.sample_linear2(x2)                   # x: [bsz, nvar, n2]

#         # 上采样混合阶段
#         src2_ = x2 + self.upsample_linear2(x3)           # src2_: [bsz, nvar, n2]
#         src2_ = self.norm2(src2_)
#         src1_ = x1 + self.upsample_linear1(src2_)         # x1_: [bsz, nvar, 512]
#         src1_ = self.norm1(src1_)

#         x =  self.pred_linear1(src1_)                   # x: [bsz, nvar, pred_len]
#         x = x.permute(0,2,1)                            # x: [bsz, pred_len, nvar]
#         # denorm
#         x = self.revin_layer(x, 'denorm')
#         return x 

#     ## TODO:待完善
#     def imputation(self, x, x_mark_enc, x_dec, x_mark_dec, mask):
        
#         return x 


#     # TODO:待完善
#     def anomaly_detection(self, x):
#         return x

    
#     # TODO:待完善
#     def classification(self, x):
#         return x 
 

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             dec_out = self.forecast(x_enc)
#             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
#         if self.task_name == 'imputation':
#             dec_out = self.imputation(
#                 x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
#             return dec_out  # [B, L, D]
#         if self.task_name == 'anomaly_detection':
#             dec_out = self.anomaly_detection(x_enc)
#             return dec_out  # [B, L, D]
#         if self.task_name == 'classification':
#             dec_out = self.classification(x_enc)
#             return dec_out  # [B, N]
#         return None


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        self.pred_len = pred_len

        self.revin_layer = RevIN(configs.enc_in, affine=False, subtract_last=False)
        n1, n2, n3 = configs.n1, configs.n2, configs.n3

        self.pred_linear = nn.Linear(seq_len, pred_len)
        self.tvi = TVI_Module(seq_len=pred_len, enc_in=configs.enc_in, dropout=configs.dropout, activation="relu")

    def forecast(self, x):

        # norm
        x = self.revin_layer(x, 'norm')
        x = x.permute(0,2,1)                            # x: [bsz, nvar, seq_len]
        x =  self.pred_linear(x)                        # x: [bsz, nvar, pred_len]
        x = self.tvi(x)
        x = x.permute(0,2,1)                            # x: [bsz, pred_len, nvar]
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
            dec_out = self.forecast(x_enc)
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
