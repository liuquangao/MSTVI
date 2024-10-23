import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.TVI import TVI_Module


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        
        self.tvi = TVI_Module(seq_len=configs.d_model, enc_in=configs.enc_in+4, dropout=configs.dropout, activation="relu")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.tvi(enc_out)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.Transformer_EncDec import Encoder, EncoderLayer
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
# from layers.Embed import DataEmbedding_inverted
# import numpy as np


# class Model(nn.Module):
#     """
#     Paper link: https://arxiv.org/abs/2310.06625
#     """

#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention

#         n1, n2, n3 = 512, 256, 128
#         # Embedding
#         self.enc_embedding = DataEmbedding_inverted(configs.seq_len, n1, configs.embed, configs.freq,
#                                                     configs.dropout)
#         # # Encoder
#         # self.encoder = Encoder(
#         #     [
#         #         EncoderLayer(
#         #             AttentionLayer(
#         #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#         #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#         #             configs.d_model,
#         #             configs.d_ff,
#         #             dropout=configs.dropout,
#         #             activation=configs.activation
#         #         ) for l in range(configs.e_layers)
#         #     ],
#         #     norm_layer=torch.nn.LayerNorm(configs.d_model)
#         # )
#         # Encoder
#         self.encoder1 = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), n1, configs.n_heads),
#                     n1,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 )
#             ],
#             norm_layer=torch.nn.LayerNorm(n1)
#         )

#         self.encoder2 = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), n2, configs.n_heads),
#                     n2,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 )
#             ],
#             norm_layer=torch.nn.LayerNorm(n2)
#         )

#         self.encoder3 = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), n3, configs.n_heads),
#                     n3,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 )
#             ],
#             norm_layer=torch.nn.LayerNorm(n3)
#         )

#         # Decoder
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             self.projection = nn.Linear(n1, configs.pred_len, bias=True)
#         if self.task_name == 'imputation':
#             self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
#         if self.task_name == 'anomaly_detection':
#             self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
#         if self.task_name == 'classification':
#             self.act = F.gelu
#             self.dropout = nn.Dropout(configs.dropout)
#             self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)


#         self.sample_linear1 = nn.Linear(n1, n2)
#         self.sample_linear2 = nn.Linear(n2, n3)
#         self.upsample_linear1 = nn.Linear(n2, n1)
#         self.upsample_linear2 = nn.Linear(n3, n2)

#         self.norm1 = nn.LayerNorm([configs.enc_in+4,n1])
#         self.norm2 = nn.LayerNorm([configs.enc_in+4,n2])

#         # self.norm1 = nn.LayerNorm(n1)
#         # self.norm2 = nn.LayerNorm(n2)

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         # Normalization from Non-stationary Transformer
#         means = x_enc.mean(1, keepdim=True).detach()
#         x_enc = x_enc - means
#         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x_enc /= stdev

#         _, _, N = x_enc.shape

#         # Embedding
#         x1 = self.enc_embedding(x_enc, x_mark_enc)

#         x1_, _= self.encoder1(x1)                            # x: [bsz, nvar, n1]

#         # 下采样
#         x2 = self.sample_linear1(x1_)                   # x: [bsz, nvar, n2]
#         x2_,_ = self.encoder2(x2)                           # x: [bsz, nvar, n2]
#         x3 = self.sample_linear2(x2_)                   # x: [bsz, nvar, n2]
#         x3_,_ = self.encoder3(x3)                           # x: [bsz, nvar, n3]

#         # 上采样混合阶段
#         src2_ = x2 + self.upsample_linear2(x3_)         # src2_: [bsz, nvar, n2]
#         src2_ = self.norm2(src2_)

#         src1_ = x1 + self.upsample_linear1(src2_)       # x1_: [bsz, nvar, 512]
#         enc_out = self.norm1(src1_)    

#         dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
#         # De-Normalization from Non-stationary Transformer
#         dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#         dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#         return dec_out

#     def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
#         # Normalization from Non-stationary Transformer
#         means = x_enc.mean(1, keepdim=True).detach()
#         x_enc = x_enc - means
#         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x_enc /= stdev

#         _, L, N = x_enc.shape

#         # Embedding
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
#         # De-Normalization from Non-stationary Transformer
#         dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#         dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#         return dec_out

#     def anomaly_detection(self, x_enc):
#         # Normalization from Non-stationary Transformer
#         means = x_enc.mean(1, keepdim=True).detach()
#         x_enc = x_enc - means
#         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x_enc /= stdev

#         _, L, N = x_enc.shape

#         # Embedding
#         enc_out = self.enc_embedding(x_enc, None)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
#         # De-Normalization from Non-stationary Transformer
#         dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#         dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
#         return dec_out

#     def classification(self, x_enc, x_mark_enc):
#         # Embedding
#         enc_out = self.enc_embedding(x_enc, None)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         # Output
#         output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
#         output = self.dropout(output)
#         output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
#         output = self.projection(output)  # (batch_size, num_classes)
#         return output

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
#         if self.task_name == 'imputation':
#             dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
#             return dec_out  # [B, L, D]
#         if self.task_name == 'anomaly_detection':
#             dec_out = self.anomaly_detection(x_enc)
#             return dec_out  # [B, L, D]
#         if self.task_name == 'classification':
#             dec_out = self.classification(x_enc, x_mark_enc)
#             return dec_out  # [B, N]
#         return None
