import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np
from layers.TVI import TVI_Module


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)
        self.tvi = TVI_Module(seq_len=self.seq_len, enc_in=512, dropout=configs.dropout, activation="relu")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.tvi(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
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
# from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
# from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
# from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
# import math
# import numpy as np


# class Model(nn.Module):
#     """
#     Autoformer is the first method to achieve the series-wise connection,
#     with inherent O(LlogL) complexity
#     Paper link: https://openreview.net/pdf?id=I55UqU-M11y
#     """

#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.task_name = configs.task_name
#         self.seq_len = configs.seq_len
#         self.label_len = configs.label_len
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention

#         # Decomp
#         kernel_size = configs.moving_avg
#         self.decomp = series_decomp(kernel_size)

#         # Embedding
#         self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                   configs.dropout)
#         # Encoder
#         self.encoder1 = Encoder(
#             [
#                 EncoderLayer(
#                     AutoCorrelationLayer(
#                         AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                         output_attention=configs.output_attention),
#                         configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     moving_avg=configs.moving_avg,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 )
#             ],
#             norm_layer=my_Layernorm(configs.d_model)
#         )
#         self.encoder2 = Encoder(
#             [
#                 EncoderLayer(
#                     AutoCorrelationLayer(
#                         AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                         output_attention=configs.output_attention),
#                         configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     moving_avg=configs.moving_avg,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 )
#             ],
#             norm_layer=my_Layernorm(configs.d_model)
#         )
#         self.encoder3 = Encoder(
#             [
#                 EncoderLayer(
#                     AutoCorrelationLayer(
#                         AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                         output_attention=configs.output_attention),
#                         configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     moving_avg=configs.moving_avg,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 )
#             ],
#             norm_layer=my_Layernorm(configs.d_model)
#         )
#         # Decoder
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                       configs.dropout)
#             self.decoder = Decoder(
#                 [
#                     DecoderLayer(
#                         AutoCorrelationLayer(
#                             AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
#                                             output_attention=False),
#                             configs.d_model, configs.n_heads),
#                         AutoCorrelationLayer(
#                             AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
#                                             output_attention=False),
#                             configs.d_model, configs.n_heads),
#                         configs.d_model,
#                         configs.c_out,
#                         configs.d_ff,
#                         moving_avg=configs.moving_avg,
#                         dropout=configs.dropout,
#                         activation=configs.activation,
#                     )
#                     for l in range(configs.d_layers)
#                 ],
#                 norm_layer=my_Layernorm(configs.d_model),
#                 projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
#             )
#         if self.task_name == 'imputation':
#             self.projection = nn.Linear(
#                 configs.d_model, configs.c_out, bias=True)
#         if self.task_name == 'anomaly_detection':
#             self.projection = nn.Linear(
#                 configs.d_model, configs.c_out, bias=True)
#         if self.task_name == 'classification':
#             self.act = F.gelu
#             self.dropout = nn.Dropout(configs.dropout)
#             self.projection = nn.Linear(
#                 configs.d_model * configs.seq_len, configs.num_class)
        
#         n1, n2 = 128, 96
#         self.linear = nn.Linear(96, n1)
#         self.linear2 = nn.Linear(n1, 96)
#         self.sample_linear1 = nn.Linear(n1, n2)
#         self.upsample_linear1 = nn.Linear(n2, n1)

#         self.norm1 = nn.LayerNorm(n1)

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         # decomp init
#         mean = torch.mean(x_enc, dim=1).unsqueeze(
#             1).repeat(1, self.pred_len, 1)
#         zeros = torch.zeros([x_dec.shape[0], self.pred_len,
#                              x_dec.shape[2]], device=x_enc.device)
#         seasonal_init, trend_init = self.decomp(x_enc)
#         # decoder input
#         trend_init = torch.cat(
#             [trend_init[:, -self.label_len:, :], mean], dim=1)
#         seasonal_init = torch.cat(
#             [seasonal_init[:, -self.label_len:, :], zeros], dim=1)

#         # Embedding
#         x1 = self.enc_embedding(x_enc, x_mark_enc) # [bsz, seq_len, d_model]
#         x1 = self.linear(x1.permute(0,2,1)).permute(0,2,1)  # [bsz, n1, d_model]

#         x1_, _= self.encoder1(x1)                      # x: [bsz, nvar, n1]

#         # 下采样
#         x2 = self.sample_linear1(x1_.permute(0,2,1)).permute(0,2,1)                   # x: [bsz, nvar, n2]
#         x2_,_ = self.encoder2(x2)

#         src1_ = x1 + self.upsample_linear1(x2_.permute(0,2,1)).permute(0,2,1)        # x1_: [bsz, nvar, 512]
#         enc_out = self.norm1(src1_.permute(0,2,1)).permute(0,2,1)

#         # dec
#         dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
#         seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
#                                                  trend=trend_init)
#         # final
#         dec_out = trend_part + seasonal_part
#         return dec_out

#     def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
#         # enc
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)
#         # final
#         dec_out = self.projection(enc_out)
#         return dec_out

#     def anomaly_detection(self, x_enc):
#         # enc
#         enc_out = self.enc_embedding(x_enc, None)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)
#         # final
#         dec_out = self.projection(enc_out)
#         return dec_out

#     def classification(self, x_enc, x_mark_enc):
#         # enc
#         enc_out = self.enc_embedding(x_enc, None)
#         enc_out, attns = self.encoder(enc_out, attn_mask=None)

#         # Output
#         # the output transformer encoder/decoder embeddings don't include non-linearity
#         output = self.act(enc_out)
#         output = self.dropout(output)
#         # zero-out padding embeddings
#         output = output * x_mark_enc.unsqueeze(-1)
#         # (batch_size, seq_length * d_model)
#         output = output.reshape(output.shape[0], -1)
#         output = self.projection(output)  # (batch_size, num_classes)
#         return output

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
#         if self.task_name == 'imputation':
#             dec_out = self.imputation(
#                 x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
#             return dec_out  # [B, L, D]
#         if self.task_name == 'anomaly_detection':
#             dec_out = self.anomaly_detection(x_enc)
#             return dec_out  # [B, L, D]
#         if self.task_name == 'classification':
#             dec_out = self.classification(x_enc, x_mark_enc)
#             return dec_out  # [B, N]
#         return None
