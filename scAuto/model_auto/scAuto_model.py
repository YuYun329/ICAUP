import json

import pandas as pd
import torch
from torch import nn

from scAuto.model_auto.AutoCorrelation import AutoCorrelationLayer, AutoCorrelation
from scAuto.model_auto.Autoformer_EncDec import series_decomp, Encoder, EncoderLayer, my_Layernorm
from scAuto.model_auto.embedding import BertEmbeddings


class Prediction(nn.Module):
    def __init__(self, latent_dim=1024, n_cells=2034, d_model=768):
        super(Prediction, self).__init__()
        self.conv = nn.Conv1d(in_channels=512, out_channels=d_model // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm1d(num_features=d_model // 2)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.linear_bottle = nn.Linear(384 ** 2, latent_dim)
        self.drop = nn.Dropout(0.3)
        self.linear = nn.Linear(in_features=latent_dim,
                                out_features=n_cells)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_norm(x)
        x = self.pool(x)

        x = self.flatten(x)
        y = self.elu(self.linear_bottle(x))
        y = self.drop(y)
        y = self.linear(y)
        y = self.sigmoid(y)
        return y


class Prediction_bc(nn.Module):
    def __init__(self, latent_dim=1024, n_batch=6, n_cells=2034, d_model=768):
        super(Prediction_bc, self).__init__()
        self.conv = nn.Conv1d(in_channels=512, out_channels=d_model // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm1d(num_features=d_model // 2)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.linear_bottle = nn.Linear(384 ** 2, latent_dim)
        self.drop = nn.Dropout(0.3)
        self.linear = nn.Linear(in_features=latent_dim,
                                out_features=n_cells)
        self.linear_bc = nn.Linear(in_features=latent_dim, out_features=n_batch)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x, batch_info):
        x = self.conv(x)
        x = self.conv_norm(x)
        x = self.pool(x)

        x = self.flatten(x)
        y = self.elu(self.linear_bottle(x))
        y = self.drop(y)
        y_cell = self.linear(y)

        y_bc = self.linear_bc(y)
        y_bc = torch.matmul(y_bc, batch_info)
        y = self.sigmoid(y_cell + y_bc)
        return y


class AutoFormer(nn.Module):
    def __init__(self, config, n_cells):
        super(AutoFormer, self).__init__()
        self.factor = config.factor
        self.dropout = config.dropout
        self.output_attention = config.output_attention
        self.d_model = config.hidden_size
        self.d_ff = config.d_ff
        self.moving_avg = config.moving_avg
        self.activation = config.activation
        self.e_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads

        kernel_size = self.moving_avg
        self.decomp = series_decomp(kernel_size)
        self.enc_embedding = BertEmbeddings(c_in=4)
        self.prediction = Prediction(latent_dim=config.latent_dim, n_cells=n_cells)

        # Encoder
        self.encoder = Encoder(
            config,
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(mask_flag=False, factor=self.factor, scale=None,
                                        attention_dropout=self.dropout,
                                        output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            conv_layers=None,
            norm_layer=my_Layernorm(self.d_model)
        )

    def forward(self, input_ids):
        x_enc = input_ids
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        enc_out = self.prediction(enc_out)
        return attns, enc_out

    def get_latent(self):
        return self.prediction.linear.weight.detach().numpy()

    def init_with_lda(self, lda_latent, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.prediction.linear.weight = torch.nn.Parameter(
            data=torch.tensor(lda_latent, device=device, dtype=torch.float), requires_grad=True)


class AutoFormer_bc(nn.Module):
    def __init__(self, config, n_cells, l2_1=1e-8):
        super(AutoFormer_bc, self).__init__()
        self.factor = config.factor
        self.dropout = config.dropout
        self.output_attention = config.output_attention
        self.d_model = config.hidden_size
        self.d_ff = config.d_ff
        self.moving_avg = config.moving_avg
        self.activation = config.activation
        self.e_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads

        kernel_size = self.moving_avg
        self.decomp = series_decomp(kernel_size)
        self.enc_embedding = BertEmbeddings(c_in=4)
        self.prediction_bc = Prediction_bc(latent_dim=config.latent_dim, n_cells=n_cells, n_batch=config.n_batch)

        # Encoder
        self.encoder = Encoder(
            config,
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(mask_flag=False, factor=self.factor, scale=None,
                                        attention_dropout=self.dropout,
                                        output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            conv_layers=None,
            norm_layer=my_Layernorm(self.d_model)
        )
        self.l2_reg = l2_1
        self.prediction_bc.linear.weight_decay = l2_1

    def forward(self, input_ids, bc_info):
        x_enc = input_ids
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)

        bc_info = torch.tensor(bc_info.T.values, dtype=torch.float, device=input_ids.device)

        enc_out = self.prediction_bc(enc_out, bc_info)

        return attns, enc_out

    def get_latent(self):
        return self.prediction_bc.linear.weight.detach().numpy()

    def init_with_lda(self, lda_latent, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.prediction_bc.linear.weight = torch.nn.Parameter(
            data=torch.tensor(lda_latent, device=device, dtype=torch.float), requires_grad=True)

    def l2_regularization(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2) ** 2
        return 0.5 * self.l2_reg * l2_loss