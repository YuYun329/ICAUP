import torch
from torch import nn

from scAuto.AutoCorrelation import AutoCorrelationLayer, AutoCorrelation
from scAuto.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm
from scAuto.embedding import BertEmbeddings


# class Prediction(nn.Module):
#     def __init__(self, latent_dim=32, n_cells=2034, d_models=768):
#         super(Prediction, self).__init__()
#         self.conv_1 = nn.Conv1d(in_channels=d_models, out_channels=363, kernel_size=3, padding=1)
#         self.pool_1 = nn.MaxPool1d(kernel_size=3)
#         self.activate = nn.GELU()
#         self.bn_1 = nn.BatchNorm1d(363)
#
#         self.conv_2 = nn.Conv1d(in_channels=363, out_channels=256, kernel_size=3)
#         self.bn_2 = nn.BatchNorm1d(256)
#         self.pool_2 = nn.MaxPool1d(kernel_size=3)
#
#         self.conv_3 = nn.Conv1d(in_channels=256, out_channels=32, kernel_size=8)
#         self.bn_3 = nn.BatchNorm1d(32)
#         self.pool_3 = nn.MaxPool1d(kernel_size=7)
#
#         self.conv_4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
#         self.bn_4 = nn.BatchNorm1d(32)
#         self.pool_4 = nn.MaxPool1d(kernel_size=7)
#         self.flatten = nn.Flatten(start_dim=1)
#
#         self.linear = nn.Linear(in_features=32, out_features=n_cells)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.activate(x)
#         x = self.pool_1(self.bn_1(self.conv_1(x)))
#         x = self.activate(x)
#         x = self.pool_2(self.bn_2(self.conv_2(x)))
#         x = self.activate(x)
#         x = self.pool_3(self.bn_3(self.conv_3(x)))
#         x = self.activate(x)
#         x = self.pool_4(self.bn_4(self.conv_4(x)))
#         x = self.flatten(x)
#         return self.sigmoid(self.linear(x))


class Prediction(nn.Module):
    def __init__(self, latent_dim=32, n_cells=2034, d_models=768):
        super(Prediction, self).__init__()
        self.conv = nn.Conv1d(in_channels=d_models,
                              out_channels=d_models // 2,
                              kernel_size=3)
        self.conv_norm = nn.BatchNorm1d(num_features=d_models // 2)
        self.conv_elu = nn.ELU()
        self.pooling = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten(start_dim=1)
        self.linear_first = nn.Linear(in_features=384 * 255,
                                      out_features=latent_dim)
        self.linear_norm = nn.LayerNorm(normalized_shape=32)
        self.dropout = nn.Dropout(p=0.2)
        self.linear_elu = nn.ELU()
        self.linear = nn.Linear(latent_dim, n_cells)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv = self.conv(x)
        conv = self.conv_norm(conv)
        conv = self.conv_elu(conv)
        pooling = self.pooling(conv)
        f = self.flatten(pooling)
        peak_embedding = self.linear_first(f)
        peak_embedding = self.linear_norm(peak_embedding)
        peak_embedding = self.dropout(peak_embedding)
        peak_embedding = self.linear_elu(peak_embedding)
        peak_embedding = self.sigmoid(self.linear(peak_embedding))
        return peak_embedding


class AutoFormerEncoder(nn.Module):
    def __init__(self, config):
        super(AutoFormerEncoder, self).__init__()
        # Encoder
        self.encoder = Encoder(
            config,
            [
                EncoderLayer(
                    AutoCorrelationLayer(config,
                                         AutoCorrelation(mask_flag=False, factor=config.factor, scale=None,
                                                         attention_dropout=config.dropout,
                                                         output_attention=config.output_attention),
                                         config.hidden_size, config.num_attention_heads),
                    config.hidden_size,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for _ in range(config.num_hidden_layers)
            ],
            conv_layers=None,
            norm_layer=my_Layernorm(config.hidden_size)
        )
        self.embeddings = BertEmbeddings()

    def forward(self, input_ids):
        embedding = self.embeddings.forward(input_ids=input_ids)
        return self.encoder(embedding)


class AutoFormerFT(nn.Module):
    def __init__(self, config, n_cells):
        super(AutoFormerFT, self).__init__()
        self.factor = config.factor
        self.dropout = config.dropout
        self.output_attention = config.output_attention
        self.d_model = config.hidden_size
        self.d_ff = config.d_ff
        self.activation = config.activation
        self.e_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads

        self.auto = AutoFormerEncoder(config)
        self.prediction = Prediction(latent_dim=config.latent_dim, n_cells=n_cells)

    def get_latent(self):
        return self.prediction.linear.weight.detach().numpy()

    def init_with_lda(self, lda_latent, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.prediction.linear.weight = torch.nn.Parameter(
            data=torch.tensor(lda_latent, device=device, dtype=torch.float), requires_grad=True)

    def forward(self, input_ids):
        att_out, att = self.auto(input_ids)
        return att, self.prediction(att_out)
