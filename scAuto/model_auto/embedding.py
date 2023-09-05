import math

import torch
from torch import nn

BertLayerNorm = torch.nn.LayerNorm

class BasePairEmbedding(nn.Module):
    def __init__(self, c_in=1, d_model=768):
        super(BasePairEmbedding, self).__init__()
        self.base_pair_conv = nn.Conv1d(in_channels=c_in,
                                        out_channels=d_model,
                                        kernel_size=3,
                                        padding=1,
                                        padding_mode="zeros")
        self.same_channel_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                           kernel_size=3, padding=1, padding_mode='zeros')
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.adapt_pool = nn.AdaptiveMaxPool1d(512)

        self.base_pair_gelu = nn.GELU()

    def forward(self, x):
        x = self.base_pair_conv(x)
        x = self.pool(x)
        x = self.same_channel_conv(x)
        x = self.adapt_pool(x)
        x = self.base_pair_gelu(x)
        return x.permute(0, 2, 1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = torch.zeros(size=(seq_len, d_model),
                                                requires_grad=False).float()

        position = torch.arange(0, seq_len).unsqueeze(dim=1).float()
        div_value = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        self.positional_embedding[:, 0::2] = torch.sin(position * div_value)
        self.positional_embedding[:, 1::2] = torch.cos(position * div_value)

        self.positional_embedding = self.positional_embedding.unsqueeze(dim=0)

    def forward(self):
        positional_embedding = self.positional_embedding
        return positional_embedding


class StemLayer(nn.Module):
    def __init__(self, number_input_features, number_output_features, kernel_size, pool_degree):
        super(StemLayer, self).__init__()
        self.stem_conv = nn.Conv1d(in_channels=number_input_features,
                                   out_channels=number_output_features,
                                   kernel_size=kernel_size,
                                   padding=kernel_size // 2,
                                   padding_mode="zeros")
        self.stem_norm = nn.BatchNorm1d(num_features=number_output_features)
        self.stem_gelu = nn.GELU()
        self.stem_pool = nn.MaxPool1d(kernel_size=pool_degree)

    def forward(self, x):
        stem_conv = self.stem_conv(x)
        stem_norm = self.stem_norm(stem_conv)
        stem_gelu = self.stem_gelu(stem_norm)
        stem_pool = self.stem_pool(stem_gelu)

        return stem_pool


class StemEmbedding(nn.Module):
    def __init__(self, c_in=1, d_model=768):
        super(StemEmbedding, self).__init__()
        # 4 stem layer
        c_increase = (d_model - c_in) // 4

        self.stem_layer_1 = StemLayer(number_input_features=c_in,
                                      number_output_features=c_in + 1 * c_increase,
                                      kernel_size=5,
                                      pool_degree=2)
        self.stem_layer_2 = StemLayer(number_input_features=c_in + 1 * c_increase,
                                      number_output_features=c_in + 2 * c_increase,
                                      kernel_size=5,
                                      pool_degree=4)
        self.stem_layer_3 = StemLayer(number_input_features=c_in + 2 * c_increase,
                                      number_output_features=c_in + 3 * c_increase,
                                      kernel_size=5,
                                      pool_degree=4)
        self.stem_layer_4 = StemLayer(number_input_features=c_in + 3 * c_increase,
                                      number_output_features=c_in + 4 * c_increase,
                                      kernel_size=5,
                                      pool_degree=4)

    def forward(self, x):
        stem_layer_1 = self.stem_layer_1(x)
        stem_layer_2 = self.stem_layer_2(stem_layer_1)
        stem_layer_3 = self.stem_layer_3(stem_layer_2)
        stem_layer_4 = self.stem_layer_4(stem_layer_3)

        return stem_layer_4.permute(0, 2, 1)


class BertEmbeddings(nn.Module):
    def __init__(self, c_in=1, d_model=768, seq_len=512, seq_len_up_bound=512):
        super(BertEmbeddings, self).__init__()
        self.seq_len = seq_len
        self.seq_len_up_bound = seq_len_up_bound
        self.base_pair_embedding = BasePairEmbedding(c_in=c_in,
                                                         d_model=d_model)

        self.positional_embedding = PositionalEmbedding(d_model=d_model,
                                                        seq_len=seq_len)

    def forward(self, input_ids=None):
        # input_ids = torch.unsqueeze(input_ids, 1)
        input_ids = input_ids.permute(0, 2, 1)
        device = input_ids.device
        positional_embedding = self.positional_embedding()
        positional_embedding = positional_embedding.to(device)
        if self.seq_len > self.seq_len_up_bound:
            stem_embedding = self.stem_embedding(input_ids)
            embedding = stem_embedding + positional_embedding
        else:
            base_pair_embedding = self.base_pair_embedding(input_ids)
            base_pair_embedding = base_pair_embedding.to(device)
            embedding = base_pair_embedding + positional_embedding

        return embedding
