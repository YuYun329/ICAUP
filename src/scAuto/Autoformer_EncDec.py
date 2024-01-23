import torch
import torch.nn as nn
import torch.nn.functional as F

from transfer_architechture.prefix_encoder import PrefixEncoder


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x, past_value=None):
        new_x, attn = self.attention(
            x, x, x,
            past_value=past_value
        )
        x = x + self.dropout(new_x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res = y
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, config, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.prefix_encoder = PrefixEncoder(config)

    def forward(self, x):
        attns = []
        past_key_value_tuple = self.prefix_encoder.get_prompt(batch_size=x.shape[0])

        if self.conv_layers is not None:
            i = 0
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, past_value=past_key_value_tuple[i])
                # x, attn = attn_layer(x, past_value=None)
                i += 1
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for i, attn_layer in enumerate(self.attn_layers):
                x, attn = attn_layer.forward(x, past_value=past_key_value_tuple[i])
                # x, attn = attn_layer.forward(x, past_value=None)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns