from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn import MultiheadAttention, Dropout, Linear, Module, LayerNorm, ModuleList

from riseqsar.models.neural_networks.sequence_neural_network import SequenceNeuralNetwork, SequenceNeuralNetworkConfig

@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int = 8
    dim_feedforward: int = 2048
    num_layers: int = 8
    dropout_rate: float = 0.1
    activation: str = 'relu'


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # pe : L x D
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe : L x 1 x D
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

ACTIVATIONS = {
    'relu': F.relu,
    'gelu': F.gelu
}


def _get_activation_fn(activation):
    try:
        return ACTIVATIONS[activation]
    except KeyError:
        raise KeyError("Activation should be one of {}, not {}.".format(list(ACTIVATIONS), activation))


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        a1, att_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src + self.norm1(self.dropout1(a1))
        a2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        a_out = src2 + self.norm2(self.dropout2(a2))
        return a_out


class TransformerEncoder(torch.nn.Module):
    def __init__(self,
                 *,
                 input_dim: int,
                 output_dim: int,
                 config: TransformerConfig):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        self.d_model = config.d_model
        self.scale = (self.d_model ** 0.5)
        self.num_heads = config.num_heads
        self.out_dim = self.d_model

        self.activation = config.activation

        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout_rate)
        self.layers = ModuleList()

        self.input_layer = None
        if self.input_dim != self.config.d_model:
            self.input_layer = Linear(input_dim, config.d_model, bias=False)

        for i in range(self.config.num_layers):
            block = TransformerEncoderBlock(config.d_model, config.num_heads,
                                            config.dim_feedforward, config.dropout_rate, config.activation)
            self.layers.append(block)

        self.output_layer = None
        if self.output_dim != self.config.d_model:
            self.output_layer = Linear(config.d_model, output_dim, bias=False)

        self._reset_parameters()


    def forward(self, embeddings, mask=None):
        # Retrieve embeddings, scale them, permute and add absolute positional encoding.
        scaled_embeddings = embeddings * self.scale  # self.embedding return dimensions (bs, L, d_model)
        if mask is not None:
            mask = torch.logical_not(mask)
        if self.input_layer is not None:
            # If the dimensionality of the embeddings does not the model we need to project them to the model dimension
            scaled_embeddings = self.input_layer(scaled_embeddings)

        transposed_embeddings = scaled_embeddings.transpose(0, 1)  # transpose: (bs, L, d_model) -> (L, bs, d_model)
        positioned_embeddings = self.pos_encoder(transposed_embeddings)  # add positional encoding
        state_from_below = positioned_embeddings


        for l in self.layers:
            state_from_below = l(state_from_below, src_key_padding_mask=mask)

        if self.output_layer is not None:
            state_from_below = self.output_layer(state_from_below)

        output = state_from_below.mean(dim=0)  # (bs, d_model)
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)



class SmilesTransformerPredictorConfig(SequenceNeuralNetworkConfig):
    pass

class SmilesTransformerPredictor(SequenceNeuralNetwork):
    pass