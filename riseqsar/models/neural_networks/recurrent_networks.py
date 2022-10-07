from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F

from riseqsar.models.neural_networks.sequence_neural_network import SequenceNeuralNetwork, SequenceNeuralNetworkConfig

@dataclass
class RecurrentNetworkConfig:
    n_layers: int
    hidden_dim: int
    dropout_rate: float = 0
    #normalization: bool = True
    bidirectional: bool = False
    #residual_connections: bool = True
    network_type: Literal['srnn', 'lstm', 'gru'] = 'gru'


class RecurrentNetwork(torch.nn.Module):
    def __init__(self, *, input_dim, output_dim, config: RecurrentNetworkConfig):
        super(RecurrentNetwork, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim

        if self.config.network_type == 'srnn':
           self.rnn_layer = torch.nn.RNN(input_size=self.input_dim,
                                      hidden_size=self.config.hidden_dim,
                                      num_layers=self.config.n_layers,
                                      dropout=self.config.dropout_rate,
                                      bidirectional=self.config.bidirectional,
                                     batch_first=True)
        elif self.config.network_type == 'lstm':
            self.rnn_layer = torch.nn.LSTM(input_size=self.input_dim,
                                      hidden_size=self.config.hidden_dim,
                                      num_layers=self.config.n_layers,
                                      dropout=self.config.dropout_rate,
                                      bidirectional=self.config.bidirectional,
                                     batch_first=True)
        elif self.config.network_type == 'gru':
            self.rnn_layer = torch.nn.GRU(input_size=self.input_dim,
                                      hidden_size=self.config.hidden_dim,
                                      num_layers=self.config.n_layers,
                                      dropout=self.config.dropout_rate,
                                      bidirectional=self.config.bidirectional,
                                     batch_first=True)
        else:
            raise NotImplementedError(f"RNN type {self.config.network_type} not implemented")
        self.output_layer = None
        if output_dim != self.config.hidden_dim:
            # If the dimensionality is different, we add a projection layer
            self.output_layer = torch.nn.Linear(self.config.hidden_dim, output_dim, bias=False)


    def forward(self, x):
        self.rnn_layer.flatten_parameters()

        if isinstance(self.rnn_layer, torch.nn.LSTM):
            output, c_n, h_n = self.rnn_layer(x)
        else:
            output, h_n = self.rnn_layer(x)
        # Output is NxLxD∗H_out. We should reduce the last axis if this is bidirectional
        if self.config.bidirectional:
            # TODO: Implement bidirectional states (we need the indices of the first element of the sequence to return
            #  the correct final state in the backwards direction
            batch_size, seq_len = x.shape[:2]
            output = output.view(batch_size, 2, self.config.hidden_dim)
            final_state_forwards = output[:, -1, 0]  # The final state
            final_state_backwards = output[:, 0, 1]  # The final state. TODO: This should not just be the first element,
                                                     #  since the sequences are right-aligned we need to know the first
                                                     # position of each element
            raise NotImplementedError("Bidirectional RNN has not been implemented correctly")
        else:
            final_state_forwards = output[:, -1]
            h = final_state_forwards
        if self.output_layer is not None:
            h = self.output_layer(h)
        return h


class RecurrentNetworkPredictorConfig(SequenceNeuralNetworkConfig):
    pass

class RecurrentNetworkPredictor(SequenceNeuralNetwork):
    pass

