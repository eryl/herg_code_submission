from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from riseqsar.models.descriptor_based_predictor import DescriptorbasedPredictor
from riseqsar.models.neural_networks.dnn import DeepNeuralNetwork, DNNConfig
from riseqsar.dataset.featurized_dataset import FeaturizedDataset, FeaturizedDatasetConfig

@dataclass
class FeedForwardNetworkConfig:
    n_layers: int
    hidden_dim: int
    dropout_rate: float = 0
    normalization: bool = True
    residual_connections: bool = True
    activation_function: str = 'relu'


class FFBlock(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 *,
                 residual_connection,
                 dropout_rate,
                 normalization,
                 activation_function='relu'
                 ):
        super(FFBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.residual_connection = residual_connection
        self.normalization = normalization
        self.activation_function = activation_function
        if activation_function == 'relu':
            self.f = torch.nn.ReLU()
        if normalization:
            self.normalization = torch.nn.LayerNorm(output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        z = self.linear(x)
        if self.normalization:
            z = self.normalization(z)
        if self.residual_connection:
            z = x + z
        a = self.f(z)
        if self.dropout_rate > 0:
            a = F.dropout(a, self.dropout_rate)
        return a


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, *, input_dim, output_dim, config: FeedForwardNetworkConfig, random_state=None):
        super(FeedForwardNetwork, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Since the input layer typically has a different dimension that the hidden dim, we can not support residual
        # connections for it
        layers = []
        input_layer = FFBlock(input_dim, self.config.hidden_dim,
                              residual_connection=False,
                              dropout_rate=0,
                              normalization=False,
                              activation_function='relu')
        layers.append(input_layer)
        n_hidden_layers = self.config.n_layers - 2
        if n_hidden_layers > 0:
            hidden_layers = [FFBlock(self.config.hidden_dim, self.config.hidden_dim,
                                     residual_connection=self.config.residual_connections,
                                     dropout_rate=self.config.dropout_rate,
                                     normalization=self.config.normalization,
                                     activation_function='relu')
                             for i in range(n_hidden_layers)]
            layers.extend(hidden_layers)
        output_layer = torch.nn.Linear(self.config.hidden_dim, output_dim)
        layers.append(output_layer)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class DeepNeuralNetworkDescriptorbasedPredictorConfig(DNNConfig):
    pass


class DeepNeuralNetworkDescriptorbasedPredictor(DeepNeuralNetwork, DescriptorbasedPredictor):
    dataset_class = FeaturizedDataset

    def __init__(self, *args, config: DNNConfig, encoder=None, decoder=None, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.config = config
        self.device = torch.device(self.config.device)
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_random_state = self.rng.integers(0, 2 ** 32 - 1)
        self.decoder_random_state = self.rng.integers(0, 2 ** 32 - 1)
        torch.manual_seed(self.rng.integers(0, 2 ** 32 - 1))

    def setup_initialization_params(self, train_dataset: FeaturizedDataset):
        self.initialization_params['input_dim'] = train_dataset.get_n_features()
        self.initialization_params['output_dim'] = len(train_dataset.get_targets())
        self.initialization_params['featurizer'] = train_dataset.featurizer

    def initialize_network(self):
        self.input_dim = self.initialization_params['input_dim']
        self.output_dim = self.initialization_params['output_dim']

        if self.featurizer is None:
            self.featurizer = self.initialization_params['featurizer']

        if self.encoder is None:
            self.encoder = self.config.encoder_class(*self.config.encoder_args,
                                                     input_dim=self.input_dim,
                                                     output_dim=self.config.hidden_dim,
                                                     random_state=self.encoder_random_state,
                                                     **self.config.encoder_kwargs)
        if self.decoder is None:
            self.decoder = self.config.decoder_class(*self.config.decoder_args,
                                                     input_dim=self.config.hidden_dim,
                                                     output_dim=self.output_dim,
                                                     random_state=self.decoder_random_state,
                                                     **self.config.decoder_kwargs)

        self.model = torch.nn.Sequential(self.encoder, self.decoder)
        self.model.to(self.device)

        params = []
        if self.config.train_encoder:
            params.extend(self.encoder.parameters())
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False

        params.extend(self.decoder.parameters())
        self.params = params

    def setup_dataloader(self, *, dataset: FeaturizedDataset, is_training: bool):
        tensor_features = torch.tensor(dataset.features.values, dtype=torch.float32)
        tensor_targets = {k: torch.tensor(v, dtype=torch.float32) for k, v in sorted(dataset.target_lists.items())}
        torch_dataset = TensorDataset(tensor_features, *tensor_targets.values())

        if is_training:
            if self.config.weighted_sampler:
                samples_weight = dataset.get_samples_weights()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

                dataloader = DataLoader(torch_dataset,
                                        batch_size=self.config.batch_size,
                                        sampler=sampler,
                                        drop_last=False,
                                        num_workers=self.config.num_dl_workers,
                                        pin_memory=True, )
                return dataloader

        dataloader = DataLoader(torch_dataset,
                                batch_size=self.config.batch_size,
                                shuffle=is_training,
                                drop_last=False,
                                num_workers=self.config.num_dl_workers,
                                pin_memory=True)

        return dataloader

    def loss_on_batch(self, batch):
        x_batch, *targets = batch
        y = torch.stack(targets, dim=1)

        if isinstance(x_batch, (tuple, list)):
            x_batch = [x.to(self.device) for x in x_batch]
        elif isinstance(x_batch, dict):
            x_batch = {k: v.to(self.device) for k, v in x_batch.items()}
        else:
            x_batch = x_batch.to(self.device)
        y = y.to(self.device)
        pred = self.model(x_batch)
        loss_mat = self.loss(pred, y)
        loss = loss_mat.mean()
        return loss, y, pred

    def predict(self, smiles: str):
        raise NotImplementedError()

    def predict_on_batch(self, batch):
        x_batch, *targets = batch
        y = torch.stack(targets, dim=1)

        if isinstance(x_batch, (tuple, list)):
            x_batch = [x.to(self.device) for x in x_batch]
        elif isinstance(x_batch, dict):
            x_batch = {k: v.to(self.device) for k, v in x_batch.items()}
        else:
            x_batch = x_batch.to(self.device)
        pred = self.model(x_batch)
        return pred

    def predict_featurized(self, featurized_mols):
        self.model.eval()
        with torch.no_grad():
            mol_tensor = torch.tensor(featurized_mols, dtype=torch.float32, requires_grad=False, device=self.device)
            prediction = self.model(mol_tensor).squeeze()
            return prediction

    def predict_proba_featurized(self, featurized_mols):
        self.model.eval()
        with torch.no_grad():
            mol_tensor = torch.tensor(featurized_mols, dtype=torch.float32, requires_grad=False, device=self.device)
            prediction = self.model(mol_tensor).squeeze()
            probability = torch.sigmoid(prediction).cpu().numpy()
            return probability