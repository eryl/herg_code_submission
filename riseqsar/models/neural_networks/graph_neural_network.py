from dataclasses import dataclass
from typing import List, Literal

import torch
from torch import nn

import torch
import numpy as np
#from torch_geometric.data import Data, DataLoader
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pickle
from sklearn.metrics import roc_auc_score
from torch.nn import Linear, LayerNorm, ReLU, Dropout, LeakyReLU
from torch_geometric.nn import NNConv, DeepGCNLayer
import torch.nn.functional as F
from torch.nn import Embedding
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Embedding
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler
import joblib

from riseqsar.dataset.dataset_specification import DatasetSpec
from riseqsar.models.neural_networks.dnn import DeepNeuralNetwork, DNNConfig
from riseqsar.dataset.graph_dataset import MolecularGraphDataset, MolecularGraphDatasetConfig, MolecularGraph


def GetAtomAndBondMatrix(mol):
    num_atom_featers = 6
    num_edge_features = 3

    B = torch.zeros(mol.GetNumAtoms(), mol.GetNumAtoms(), 3)
    A = torch.zeros(mol.GetNumAtoms(), num_atom_featers)

    for i in range(mol.GetNumAtoms()):
        a1 = mol.GetAtomWithIdx(i)
        A[i, 0] = a1.GetAtomicNum()
        A[i, 1] = a1.GetExplicitValence()
        A[i, 2] = a1.GetImplicitValence()
        A[i, 3] = a1.GetIsAromatic()
        A[i, 4] = a1.GetFormalCharge()
        A[i, 5] = a1.GetDegree()

        for j in range(mol.GetNumAtoms()):
            if (i == j):
                continue
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    B[i, j, 0] = 1
                elif bond.GetBondType() == Chem.BondType.DOUBLE:
                    B[i, j, 0] = 2
                elif bond.GetBondType() == Chem.BondType.AROMATIC:
                    B[i, j, 0] = 4
                elif bond.GetBondType() == Chem.BondType.TRIPLE:
                    B[i, j, 0] = 3
                else:
                    print("Unknown bond type")
                    print(bond.GetBondType())
                B[i, j, 1] = bond.IsInRing()
                B[i, j, 2] = bond.GetIsConjugated()

            # print(atom.GetAtomicNum())
            # print(atom.GetIdx())
    return A, B


class MapE2NxN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(MapE2NxN, self).__init__()
        self.linear1 = Linear(in_channels, hidden_channels)
        self.linear2 = Linear(hidden_channels, out_channels)
        self.relu = LeakyReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

@dataclass
class GNNEncoderConfig:
    num_layers: int
    d_model: int
    ffn_hidden_dim: int
    block: str = 'res+'
    dropout: float = 0.1


class GNNLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden_dim, block, dropout):
        super(GNNLayer, self).__init__()
        deep = MapE2NxN(d_model,
                        d_model * d_model,
                        ffn_hidden_dim)
        conv = NNConv(d_model, d_model, deep, aggr='mean')
        norm = LayerNorm(d_model, elementwise_affine=True)
        act = LeakyReLU(inplace=True)

        self.layer = DeepGCNLayer(conv, norm, act, block=block, dropout=dropout)

    def forward(self, x, edge_index, edge_attr):
        result = self.layer(x, edge_index, edge_attr)
        return result


class GNNEncoder(torch.nn.Module):
    def __init__(self, *, input_dim, output_dim, config: GNNEncoderConfig, ):
        super(GNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = input_dim
        self.config = config
        self.relu = torch.nn.LeakyReLU()
        self.gnn_layers = torch.nn.ModuleList()
        #self.input_layer = Linear(self.input_dim, self.d_model, bias=False)
        for i in range(config.num_layers):
            layer = GNNLayer(self.d_model, config.ffn_hidden_dim, config.block, config.dropout)
            self.gnn_layers.append(layer)
        self.dense_layer = Linear(self.d_model, self.d_model)
        self.output_layer = Linear(self.d_model, output_dim)
        self.dropout = Dropout(config.dropout)

    def forward(self, data):
        x = data.x
        #x = self.input_layer(x)  # Make sure x is the right dimensionality
        for layer in self.gnn_layers:
            x = layer(x, data.edge_index, data.edge_attr)
        x = self.dense_layer(x)
        x = global_mean_pool(x, data.batch)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def make_data_graph(graph: MolecularGraph, y=None, atom_features=None, bond_features=None):
    #x: (num_nodes, num_features)
    x = graph.get_atom_features(atom_features)
    edge_index = graph.get_coo_bonds()
    edge_attr = graph.get_bond_features(bond_features)
    x = torch.tensor(x, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    if y is not None:
        y = torch.tensor(y, dtype=torch.float32)
    ptg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return ptg_data

def make_data_graphs(graphs: List[MolecularGraph], targets, atom_features=None, bond_features=None):
    data_graphs = [make_data_graph(graph, y, atom_features, bond_features) for graph, y in tqdm(zip(graphs, targets), desc="converting graphs", total=len(graphs))]
    return data_graphs


class PTGGraphDataset(MolecularGraphDataset):

    @classmethod
    def from_dataset_spec(cls, *, dataset_spec: DatasetSpec, config: MolecularGraphDatasetConfig=None,
                          indices = None, tag=None, rng=None, **kwargs):
        if config is None:
            config = MolecularGraphDatasetConfig()
        
        file_name = dataset_spec.file_name
        base_name = file_name.with_suffix('').name
        
        dataset_dirname = file_name.parent / f'{base_name}_graphdataset_pytorch'
        dataset_fields_path = dataset_dirname / 'dataset_fields.pkl'
        pickled_graphs_dir = dataset_dirname / 'pickled_graphs'
        
        if dataset_dirname.exists():
            with open(dataset_fields_path, 'rb') as fp:
                dataset_fields = pickle.load(fp)
                molecules = dataset_fields['molecules']
                properties = dataset_fields['properties']
                target_lists = dataset_fields['target_lists']
                graphs_paths = dataset_fields['graphs_paths']
        else:
            graph_dataset = MolecularGraphDataset.from_dataset_spec(dataset_spec=dataset_spec)
            molecules = graph_dataset.molecules
            properties = graph_dataset.properties
            target_lists = graph_dataset.target_lists
            graphs_paths = graph_dataset.graphs_paths

        return cls(molecules=molecules,
                   properties=properties,
                   target_lists=target_lists,
                   config=config,
                   dataset_spec=dataset_spec,
                   graphs_paths=graphs_paths,
                   **kwargs)

    def __getitem__(self, item):
        if self.cache_graphs:
            datagraph = self.graphs[item]
            if datagraph is not None:
                return datagraph
        
        graph_path = self.graphs_paths[item]
        with open(graph_path, 'rb') as fp:
            graph = pickle.load(fp)
        
        y = self.get_only_targets()[item]
        datagraph = make_data_graph(graph, y, self.atom_feature_spec, self.bond_feature_spec) 

        if self.cache_graphs:
            self.graphs[item] = datagraph
        
        return datagraph


class GraphDeepNeuralNetworkConfig(DNNConfig):
    def __init__(self, *args, embedding_dim: int, embedding_aggregation: Literal['concatenate', 'sum'] = 'sum', **kwargs):
        super(GraphDeepNeuralNetworkConfig, self).__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.embedding_aggregation = embedding_aggregation


class GraphDeepNeuralNetwork(nn.Module):
    def __init__(self, node_embeddings, edge_embeddings, encoder, decoder, embedding_aggregation):
        super(GraphDeepNeuralNetwork, self).__init__()
        self.node_embeddings = nn.ModuleList(node_embeddings)
        self.edge_embeddings = nn.ModuleList(edge_embeddings)
        self.encoder = encoder
        self.decoder = decoder
        self.embedding_aggregation = embedding_aggregation

    def forward(self, data):
        if self.embedding_aggregation == 'sum':
            node_embeddings = torch.stack([emb(data.x[:,i]) for i, emb in enumerate(self.node_embeddings)], dim=1)
            node_embeddings_reduced = torch.sum(node_embeddings, dim=1)
            data.x = node_embeddings_reduced

            edge_embeddings = torch.stack([emb(data.edge_attr[:,i]) for i, emb in enumerate(self.edge_embeddings)], dim=1)
            edge_embeddings_reduced = torch.sum(edge_embeddings, dim=1)
            data.edge_attr = edge_embeddings_reduced
        elif self.embedding_aggregation == 'concatenate':
            raise NotImplementedError()

        encoding = self.encoder(data)
        result = self.decoder(encoding)
        return result


class GraphDeepNeuralNetworkPredictor(DeepNeuralNetwork):
    dataset_class = PTGGraphDataset
    def __init__(self, *args, config: GraphDeepNeuralNetworkConfig, **kwargs):
        super(GraphDeepNeuralNetworkPredictor, self).__init__(*args, config=config, **kwargs)
        self.config = config

    def setup_initialization_params(self, train_dataset: PTGGraphDataset):
        self.initialization_params['node_num_embeddings'] = [feature.n_values() for feature in train_dataset.get_node_feature_spec()]
        self.initialization_params['edge_num_embeddings'] = [feature.n_values() for feature in train_dataset.get_edge_feature_spec()]
        self.initialization_params['output_dim'] = len(train_dataset.get_targets())

    def initialize_network(self):
        self.output_dim = self.initialization_params['output_dim']

        self.node_embeddings = [Embedding(num_embeddings, self.config.embedding_dim)
                                for num_embeddings in self.initialization_params['node_num_embeddings']]
        self.edge_embeddings = [Embedding(num_embeddings, self.config.embedding_dim)
                                for num_embeddings in self.initialization_params['edge_num_embeddings']]

        if self.encoder is None:
            self.encoder = self.config.encoder_class(*self.config.encoder_args,
                                                     input_dim=self.config.embedding_dim,
                                                     output_dim=self.config.hidden_dim,
                                                     **self.config.encoder_kwargs)
        if self.decoder is None:
            self.decoder = self.config.decoder_class(*self.config.decoder_args,
                                                     input_dim=self.config.hidden_dim,
                                                     output_dim=self.output_dim,
                                                     **self.config.decoder_kwargs)

        self.model = GraphDeepNeuralNetwork(self.node_embeddings,
                                            self.edge_embeddings,
                                            self.encoder,
                                            self.decoder,
                                            self.config.embedding_aggregation)

        super().initialize_network()


    def setup_dataloader(self, *, dataset: PTGGraphDataset, is_training: bool):
        if is_training:
            if self.config.weighted_sampler:
                samples_weight = dataset.get_samples_weights()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

                dataloader = DataLoader(dataset,
                                        batch_size=self.config.batch_size,
                                        sampler=sampler,
                                        drop_last=False,
                                        num_workers=self.config.num_dl_workers,
                                        pin_memory=True)
                return dataloader

        dataloader = DataLoader(dataset,
                                batch_size=self.config.batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=self.config.num_dl_workers,
                                pin_memory=True)
        return dataloader

    #     dataloader = DataLoader(dataset,
    #                           batch_size=self.config.batch_size,
    #                           shuffle=is_training,
    #                           pin_memory=True,
    #                           num_workers=self.config.num_dl_workers)
    #     return dataloader

    def loss_on_batch(self, batch):
        batch = batch.to(self.device)
        pred = self.model(batch)
        loss_mat = self.loss(pred.squeeze(), batch.y)
        loss = loss_mat.mean()
        y = batch.y
        if len(y.shape) == 1:
            y = y.unsqueeze(dim=1)
        return loss, y, pred

    def predict_on_batch(self, batch):
        with torch.no_grad():
            self.model.eval()
            batch = batch.to(self.device)
            pred = self.model(batch)
            return pred

    def predict_proba(self, smiles: str):
        self.model.eval()
        with torch.no_grad():
            mol_graph = MolecularGraph.from_smiles(smiles)
            ptgraph = make_data_graph(mol_graph)
            ptgraph.to(self.device)
            pred_logistic = self.model(ptgraph)
            pred_prob = F.sigmoid(pred_logistic)
            return pred_prob.detach().cpu().numpy()

    
    def predict_dataset_proba(self, dataset: PTGGraphDataset):
        self.model.eval()
        with torch.no_grad():
            dataloader = self.setup_dataloader(dataset=dataset, is_training=False)
            batch_predictions = [self.predict_on_batch(batch).detach().cpu() for batch in dataloader]
            dataset_predictions = torch.cat(batch_predictions, dim=0)
            dataset_probas = torch.sigmoid(dataset_predictions)
            return dataset_probas.detach().cpu().numpy()

            
