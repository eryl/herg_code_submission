from dataclasses import dataclass, field
from tkinter import E
from typing import Union, List, Optional, Sequence, Mapping
from collections import Counter
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from riseqsar.dataset.constants import TRAIN, DEV, TEST
from riseqsar.dataset.rdkit_dataset import RDKitMolDataset, shuffle_atoms, mol_to_smiles

from riseqsar.models.neural_networks.dnn import DNNConfig, DeepNeuralNetwork




class SequenceNeuralNetworkConfig(DNNConfig):
    def __init__(self, *args,
                 embedding_dim: int,
                 tokenizer_class: type,
                 tokenizer_args=None,
                 tokenizer_kwargs=None,
                 **kwargs):
        super(SequenceNeuralNetworkConfig, self).__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.tokenizer_class = tokenizer_class
        if tokenizer_args is None:
            tokenizer_args = tuple()
        self.tokenizer_args = tokenizer_args
        if tokenizer_kwargs is None:
            tokenizer_kwargs = dict()
        self.tokenizer_kwargs = tokenizer_kwargs


class SequenceDatasetCollator(object):
    def __init__(self, *, tokenizer, augment_smiles=True, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.tokenizer = tokenizer
        self.augment_smiles = augment_smiles

    def __call__(self, batch):
        rd_mols, *targets = zip(*batch)
        if self.augment_smiles:
            rd_mols = [shuffle_atoms(rd_mol, self.rng) for rd_mol in rd_mols]
        smiles_list = mol_to_smiles(rd_mols)
        tokenized_smiles, mask = self.tokenizer.tokenize_and_pad(smiles_list)
        #tokenized_smiles = [self.tokenizer.tokenize(smiles) for smiles in smiles_list]
        #tokenized_tensors = [torch.tensor(smiles, dtype=torch.long) for smiles in tokenized_smiles]
        #packed_tensors = torch.nn.utils.rnn.pack_sequence(tokenized_tensors, enforce_sorted=False)
        token_tensor = torch.tensor(tokenized_smiles, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        target_tensors = [torch.tensor(target, dtype=torch.float32) for target in targets]
        return token_tensor, mask_tensor, *target_tensors



class RDKitTorchDataset(RDKitMolDataset, Dataset):
    """Just a mix-class so the torch dataloader accepts our RDKitMolDataset"""
    pass


class SequenceNeuralNetwork(DeepNeuralNetwork):
    dataset_class = RDKitTorchDataset

    """Predictor wrapper for neural networks which takes the molecule structure as a sequential representation"""
    def __init__(self, *args, config: SequenceNeuralNetworkConfig, tokenizer=None, **kwargs):
        super(SequenceNeuralNetwork, self).__init__(*args, config=config, **kwargs)
        self.initialization_params['tokenizer'] = tokenizer


    def setup_dataloader(self, dataset: RDKitTorchDataset, is_training: bool):
        collate_fn = SequenceDatasetCollator(tokenizer=self.tokenizer, augment_smiles=is_training, rng=self.rng)

        if is_training:
            if self.config.weighted_sampler:
                samples_weight = dataset.get_samples_weights()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
                dataloader = DataLoader(dataset,
                                        batch_size=self.config.batch_size,
                                        #shuffle=is_training,
                                        sampler=sampler,
                                        num_workers=self.config.num_dl_workers,
                                        collate_fn=collate_fn,
                                        pin_memory=True,
                                        drop_last=False)

                return dataloader
        dataloader = DataLoader(dataset,
                                batch_size=self.config.batch_size,
                                shuffle=is_training,
                                num_workers=self.config.num_dl_workers,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                drop_last=False)

        return dataloader

    def setup_initialization_params(self, train_dataset):
        self.initialization_params['output_dim'] = len(train_dataset.get_targets())
        
        if 'tokenizer' not in self.initialization_params or self.initialization_params['tokenizer'] is None:
            self.initialization_params['tokenizer'] = self.config.tokenizer_class(*self.config.tokenizer_args, **self.config.tokenizer_kwargs)

        if not self.initialization_params['tokenizer'].is_fitted():
            self.initialization_params['tokenzier'].fit(train_dataset.get_smiles())

    def initialize_network(self):
        self.output_dim = self.initialization_params['output_dim']
        self.tokenizer = self.initialization_params['tokenizer']

        self.embedding = torch.nn.Embedding(self.tokenizer.get_num_embeddings(),
                                            embedding_dim=self.config.embedding_dim,
                                            padding_idx=self.tokenizer.padding_idx)

        if self.encoder is None:
            self.encoder = self.config.encoder_class(*self.config.encoder_args,
                                                     output_dim=self.config.hidden_dim,
                                                     input_dim=self.config.embedding_dim,
                                                     **self.config.encoder_kwargs)

        if self.decoder is None:
            self.decoder = self.config.decoder_class(*self.config.decoder_args,
                                                     input_dim=self.config.hidden_dim,
                                                     output_dim=self.output_dim,
                                                     **self.config.decoder_kwargs)
        self.model = torch.nn.Sequential(self.embedding, self.encoder, self.decoder)
        super().initialize_network()


    def predict_on_batch(self, batch):
        with torch.no_grad():
            x_batch, mask, *targets = batch

            if isinstance(x_batch, (tuple, list)):
                x_batch = [x.to(self.device) for x in x_batch]
            elif isinstance(x_batch, dict):
                x_batch = {k: v.to(self.device) for k, v in x_batch.items()}
            else:
                x_batch = x_batch.to(self.device)
            pred = self.model(x_batch)
            return pred

    def loss_on_batch(self, batch):
        x_batch, mask, *targets = batch
        y = torch.stack(targets, dim=1)

        if isinstance(x_batch, (tuple, list)):
            x_batch = [x.to(self.device) for x in x_batch]
        elif isinstance(x_batch, dict):
            x_batch = {k: v.to(self.device) for k, v in x_batch.items()}
        else:
            x_batch = x_batch.to(self.device)
        y = y.to(self.device)
        mask = mask.to(dtype=torch.float32, device=self.device)
        pred = self.model(x_batch)
        loss_mat = self.loss(pred, y)
        #loss_masked = loss_mat * mask
        #loss = loss_masked.sum()/mask.sum()
        loss = loss_mat.mean()
        return loss, y, pred

    def predict(self, smiles):
        raise NotImplementedError("Figure out how to make accurate predictions")
        self.model.eval()
        with torch.no_grad():
            tokenized = self.tokenizer.tokenize(smiles)
            tokenized_tensor = torch.tensor(tokenized, dtype=torch.long).unsqueeze(0)
            prediction = self.model(tokenized_tensor)
            probability = torch.sigmoid(prediction).cpu().numpy()
            return probability

    def predict_proba(self, smiles):
        self.model.eval()
        with torch.no_grad():
            tokenized = self.tokenizer.tokenize(smiles)
            tokenized_tensor = torch.tensor(tokenized, dtype=torch.long, device=self.device).unsqueeze(0)
            prediction = self.model(tokenized_tensor).sequeeze()
            probability = torch.sigmoid(prediction).cpu().numpy()
            return probability


    def predict_dataset_proba(self, dataset: RDKitTorchDataset):
        with torch.no_grad():
            dataloader = self.setup_dataloader(dataset, is_training=False)
            batch_predictions = [self.predict_on_batch(batch).detach().cpu() for batch in tqdm(dataloader, desc='predict on dataset_proba')]
            dataset_predictions = torch.cat(batch_predictions, dim=0)
            dataset_probas = torch.sigmoid(dataset_predictions)
            return dataset_probas.detach().numpy()

    def predict_dataset(self, dataset: RDKitMolDataset):
        prediction = [self.predict(smiles) for smiles in tqdm(dataset, desc='predict dataset')]
        return prediction