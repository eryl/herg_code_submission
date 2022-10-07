import pickle
from dataclasses import dataclass
import copy
from pathlib import Path

import pandas as pd
import numpy as np

from riseqsar.featurizer import FeaturizerConfig, make_featurizer
from riseqsar.dataset.dataset_specification import DatasetSpec
#from riseqsar.dataset.smiles_dataset import SmilesDataset
from riseqsar.dataset.rdkit_dataset import RDKitMolDataset, RDKitMolDatasetConfig


class FeaturizedDatasetConfig(RDKitMolDatasetConfig):
    def __init__(self, *args, featurizer_config: FeaturizerConfig, **kwargs):
        self.featurizer_config = featurizer_config


class FeaturizedDataset(RDKitMolDataset):
    def __init__(self, *, featurizer, features: pd.DataFrame, **kwargs):
        super(FeaturizedDataset, self).__init__(**kwargs)
        self.featurizer = featurizer
        if self.indices is not None:
            features = features.iloc[self.indices]
        self.features = features


    def make_copy(self, dataset_spec=None, identifier=None, tag=None):
        "Return a copy of this dataset"
        if dataset_spec is None:
            dataset_spec = copy.deepcopy(self.dataset_spec)

        dataset_class = type(self)
        dataset_copy = dataset_class(molecules = copy.deepcopy(self.molecules), 
                                     properties = copy.deepcopy(self.properties), 
                                     target_lists = copy.deepcopy(self.target_lists), 
                                     config = copy.deepcopy(self.config),
                                     featurizer = copy.deepcopy(self.featurizer),
                                     features = copy.deepcopy(self.features),
                                     dataset_spec = dataset_spec,
                                     identifier = identifier,
                                     tag=tag)
        return dataset_copy

    def get_n_features(self):
        return len(self.features.columns)

    def merge(self, other_dataset):
        '''Merge two datasets. This does not check for duplicate entries'''
        molecules = self.molecules + other_dataset.molecules
        properties = self.properties + other_dataset.properties
        target_lists = copy.deepcopy(self.target_lists)
        features = pd.concat([self.features, other_dataset.features])
        for target, target_values in other_dataset.get_targets():
            target_lists[target].extend(target_values)
        identifier = '_'.join(sorted({self.get_identifier(), other_dataset.get_identifier()}))
        tag = '_'.join(sorted({self.get_tag(), other_dataset.get_tag()}))

        # Since this dataset might have been derived from selected of the dataset, we have to fix that here
        dataset_spec = copy.deepcopy(self.dataset_spec)
        dataset_spec.indices = None

        return type(self)(molecules=molecules,
                          properties=properties,
                          target_lists=target_lists,
                          config=self.config,
                          dataset_spec=dataset_spec,
                          featurizer=self.featurizer,
                          features=features,
                          identifier=identifier,
                          tag=tag)

    def select(self, indices, tag=None):
        if tag is None:
            tag = self.tag
        dataset_spec = copy.deepcopy(self.dataset_spec)
        dataset_spec.indices = indices

        return type(self)(molecules=copy.deepcopy(self.molecules),
                          properties=copy.deepcopy(self.properties),
                          target_lists=copy.deepcopy(self.target_lists),
                          config=self.config,
                          dataset_spec=self.dataset_spec,
                          featurizer=self.featurizer,
                          features=copy.deepcopy(self.features),
                          identifier=self.identifier,
                          tag=tag)

    @classmethod
    def from_dataset_spec(cls, dataset_spec: DatasetSpec, *args, config: FeaturizedDatasetConfig=None, tag=None, featurizer=None, **kwargs):
        dataset_filename = Path(dataset_spec.file_name)
        base_name = dataset_filename.with_suffix('').name

        features_filename = dataset_filename.with_name(base_name + f'_featurized_{config.featurizer_config.method}.pkl')

        if not features_filename.exists():
            rdkit_dataset = RDKitMolDataset.from_dataset_spec(dataset_spec=dataset_spec)  # We don't cull indices at this step
            if featurizer is None:
                featurizer = make_featurizer(config.featurizer_config)

            if featurizer.is_fitted():
                features = featurizer.featurize(rdkit_dataset)
            else:
                features = featurizer.fit(rdkit_dataset)

            molecules = rdkit_dataset.molecules
            target_lists = rdkit_dataset.target_lists
            properties = rdkit_dataset.properties
            with open(features_filename, 'wb') as fp:
                pickle.dump(dict(features=features,
                                 featurizer=featurizer,
                                 molecules=molecules,
                                 target_lists=target_lists,
                                 properties=properties),
                            fp)
        else:
            with open(features_filename, 'rb') as fp:
                featurized_dataset = pickle.load(fp)
            featurizer = featurized_dataset['featurizer']
            features = featurized_dataset['features']
            molecules = featurized_dataset['molecules']
            target_lists = featurized_dataset['target_lists']
            properties = featurized_dataset['properties']

        return cls(*args,
                   featurizer=featurizer,
                   features=features,
                   molecules=molecules,
                   target_lists=target_lists,
                   dataset_spec=dataset_spec,
                   properties=properties,
                   config=config,
                   tag=tag,
                   **kwargs)
