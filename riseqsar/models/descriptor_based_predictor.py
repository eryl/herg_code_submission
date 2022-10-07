from cmath import exp
import pickle
from pathlib import Path
from typing import List

import numpy as np

from riseqsar.dataset.molecular_dataset import MolecularDataset, MolecularDatasetConfig

from riseqsar.featurizer import make_featurizer, FeaturizerConfig
from riseqsar.dataset.featurized_dataset import FeaturizedDatasetConfig, FeaturizedDataset

from riseqsar.dataset.dataset_specification import DatasetSpecCollection, DatasetSpec, MissingDatasetSpecError
from riseqsar.dataset.constants import TRAIN, DEV, TEST
from riseqsar.evaluation.calculate_performance import find_threshold
from riseqsar.evaluation.performance import EvaluationMetric
from riseqsar.models.molecular_predictor import MolecularPredictor

class DescriptorbasedPredictor(MolecularPredictor):
    dataset_class = FeaturizedDataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.featurizer = None
        self.model = None

    def fit(self, *, train_dataset: FeaturizedDataset, evaluation_metrics: List[EvaluationMetric], dev_dataset=None, experiment_tracker=None):
        if self.featurizer is None:
            self.featurizer = train_dataset.featurizer

        featurized_mols = train_dataset.features.values
        target_values = train_dataset.get_only_targets()
        self.model.fit(featurized_mols, target_values)

    def predict_dataset_proba(self, dataset: FeaturizedDataset):
        featurized_mols = dataset.features.values
        return self.predict_proba_featurized(featurized_mols)

    def predict_dataset(self, dataset: FeaturizedDataset):
        featurized_mols = dataset.features.values
        prediction = self.predict_featurized(featurized_mols)
        return prediction

    def predict_proba_featurized(self, featurized_mols):
        pred_prediction = self.model.predict_proba(featurized_mols)
        return pred_prediction

    def predict_featurized(self, featurized_mols):
        prediction = self.model.predict(featurized_mols)
        return prediction

    def predict_proba(self, smiles):
        featurized_mols = self.featurizer.featurize([smiles]).values
        return self.predict_proba_featurized(featurized_mols)[0]

    def predict(self, smiles):
        featurized_mols = self.featurizer.featurize([smiles]).values
        prediction = self.predict_featurized(featurized_mols)
        return prediction[0]

    @classmethod
    def make_train_dev_test_datasets(cls,
                                     experiment_config: 'ExperimentConfig'):
        """By default the dataset is treated as a smiles dataset"""
        dataset_specs_collection = experiment_config.dataset_spec_collection
        dataset_config = experiment_config.model_specification.dataset_config
        train_dataset_spec = dataset_specs_collection.by_intended_use(TRAIN)
        train_dataset = cls.dataset_class.from_dataset_spec(dataset_spec=train_dataset_spec, config=dataset_config, tag=TRAIN)
        featurizer = train_dataset.featurizer
        
        try:
            dev_dataset_spec = dataset_specs_collection.by_intended_use(DEV)
            dev_dataset = cls.dataset_class.from_dataset_spec(dataset_spec=dev_dataset_spec, config=dataset_config, tag=DEV,
                                                     featurizer=featurizer)
        
        except MissingDatasetSpecError:
            print("Dataset specs has no dev set described")
            dev_dataset = None
        try:
            test_dataset_spec = dataset_specs_collection.by_intended_use(TEST)
            test_dataset = cls.dataset_class.from_dataset_spec(dataset_spec=test_dataset_spec, config=dataset_config, tag=TEST,
                                                      featurizer=featurizer)
        except MissingDatasetSpecError:
            print("Dataset specs has no test set described")
            test_dataset = None

        return train_dataset, dev_dataset, test_dataset
