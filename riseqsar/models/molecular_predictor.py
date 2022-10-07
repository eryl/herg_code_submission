from cmath import exp
import pickle
from pathlib import Path
from typing import List

import numpy as np

from riseqsar.dataset.molecular_dataset import MolecularDataset


from riseqsar.dataset.dataset_specification import DatasetSpec, MissingDatasetSpecError
from riseqsar.dataset.constants import TRAIN, DEV, TEST
from riseqsar.evaluation.calculate_performance import find_threshold
from riseqsar.evaluation.performance import EvaluationMetric


class MolecularPredictorConfig(object):
    def __init__(self):
        pass


class MolecularPredictor(object):
    dataset_class = MolecularDataset

    def __init__(self, *, config: MolecularPredictorConfig, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.random_state = self.rng.integers(0, 2**32-1)  # This will be a constant saved with the model which
                                                           # can be used to initialize any submodels precisely
        self.threshold = None
        self.config = config

    def fit(self, *, train_dataset: MolecularDataset, evaluation_metrics: List[EvaluationMetric], dev_dataset=None, experiment_tracker=None):
        raise NotImplementedError(f'fit() has not been implemented for {self.__class__.__name__}')

    def fit_threshold(self, dataset: MolecularDataset):
        prediction_scores = self.predict_dataset_proba(dataset)

        target_values = dataset.get_only_targets()
        self.threshold = find_threshold(target_values, prediction_scores)
        return self.threshold

    def predict_dataset_proba(self, dataset: MolecularDataset):
        raise NotImplementedError(f'predict_dataset_proba() has not been implemented for {self.__class__.__name__}')

    def predict_dataset(self, dataset: MolecularDataset):
        probabilities = self.predict_dataset_proba(dataset)
        binarized_predictions = (probabilities >= self.threshold).astype(int)
        return binarized_predictions

    def predict_proba(self, smiles: str):
        raise NotImplementedError(f'predict_proba() has not been implemented for {self.__class__.__name__}')
    
    def predict(self, smiles: str):
        raise NotImplementedError(f'predict() has not been implemented for {self.__class__.__name__}')

    def save(self, output_dir: Path, tag=None):
        if tag is None:
            model_name = f'{self.__class__.__name__}.pkl'
        else:
            model_name = f'{self.__class__.__name__}_{tag}.pkl'
        with open(output_dir / model_name, 'wb') as fp:
            pickle.dump(self, fp)

    def set_device(self, device):
        print(f"set_device has not been implmented for {self.__class__.__name__}")

    @classmethod
    def load(cls, load_dir, tag=None):
        if tag is None:
            model_name = f'{cls.__name__}.pkl'
        else:
            model_name = f'{cls.__name__}_{tag}.pkl'
        with open(load_dir / model_name, 'rb') as fp:
            model = pickle.load(fp)
        return model

    def serialize(self, working_dir, tag=None):
        """Returns a factory function for recreating this model as well as the state required to do so"""
        model_bytes = pickle.dumps(self)
        model_factory = pickle.loads
        return model_factory, model_bytes

    @classmethod
    def make_train_dev_test_datasets(cls,
                                     experiment_config: 'ExperimentConfig'):
        """By default the dataset is treated as a smiles dataset"""

        dataset_specs_collection = experiment_config.dataset_spec_collection
        train_dataset_spec = dataset_specs_collection.by_intended_use(TRAIN)
        dataset_config = experiment_config.model_specification.dataset_config
        train_dataset = cls.dataset_class.from_dataset_spec(dataset_spec=train_dataset_spec, config=dataset_config, tag=TRAIN)
        try:
            dev_dataset_spec = dataset_specs_collection.by_intended_use(DEV)
            dev_dataset = cls.dataset_class.from_dataset_spec(dataset_spec=dev_dataset_spec, config=dataset_config, tag=DEV)
        except MissingDatasetSpecError:
            print("Dataset specs has no dev set described")
            dev_dataset = None
        try:
            test_dataset_spec = dataset_specs_collection.by_intended_use(TEST)
            test_dataset = cls.dataset_class.from_dataset_spec(dataset_spec=test_dataset_spec, config=dataset_config, tag=TEST)
        except MissingDatasetSpecError:
            print("Dataset specs has no test set described")
            test_dataset = None

        return train_dataset, dev_dataset, test_dataset

    
