from riseqsar.models.molecular_predictor import (MolecularPredictor, MolecularPredictorConfig,
                                                 MolecularDataset, MolecularDatasetConfig)
from riseqsar.dataset.dataset_specification import DatasetSpecs, DatasetSpec, MissingDatasetSpecError
from riseqsar.dataset.constants import TRAIN, DEV, TEST
from riseqsar.evaluation.calculate_performance import find_threshold
from riseqsar.evaluation.performance import EvaluationMetric
import numpy as np
from typing import List
from pathlib import Path
import pickle

class EnsembleWrapper(MolecularPredictor):
    dataset_class = MolecularDataset

    def __init__(self, *, training_config: TrainingConfig, config: MolecularPredictorConfig, rng=None):
        super().__init__(config=config, rng=rng)
        
        if rng is None:
            rng = np.random.default_rng()

        self.rng = rng
        self.random_state = self.rng.integers(0, 2**32-1)  # This will be a constant saved with the model which
                                                           # can be used to initialize any submodels precisely
        self.threshold = None
        self.config = config
        self.training_config = training_config


    def fit(self, *, train_dataset: MolecularDataset, evaluation_metrics: List[EvaluationMetric], dev_dataset=None, experiment_tracker=None):
        n_ensemble_members = self.training_config.n_ensemble_members

        for i in range(n_ensemble_members):
            model, final_performance = training_sequence(training_dataset=proper_train_set,
                                                         dev_dataset=proper_dev_set,
                                                         experiment_tracker=experiment_tracker,
                                                         test_dataset=proper_test_set,
                                                         training_config=self.training_config,
                                                         rng=self.rng)

        raise NotImplementedError(f'fit() has not been implemented for {self.__class__.__name__}')

    def fit_threshold(self, dataset: MolecularDataset):
        prediction_scores = self.predict_dataset_proba(dataset)

        target_values = dataset.get_only_targets()
        self.threshold = find_threshold(target_values, prediction_scores)
        return self.threshold

    def predict_dataset_proba(self, dataset: MolecularDataset):
        raise NotImplementedError(f'predict_dataset_proba() has not been implemented for {self.__class__.__name__}')

    def predict_dataset(self, dataset: MolecularDataset):
        raise NotImplementedError(f'predict_dataset() has not been implemented for {self.__class__.__name__}')

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
                                     dataset_specs: DatasetSpecs,
                                     dataset_config: MolecularDatasetConfig,
                                     model_config: MolecularPredictorConfig = None,
                                     rng=None):
        """By default the dataset is treated as a smiles dataset"""
        train_dataset_spec = dataset_specs.by_intended_use(TRAIN)
        train_dataset = cls.dataset_class.from_dataset_spec(dataset_spec=train_dataset_spec, config=dataset_config, tag=TRAIN,
                                                   rng=rng)
        try:
            dev_dataset_spec = dataset_specs.by_intended_use(DEV)
            dev_dataset = cls.dataset_class.from_dataset_spec(dataset_spec=dev_dataset_spec, config=dataset_config, tag=DEV, rng=rng)
        except MissingDatasetSpecError:
            print("Dataset specs has no dev set described")
            dev_dataset = None
        try:
            test_dataset_spec = dataset_specs.by_intended_use(TEST)
            test_dataset = cls.dataset_class.from_dataset_spec(dataset_spec=test_dataset_spec, config=dataset_config, tag=TEST, rng=rng)
        except MissingDatasetSpecError:
            print("Dataset specs has no test set described")
            test_dataset = None

        return train_dataset, dev_dataset, test_dataset