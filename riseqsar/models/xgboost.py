import copy
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, fields, asdict
from tempfile import NamedTemporaryFile
from typing import Literal, Union, Dict, Optional

warnings.filterwarnings('ignore')

import numpy as np

import xgboost as xgb

from riseqsar.models.molecular_predictor import MolecularPredictorConfig
from riseqsar.dataset.featurized_dataset import FeaturizedDataset
from riseqsar.models.descriptor_based_predictor import DescriptorbasedPredictor

# The parameters which goes into XGBoost's param field in the train has its own dataclass for simplicity
@dataclass
class XGBoostParams:
    learning_rate: float
    min_split_loss: float = 0
    max_depth: int = 6
    min_child_weight: float = 1.
    max_delta_step: float = 0
    subsample: float = 1
    sampling_method: Literal['uniform', 'gradient_based'] = 'uniform'
    colsample_bytree: float = 1
    colsample_bylevel: float = 1
    colsample_bynode: float = 1
    reg_lambda: float = 1
    reg_alpha: float = 0
    tree_method: Literal['auto', 'exact', 'approx', 'hist', 'gpu_hist'] = 'auto'
    booster: Literal['gbtree', 'gblinear', 'dart'] = 'gbtree'
    sketch_eps: float = 0.03
    scale_pos_weight: float = 1
    objective: str = 'binary:logistic'
    eval_metric: str = 'auc'
    gpu_id: int = 0


@dataclass
class XGBoostConfig(MolecularPredictorConfig):
    params: XGBoostParams
    num_round: int
    weighted_samples: bool = False
    early_stopping_rounds: Optional[int] = None


class XGBoostPredictor(DescriptorbasedPredictor):
    def __init__(self, *args, config: XGBoostConfig, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.config = config

    def fit(self, *, train_dataset: FeaturizedDataset, dev_dataset=None,
            experiment_tracker=None, evaluation_metrics=None):
        if self.featurizer is None:
            self.featurizer = train_dataset.featurizer

        featurized_mols = train_dataset.features
        featurized_mols_ndarray = featurized_mols.values
        target_values = np.array(train_dataset.get_only_targets())
        if self.config.weighted_samples:
            samples_weight = train_dataset.get_samples_weights()
            dtrain = xgb.DMatrix(featurized_mols_ndarray, label=target_values, weight=samples_weight)
        else:
            dtrain = xgb.DMatrix(featurized_mols_ndarray, label=target_values)


        dev_featurized_mols = dev_dataset.features
        dev_featurized_mols_ndarray = dev_featurized_mols.values
        dev_target_values = np.array(dev_dataset.get_only_targets())
        ddev = xgb.DMatrix(dev_featurized_mols_ndarray, label=dev_target_values)

        evals = [(ddev, 'dev')]
        param = asdict(self.config.params)
        seed = self.random_state
        param['seed'] = seed
        self.model = xgb.train(param, dtrain,
                                  self.config.num_round,
                                  evals=evals,
                                  early_stopping_rounds=self.config.early_stopping_rounds)

    def predict_proba_featurized(self, featurized_mols):
        data = np.ascontiguousarray(featurized_mols)
        if hasattr(self.model, 'best_iteration'):
            prediction = self.model.inplace_predict(data, iteration_range=(0, self.model.best_iteration))
        else:
            prediction = self.model.inplace_predict(data)
        return prediction

    def predict_featurized(self, featurized_mols):
        if not hasattr(self, 'threshold'):
            raise RuntimeError("Can't make class predictions without threshold, fit it first")
        predictions = self.predict_proba_featurized(featurized_mols)
        return (predictions >= self.threshold).astype(int)

    def serialize(self, working_dir: Path, tag=None):
        """Returns a factory function for recreating this model as well as the state required to do so"""
        tmp_path = working_dir / 'tmp_xgboost.bin'
        self.model.save_model(tmp_path)
        with open(tmp_path, 'rb') as fp:
            model_bytes = fp.read()
        tmp_path.unlink()
        self_copy = copy.deepcopy(self)
        self_copy.model = model_bytes
        self_bytes = pickle.dumps(self_copy)
        return deserializes, self_bytes


def deserializes(predictor_bytes):
    predictor = pickle.loads(predictor_bytes)
    model_bytes = predictor.model
    with NamedTemporaryFile('wb') as fp:
        fp.write(model_bytes)
        predictor.model = xgb.Booster()
        predictor.model.load_model(fp.name)
    return predictor
