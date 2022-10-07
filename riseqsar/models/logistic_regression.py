import pickle
import warnings
from dataclasses import dataclass, asdict
from typing import Literal, Union, Dict, Optional

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

from riseqsar.models.molecular_predictor import MolecularPredictorConfig
from riseqsar.dataset.featurized_dataset import FeaturizedDataset

from riseqsar.models.descriptor_based_predictor import DescriptorbasedPredictor


@dataclass
class LogisticRegressionConfig(MolecularPredictorConfig):
    penalty: Literal['l1', 'l2', 'elasticnet', 'none'] = 'l2'
    C: float = 1.
    class_weight: Optional[Union[Dict, Literal['balanced']]] = None
    solver: Literal['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] = 'lbfgs'
    dual: bool = False
    tol: float = 0.0001
    fit_intercept: bool = True
    intercept_scaling: float = 1
    max_iter: int = 100
    multi_class: str ='auto'
    verbose: int = 0
    warm_start: bool = False
    n_jobs = None
    l1_ratio = None


class LogisticRegressionPredictor(DescriptorbasedPredictor):
    def __init__(self, *args, config: LogisticRegressionConfig, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.config = config
        params = asdict(config)
        self.model = LogisticRegression(random_state=self.random_state, **params)

    def fit(self, *, train_dataset: FeaturizedDataset, dev_dataset=None,
            experiment_tracker=None, evaluation_metrics=None):
        if self.featurizer is None:
            self.featurizer = train_dataset.featurizer

        featurized_mols = train_dataset.features
        featurized_mols_ndarray = featurized_mols.values

        if len(train_dataset.get_targets()) != 1:
            raise ValueError("SVMClassifier can not be used for multiple targets")

        target_name, target_values = train_dataset.get_targets()[0]
        self.model.fit(featurized_mols_ndarray, target_values)

    def predict_proba_featurized(self, featurized_mols):
        pred_prediction = self.model.predict_proba(featurized_mols)
        # The predictions are for both classes ordered by the label. TODO: Make sure this is actually according to the
        # 0-1 labels
        return pred_prediction[:,1]  # Return the probability for the positive class

    def predict_featurized(self, featurized_mols):
        prediction = self.model.predict(featurized_mols)
        return prediction

    def serialize(self, working_dir, tag=None):
        """Returns a factory function for recreating this model as well as the state required to do so"""
        model_bytes = pickle.dumps(self)
        model_factory = pickle.loads
        return model_factory, model_bytes

