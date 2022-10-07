import copy
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Any, Literal

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier

from riseqsar.models.molecular_predictor import MolecularPredictorConfig
from riseqsar.models.descriptor_based_predictor import DescriptorbasedPredictor
from riseqsar.dataset.featurized_dataset import FeaturizedDataset

@dataclass
class RandomForestPredictorConfig(MolecularPredictorConfig):
    n_estimators: int = 100
    criterion: Literal["gini", "entropy", "log_loss"] = 'gini'
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Any = 'auto'
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: str = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    n_jobs: int = 1
    class_weight: Optional[Literal["balanced", "balanced_subsample"]] = 'balanced'  


class RandomForestPredictor(DescriptorbasedPredictor):
    def __init__(self, *args, config: RandomForestPredictorConfig, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.config = config
        self.model = RandomForestClassifier(random_state=self.random_state, **asdict(self.config))

    def fit(self, *, train_dataset: FeaturizedDataset, dev_dataset=None,
            experiment_tracker=None, evaluation_metrics=None):
        if self.featurizer is None:
            self.featurizer = train_dataset.featurizer

        featurized_mols = train_dataset.features
        featurized_mols_ndarray = featurized_mols.values

        if len(train_dataset.get_targets()) != 1:
            raise ValueError("RFClassifier can not be used for multiple targets")

        target_name, target_values = train_dataset.get_targets()[0]
        self.model.fit(featurized_mols_ndarray, target_values)

    def predict_proba_featurized(self, featurized_mols):
        pred_prediction = self.model.predict_proba(featurized_mols)
        # The predictions are for both classes ordered by the label. TODO: Make sure this is actually according to the
        #  0-1 labels
        return pred_prediction[:,1]  # Return the probability for the positive class

    def predict_featurized(self, featurized_mols):
        prediction = self.model.predict(featurized_mols)
        return prediction

    def serialize(self, working_dir, tag=None):
        """Returns a factory function for recreating this model as well as the state required to do so"""
        model_bytes = pickle.dumps(self)
        model_factory = pickle.loads
        return model_factory, model_bytes
