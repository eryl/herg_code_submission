from riseqsar.models.model_specification import ModelSpecification
from riseqsar.models.xgboost import XGBoostParams, XGBoostConfig, XGBoostPredictor
from riseqsar.experiment.hyperparameter_optimization import  HyperParameterLogUniform, HyperParameterInteger

from riseqsar.featurizer import FeaturizerConfig
from riseqsar.dataset.featurized_dataset import FeaturizedDatasetConfig

learning_rate = HyperParameterLogUniform(name='learning_rate', low=1e-5, high=0.1)
max_depth = HyperParameterInteger(name='max_depth', low=2, high=20)

xgboost_params = XGBoostParams(learning_rate=learning_rate, max_depth=max_depth, gpu_id=0, tree_method='gpu_hist')
model_config = XGBoostConfig(params=xgboost_params, num_round=500, early_stopping_rounds=10)

featurizer_config = FeaturizerConfig(method='mordred')

dataset_config = FeaturizedDatasetConfig(featurizer_config=featurizer_config)
model_specification = ModelSpecification(model_class=XGBoostPredictor, model_config=model_config, dataset_config=dataset_config)