from riseqsar.models.model_specification import ModelSpecification
from riseqsar.models.svm import SVMClassifier, SVMClassifierConfig
from riseqsar.experiment.hyperparameter_optimization import HyperParameterUniform, HyperParameterLogUniform

from riseqsar.featurizer import FeaturizerConfig
from riseqsar.dataset.featurized_dataset import FeaturizedDatasetConfig



hp_C = HyperParameterUniform(name='C', low=0.1, high=100.0)
hp_gamma = HyperParameterLogUniform(name='gamma', low=1e-6, high=0.1)

model_config = SVMClassifierConfig(gpu_id=0,
                                                   C=hp_C,
                                                   gamma=hp_gamma)
featurizer_config = FeaturizerConfig(method='mordred')

dataset_config = FeaturizedDatasetConfig(featurizer_config=featurizer_config)

model_specification = ModelSpecification(model_class=SVMClassifier, model_config=model_config, dataset_config=dataset_config)
