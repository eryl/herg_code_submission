from riseqsar.models.model_specification import ModelSpecification
from riseqsar.models.logistic_regression import LogisticRegressionPredictor, LogisticRegressionConfig
from riseqsar.experiment.hyperparameter_optimization import HyperParameterLogUniform, HyperParameterCatergorical

from riseqsar.featurizer import FeaturizerConfig
from riseqsar.dataset.featurized_dataset import FeaturizedDatasetConfig


hp_C = HyperParameterLogUniform(name='C', low=0.001, high=100.0)

#penalty = HyperParameterCatergorical(name='penalty', choices=['l1', 'l2', 'elasticnet', 'none'])
penalty = HyperParameterCatergorical(name='penalty', choices=['l2', 'none'])
#solver = HyperParameterCatergorical(name='sovler', choices=[])
model_config = LogisticRegressionConfig(penalty=penalty, C=hp_C, class_weight='balanced')

featurizer_config = FeaturizerConfig(method='mordred')
dataset_config = FeaturizedDatasetConfig(featurizer_config=featurizer_config)

model_specification = ModelSpecification(model_class=LogisticRegressionPredictor, 
                                         model_config=model_config, 
                                         dataset_config=dataset_config)