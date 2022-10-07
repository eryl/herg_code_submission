from riseqsar.models.random_forest import RandomForestPredictor, RandomForestPredictorConfig
from riseqsar.models.model_specification import ModelSpecification
from riseqsar.experiment.hyperparameter_optimization import  HyperParameterCatergorical


from riseqsar.featurizer import FeaturizerConfig
from riseqsar.dataset.featurized_dataset import FeaturizedDatasetConfig


n_estimators = HyperParameterCatergorical(name='n_estimators', choices=[10, 100, 200, 300, 500])
criterion = HyperParameterCatergorical(name='criterion', choices=['gini', 'entropy'])
max_depth = HyperParameterCatergorical(name='max_depth', choices=[None, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
max_leaf_nodes = HyperParameterCatergorical(name='max_leaf_nodes', choices=[None, 8, 32, 64, 128, 256, 512, 1024])

#solver = HyperParameterCatergorical(name='sovler', choices=[])
model_config = RandomForestPredictorConfig(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=2,
                        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                        max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.0, bootstrap=True,
                        oob_score=False, n_jobs=-1, class_weight='balanced')

featurizer_config = FeaturizerConfig(method='mordred')

dataset_config = FeaturizedDatasetConfig(featurizer_config=featurizer_config)

model_specification = ModelSpecification(model_class=RandomForestPredictor, model_config=model_config, dataset_config=dataset_config)