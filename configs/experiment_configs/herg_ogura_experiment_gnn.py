from riseqsar.dataset.resampling import SubsamplingConfig
from riseqsar.dataset.constants import TRAIN, DEV, TEST
from riseqsar.experiment.hyperparameter_optimization import HyperParameterOptimizationConfig
from riseqsar.experiment.experiment_config import ExperimentSpecification, ExperimentSpecificationCollection
from riseqsar.evaluation.performance import metric_roc_auc



n_resamples = 20
random_seed_base = 1234
experiment_name = 'herg_ogura'
experiments = []

dataset_spec_path = 'dataset/herg_ogura_filtered/dataset_spec.py'
model_name = 'graph_neural_network'
model_common_kwargs = dict(dataset_spec_path=dataset_spec_path,
                           evaluation_metrics=[metric_roc_auc],
                           experiment_environment='rise-qsar-torch',
                           model_spec_path='configs/model_configs/herg_ogura/gnn_hpsearch.py',
                           )

for i in range(n_resamples):
    resample_random_seed = random_seed_base + i
    resample_config = SubsamplingConfig(subsampling_ratios={TRAIN: .8, DEV: .1, TEST: .1},
                                        n_subsamples=1,            
                                        random_seed=resample_random_seed,
                                        mol_sample_strategy='stratified')
    
    model_rng_seed = resample_random_seed
    dataset_rng_seed = resample_random_seed

    # Note that the resamples of the hyper parameter tuning will use the 
    # dev set from the upper resample loop as its test set, so doesn't need a TEST split
    # hp_tune_resample_config = SubsamplingConfig(subsampling_ratios={TRAIN: .9, DEV: .1},
    #                                             n_subsamples=1,            
    #                                             random_seed=resample_random_seed,
    #                                             mol_sample_strategy='stratified')
    hp_tune_resample_config = None

    hp_config = HyperParameterOptimizationConfig(hp_iterations=20,
                                                    hp_direction='maximize', 
                                                    hp_evaluation_metric=metric_roc_auc,
                                                    hp_resample_config=hp_tune_resample_config)

    experiment_specification = ExperimentSpecification(hp_config=hp_config, 
                                                        name=f"{model_name}_{i:02}",
                                                        resample_config=resample_config,
                                                        model_rng_seed=model_rng_seed,
                                                        dataset_rng_seed=dataset_rng_seed,
                                                        **model_common_kwargs)
    experiments.append(experiment_specification)


experiment_config = ExperimentSpecificationCollection(name=experiment_name, output_dir='experiments', 
                                                      experiments=experiments)
