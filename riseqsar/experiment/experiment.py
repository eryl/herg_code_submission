from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import copy

import numpy as np

from riseqsar.experiment.experiment_tracker import ExperimentTracker
from riseqsar.experiment.hyperparameter_optimization import HyperParameterOptimizationConfig, hyper_parameter_search
from riseqsar.util import load_config, timestamp
from riseqsar.evaluation.calculate_performance import calculate_performance
from riseqsar.experiment.training_sequence import training_sequence
from riseqsar.dataset.constants import TRAIN, DEV, TEST
from riseqsar.experiment.experiment_config import ExperimentConfig



def run_experiment(*,
                    experiment_config: ExperimentConfig,
                    output_dir: Path,
                    metadata=None,
                    files: Dict[str, Path]=None,
                    artifacts: Dict[str, Any]=None,
                    model_rng=None,
                    dataset_rng=None):

    if model_rng is None:
        model_rng = experiment_config.model_rng
    if dataset_rng is None:
        dataset_rng = experiment_config.model_rng

    model_specification = experiment_config.model_specification
    model_class = model_specification.model_class
    proper_train_set, proper_dev_set, proper_test_set = model_class.make_train_dev_test_datasets(experiment_config)
    dataset_identifier = proper_train_set.identifier

    experiment_tracker = ExperimentTracker(output_dir)
    experiment_tracker.log_json('metadata', metadata)
    experiment_tracker.log_files(files)
    experiment_tracker.log_artifacts(artifacts)
    start_event_datetime = experiment_tracker.log_event(f'Starting experiment {experiment_config.name}')

    if experiment_config.resample_config is not None:
        resamples = proper_train_set.make_resamples(experiment_config.resample_config, tag='resample', rng=dataset_rng)
        for resample_index, resample in enumerate(resamples):
            experiment_start_datetime = experiment_tracker.log_event(f'Starting run on resample {resample_index}')
            run_tracker = experiment_tracker.make_child(child_directory='resamples', tag = f'resample_{resample_index}')

            print(f"Training resample {resample_index}")

            training_dataset = resample[TRAIN]
            dev_dataset = resample[DEV]
            test_dataset = resample[TEST]
            
            training_dataset.set_tag(f'resample_{resample_index}_{training_dataset.get_tag()}')
            dev_dataset.set_tag(f'resample_{resample_index}_{dev_dataset.get_tag()}')
            test_dataset.set_tag(f'resample_{resample_index}_{test_dataset.get_tag()}')

            print(f'Training: {len(training_dataset)}')
            print(f'Dev: {len(dev_dataset)}')
            print(f'Test: {len(test_dataset)}')

            fold_experiment_config = copy.deepcopy(experiment_config)
            if experiment_config.hp_config is not None:
                model_config = hyper_parameter_search(training_dataset,
                                                      dev_dataset,
                                                      fold_experiment_config,
                                                      run_tracker)
                fold_experiment_config.model_specification.model_config = model_config

            model, final_performance = training_sequence(training_dataset=training_dataset,
                                                        dev_dataset=dev_dataset,
                                                        experiment_tracker=run_tracker,
                                                        test_dataset=test_dataset,
                                                        experiment_config=fold_experiment_config)
            if proper_test_set is not None:
                experiment_tracker.log_artifact('proper_test_dataset_spec', proper_test_set.dataset_spec)
                test_predictions = model.predict_dataset_proba(proper_test_set)
                test_performance = calculate_performance(true_class=proper_test_set.get_only_targets(),
                                                         prediction_scores=test_predictions.squeeze(),
                                                         experiment_tracker=run_tracker,
                                                         dataset_name=proper_test_set.get_identifier(),
                                                         tag=proper_test_set.get_tag(),
                                                         threshold=model.threshold)
                final_performance['PROPER_TEST_SET'] = test_performance
            experiment_tracker.log_event(f'Finished run on resample {resample_index}', experiment_start_datetime)
    else:
        if experiment_config.hp_config is not None:
            model_config = hyper_parameter_search(proper_train_set,
                                                  proper_dev_set,
                                                  experiment_config,
                                                  experiment_tracker)
            experiment_config.model_specification.model_config = model_config

        
        model, final_performance = training_sequence(training_dataset=proper_train_set,
                                                    dev_dataset=proper_dev_set,
                                                    experiment_tracker=experiment_tracker,
                                                    test_dataset=proper_test_set,
                                                    experiment_config=experiment_config)

    experiment_tracker.log_event(f'Finished experiment {experiment_config.name}', start_event_datetime)

