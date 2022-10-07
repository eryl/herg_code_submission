import argparse
from cmath import exp
import copy
import json
import sys
from collections import Collection
from dataclasses import dataclass, field
from pathlib import Path
import types
from typing import Sequence, Mapping, Type, Dict, Any, Optional, List, Union, Literal

import numpy as np
import optuna

from riseqsar.dataset.molecular_dataset import MolecularDataset
from riseqsar.dataset.constants import TRAIN, DEV, TEST

from riseqsar.dataset.resampling import ResamplingConfig
from riseqsar.evaluation.performance import EvaluationMetric, HigherIsBetterMetric
from riseqsar.experiment.training_sequence import training_sequence
from riseqsar.experiment.experiment_tracker import ExperimentTracker
from riseqsar.experiment.experiment_config import ExperimentConfig


@dataclass
class HyperParameterOptimizationConfig:
    hp_iterations: int
    hp_direction: str
    hp_evaluation_metric: EvaluationMetric
    hp_resample_config: Optional[ResamplingConfig] = None
    sampler: Literal['tpe', 'random'] = 'tpe'
    save_model: bool = False


class HyperParameterError(Exception):
    def __init__(self, obj, message):
        self.obj = obj
        self.message = message

    def __str__(self):
        return f"{self.message}: {self.obj}"


class HyperParameter(object):
    def __init__(self, *, name: str):
        self.name = name
        self.trial_values = dict()

    def get_value(self, trial_or_study: Union[optuna.Trial, optuna.Study]):
        if isinstance(trial_or_study, optuna.Trial):
            return self.get_trial_value(trial_or_study)
        elif isinstance(trial_or_study, optuna.Study):
            return self.get_best_value(trial_or_study)
        else:
            raise ValueError(f"Can't get value with context object {trial_or_study}")

    def get_trial_value(self, trial: optuna.Trial):
        # It seems like we would not have to do this, since a trial will return the same value for the same parameter,
        # on the other hand this makes the framework robust to other implementations. Using the datetime start should be
        # more robust trials belonging to different studies (as will the outer cross validation loop)
        trial_id = trial.datetime_start
        if trial_id not in self.trial_values:
            self.trial_values[trial_id] = self.suggest_value(trial)
        else:
            pass
        return self.trial_values[trial_id]

    def get_best_value(self, study: optuna.Study):
        return study.best_params[self.name]

    def suggest_value(self, trial: optuna.Trial):
        raise NotImplementedError("Can not suggest value for base class HyperParameter")


class HyperParameterCatergorical(HyperParameter):
    def __init__(self, *, choices: Sequence[Any], **kwargs):
        super(HyperParameterCatergorical, self).__init__(**kwargs)
        self.choices = choices

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_categorical(self.name, self.choices)


class HyperParameterDiscreteUniform(HyperParameter):
    def __init__(self, *, low:  float, high: float, q: int, **kwargs):
        super(HyperParameterDiscreteUniform, self).__init__(**kwargs)
        self.low = low
        self.high = high
        self.q = q

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_discrete_uniform(self.name, self.low, self.high, self.q)


class HyperParameterFloat(HyperParameter):
    def __init__(self, *, low:  float,
                 high: float,
                 step: Optional[float] = None,
                 log: Optional[bool] = False,
                 **kwargs):
        super(HyperParameterFloat, self).__init__(**kwargs)
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)


class HyperParameterInteger(HyperParameter):
    def __init__(self, *, low: int,
                 high: int,
                 step: Optional[int] = 1,
                 log: Optional[bool] = False,
                 **kwargs):
        super(HyperParameterInteger, self).__init__(**kwargs)
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_int(self.name, self.low, self.high, step=self.step, log=self.log)


class HyperParameterLogUniform(HyperParameter):
    def __init__(self, *, low: float, high: float, **kwargs):
        super(HyperParameterLogUniform, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_loguniform(self.name, self.low, self.high)


class HyperParameterUniform(HyperParameter):
    def __init__(self, *, low: float, high: float, **kwargs):
        super(HyperParameterUniform, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def suggest_value(self, trial: optuna.Trial):
        return trial.suggest_uniform(self.name, self.low, self.high)


class HyperParameterFunction(HyperParameter):
    def __init__(self, *, function, **kwargs):
        super(HyperParameterFunction, self).__init__(**kwargs)
        self.function = function

    def get_value(self, trial_or_study):
        return self.function(trial_or_study)


def instantiate_hp_value(obj, trial_or_study: Union[optuna.Trial, optuna.Study]):
    non_collection_types = (str, bytes, bytearray, np.ndarray)
    try:
        if isinstance(obj, (type, types.FunctionType, types.LambdaType, types.ModuleType)):
            return obj
        if isinstance(obj, HyperParameter):
            return obj.get_value(trial_or_study)
        elif isinstance(obj, Mapping):
            return type(obj)({k: instantiate_hp_value(v, trial_or_study) for k, v in obj.items()})
        elif isinstance(obj, Collection) and not isinstance(obj, non_collection_types):
            return type(obj)(instantiate_hp_value(x, trial_or_study) for x in obj)
        elif hasattr(obj, '__dict__'):
            try:
                obj_copy = copy.copy(obj)
                obj_copy.__dict__ = instantiate_hp_value(obj.__dict__, trial_or_study)
                return obj_copy
            except TypeError:
                return obj
        else:
            return obj
    except TypeError as e:
        raise HyperParameterError(obj, "Failed to materialize") from e


def hyper_parameter_search(training_dataset: MolecularDataset,
                           development_dataset: MolecularDataset,
                           experiment_config: ExperimentConfig,
                           experiment_tracker: ExperimentTracker, rng=None):
    if rng is None:
        rng = experiment_config.dataset_rng
    
    experiment_config_copy = copy.deepcopy(experiment_config)
    hp_config = experiment_config_copy.hp_config
    model_config = copy.deepcopy(experiment_config.model_specification.model_config)

    def objective(trial: optuna.Trial):
        #instantiated_model_args = tuple(instantiate_hp_value(v, trial) for v in model_args)
        #instantiated_model_kwargs = {k: instantiate_hp_value(v, trial) for k, v in model_kwargs.items()}
        instantiated_config = instantiate_hp_value(model_config, trial)
       
        experiment_config_copy.model_specification.model_config = instantiated_config
        start_event = experiment_tracker.log_event(f"Starting hp optimization trial {trial.number}")
        tracker = experiment_tracker.make_child(child_directory='hp_optimizations', tag=f'hp_trial_{trial.number}')
        model, performance = training_sequence(experiment_config=experiment_config_copy,
                                               training_dataset=training_dataset,
                                               dev_dataset=development_dataset,
                                               experiment_tracker=tracker,
                                               save_model=hp_config.save_model)
        dev_performance = performance[DEV]
        performance = dev_performance[hp_config.hp_evaluation_metric]
        experiment_tracker.log_event(f"Finished hp optimization trial {trial.number}", start_event)
        return performance

    def objective_nested_crossvalidation(trial: optuna.Trial):
        '''
        When evaluating each hyper parameter, do it on a nested cross validation and return the mean and variance of
        the results
        :param trial:
        :return:
        '''
        instantiated_config = instantiate_hp_value(model_config, trial)
        experiment_config_copy.model_specification.model_config = instantiated_config
        start_event = experiment_tracker.log_event(f"Starting hp optimization trial {trial.number}")
        
        tracker = experiment_tracker.make_child(child_directory='hp_optimizations', tag=f'trial_{trial.number}')
        resamples = training_dataset.make_resamples(hp_config.hp_resample_config, tag = 'hp_resample', rng=rng)
        performances = []
        # Here we hardcode that the HP resample config should contain exactly two resample ratios for the TRAIN and DEV intended use
        for i, resample in enumerate(resamples):
            print(f"HP resample {i}")
            
            resampled_train_dataset = resample[TRAIN]
            resampled_dev_dataset = resample[DEV]
            nested_tracker = tracker.make_child(child_directory='resamples', tag=f'resample_{i}')
            
            model, performance = training_sequence(experiment_config=experiment_config_copy,
                                                   training_dataset=resampled_train_dataset,
                                                   dev_dataset=resampled_dev_dataset,
                                                   experiment_tracker=nested_tracker,
                                                   test_dataset=development_dataset,
                                                   save_model=hp_config.save_model)
            dev_performance = performance[TEST]
            performance = dev_performance[hp_config.hp_evaluation_metric]
            performances.append(performance)

        performance_mean = float(np.mean(performances))
        std = float(np.std(performances))
        tracker.log_json('nested_resample_performance_summary', dict(performances=performances, mean=performance_mean, std=std))
        experiment_tracker.log_event(f"Finished hp optimization trial {trial.number}", start_event)
        return performance_mean#, std  # we stick to the mean for now

    # TODO: Make optimization target be dynamically set by configuration and not hard coded. This requires some
    #  refactoring of the training pipeline

    sampler_seed = rng.integers(0, 2**32-1)
    if hp_config.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    elif hp_config.sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
    else:
        raise NotImplementedError(f"Sample strategy {hp_config.sampler} has not been implemented")
    hp_direction = 'maximize' if isinstance(hp_config.hp_evaluation_metric, HigherIsBetterMetric) else 'minimize'
    if hp_config.hp_resample_config is not None:
        # Doing multiobjective optimization autoamtically is difficult. We're going to maximize the mean instead
        #study = optuna.create_study(directions=[hp_direction, 'minimize'])  ## Use direction from hp config and minimize variance
        study = optuna.create_study(direction=hp_direction, sampler=sampler)
        study.optimize(objective_nested_crossvalidation, n_trials=hp_config.hp_iterations)
        # finalized_args, finalized_kwargs = finalize_params(model_args, model_kwargs, study.best_params)
    else:
        study = optuna.create_study(direction=hp_direction, sampler=sampler)
        study.optimize(objective, n_trials=hp_config.hp_iterations)
    experiment_tracker.log_artifact('optuna_study', study)
    
    finalized_config = instantiate_hp_value(model_config, study)
    return finalized_config