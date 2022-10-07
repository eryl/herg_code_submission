from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any, TYPE_CHECKING

import numpy as np

from riseqsar.util import load_config, timestamp

if TYPE_CHECKING:
    from riseqsar.dataset.dataset_specification import DatasetSpecCollection
    from riseqsar.dataset.resampling import ResamplingConfig
    from riseqsar.models.model_specification import ModelSpecification
    from riseqsar.evaluation.performance import EvaluationMetric
    from riseqsar.experiment.hyperparameter_optimization import HyperParameterOptimizationConfig



@dataclass
class ExperimentSpecification:
    """Dataclass for specifying experiments in python files"""
    name: str
    experiment_environment: str  # Identifier for the conda/docker environment to run the experiments in 
    model_spec_path: Union[Path, str]
    dataset_spec_path: Union[Path, str]
    evaluation_metrics: List[ "EvaluationMetric" ]
    hp_config: Optional['HyperParameterOptimizationConfig'] = None
    resample_config: Optional['ResamplingConfig'] = None
    model_rng_seed: Optional[int] = None
    dataset_rng_seed: Optional[int] = None
    num_workers: Optional[int] = None


@dataclass
class ExperimentSpecificationCollection:
    name: str
    experiments: List['ExperimentSpecification']
    output_dir: Union[Path, str]


@dataclass
class ExperimentConfig:
    """Configuration class used to pass information around during experiments. 
    Is constructed from the information given by the ExperimentSpecification"""
    name: str
    evaluation_metrics: List['EvaluationMetric']
    model_specification: 'ModelSpecification'
    dataset_spec_collection: 'DatasetSpecCollection'
    hp_config: Optional['HyperParameterOptimizationConfig'] = None
    resample_config: Optional['ResamplingConfig'] = None
    model_rng: np.random.Generator = None
    dataset_rng: np.random.Generator = None
    num_worksers: Optional[int] = None

def make_experiment_config(experiment_specification: 'ExperimentSpecification') -> 'ExperimentConfig':
    from riseqsar.models.model_specification import ModelSpecification
    from riseqsar.dataset.dataset_specification import load_dataset_specs
    from riseqsar.experiment.experiment_config import ExperimentConfig

    model_rng = np.random.default_rng(experiment_specification.model_rng_seed)
    dataset_rng = np.random.default_rng(experiment_specification.dataset_rng_seed)
    
    model_specification = load_config(experiment_specification.model_spec_path, ModelSpecification)
    dataset_spec_collection = load_dataset_specs(experiment_specification.dataset_spec_path)

    experiment_config = ExperimentConfig(name=experiment_specification.name,
                                         evaluation_metrics=experiment_specification.evaluation_metrics,
                                         model_specification=model_specification,
                                         dataset_spec_collection=dataset_spec_collection,
                                         hp_config=experiment_specification.hp_config,
                                         resample_config=experiment_specification.resample_config,
                                         model_rng=model_rng,
                                         dataset_rng=dataset_rng
                                         )
    return experiment_config