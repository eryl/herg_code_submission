from dataclasses import dataclass
from typing import Type

from riseqsar.models.molecular_predictor import MolecularPredictor, MolecularPredictorConfig
from riseqsar.dataset.molecular_dataset import MolecularDatasetConfig

@dataclass
class ModelSpecification:
    model_class: Type[MolecularPredictor]
    model_config: MolecularPredictorConfig
    dataset_config: MolecularDatasetConfig