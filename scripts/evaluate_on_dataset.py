from argparse import ArgumentParser
from pathlib import Path
from typing import List
from collections import defaultdict
import pickle

import pandas as pd

from riseqsar.experiment.experiment_tracker import find_top_level_resamples, ExperimentTracker
from riseqsar.dataset.dataset_specification import load_dataset_specs, DatasetSpec
from riseqsar.dataset.constants import TEST

from riseqsar.models.descriptor_based_predictor import DescriptorbasedPredictor
from riseqsar.models.molecular_predictor import MolecularPredictor    
    
from riseqsar.evaluation.calculate_performance import calculate_performance

def make_dataset(dataset_spec: DatasetSpec, model: MolecularPredictor, experiment_dir: Path):
    """Create a dataset for the given model"""
    # For now this is a rathern nasty hack. The framework should be refactored so that 
    # all metadata needede to create a dataset is contained in the predictor.
    # Note that for each predictor class which has its own make_train_dev_test_split, 
    # there should be an if/elif clause below
    with open(experiment_dir / 'artifacts' / 'model_specification.pkl', 'rb') as fp:
        model_spec = pickle.load(fp)
        dataset_config = model_spec.dataset_config

    if isinstance(model, DescriptorbasedPredictor):
        # Need to handle the features
        dataset = model.dataset_class.from_dataset_spec(dataset_spec=dataset_spec, config=dataset_config, tag=TEST, featurizer=model.featurizer)
    elif isinstance(model, MolecularPredictor):
        dataset = model.dataset_class.from_dataset_spec(dataset_spec=dataset_spec, config=dataset_config, tag=TEST)
    else:
        raise NotImplementedError(f"make_dataset() has not been implemented for model typ {type(model)}")

    return dataset

        
def evaluate_models(dataset_spec, experiment_roots):
    for path_to_run in experiment_roots:
        top_level_resamples = find_top_level_resamples(path_to_run)

        for resample_dir in top_level_resamples:
            try:
                experiment_tracker = ExperimentTracker(resample_dir)
                model = experiment_tracker.load_model('final_model')
            except FileNotFoundError:
                continue
            dataset = make_dataset(dataset_spec, model, resample_dir)
            test_predictions = model.predict_dataset_proba(dataset)
            test_performance = calculate_performance(true_class=dataset.get_only_targets(),
                                                     prediction_scores=test_predictions,
                                                     experiment_tracker=experiment_tracker,
                                                     dataset_name=dataset.get_identifier(),
                                                     tag=dataset.get_tag(),
                                                     threshold=model.threshold)
            
def main():
    parser = ArgumentParser(description="Run the model evaluation on a dataset")
    parser.add_argument('dataset_spec', help="Dataset spec to use for evaluation", type=Path)
    parser.add_argument('experiment_roots', help="Root directories to search for experiments in", nargs='+', type=Path)
    parser.add_argument('--spec-identifier', help="If given, use the DatasetSpec with this idenfier, otherwise use the one where the intended use is TEST")
    args = parser.parse_args()

    dataset_spec_collection = load_dataset_specs(args.dataset_spec)
    if args.spec_identifier is not None:
        dataset_spec = dataset_spec_collection[args.spec_identifier]
    else:
        dataset_spec = dataset_spec_collection.by_intended_use(TEST)

    evaluate_models(dataset_spec, args.experiment_roots)

if __name__ == '__main__':
    main()
    
    
