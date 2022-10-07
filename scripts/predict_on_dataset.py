from argparse import ArgumentParser
from pathlib import Path
from typing import List
from collections import defaultdict
import pickle

import pandas as pd
from tqdm import tqdm
from riseqsar.dataset.smi_dataset import SMIDataset
import numpy as np

from riseqsar.experiment.experiment_tracker import find_top_level_resamples, ExperimentTracker
from riseqsar.dataset.dataset_specification import load_dataset_specs, DatasetSpec
from riseqsar.dataset.constants import TEST

from riseqsar.models.neural_networks.graph_neural_network import GraphDeepNeuralNetworkPredictor
from riseqsar.models.descriptor_based_predictor import DescriptorbasedPredictor
from riseqsar.models.molecular_predictor import MolecularPredictor    
    
from riseqsar.evaluation.calculate_performance import calculate_performance

def make_dataset(dataset_spec: DatasetSpec, model: MolecularPredictor, experiment_dir: Path):
    """Create a dataset for the given model"""
    # For now this is a rathern nasty hack. The framework should be refactored so that 
    # all metadata needed to create a dataset is contained in the predictor.
    # Note that for each predictor class which has its own make_train_dev_test_split, 
    # there should be an if/elif clause below
    with open(experiment_dir / 'artifacts' / 'model_specification.pkl', 'rb') as fp:
        model_spec = pickle.load(fp)
        dataset_config = model_spec.dataset_config

    if isinstance(model, DescriptorbasedPredictor):
        # Need to handle the features        
        #dataset = make_dataset(dataset_spec, model, resample_dir) 
        dataset = model.dataset_class.from_dataset_spec(dataset_spec=dataset_spec, config=dataset_config, tag=TEST, featurizer=model.featurizer)
    elif isinstance(model, MolecularPredictor):
        dataset = model.dataset_class.from_dataset_spec(dataset_spec=dataset_spec, config=dataset_config, tag=TEST)
    else:
        raise NotImplementedError(f"make_dataset() has not been implemented for model typ {type(model)}")

    return dataset

        
def evaluate_models(dataset_specs: List[DatasetSpec], experiment_roots, output_dir: Path):
    output_datasets = [SMIDataset.from_dataset_spec(dataset_spec=dataset_spec) for dataset_spec in dataset_specs]
    model_i = 0
    model_details = []
    for path_to_run in tqdm(experiment_roots, desc="Experiment"):
        top_level_resamples = find_top_level_resamples(path_to_run)

        for resample_dir in tqdm(top_level_resamples, desc='Resample'):
            try:
                experiment_tracker = ExperimentTracker(resample_dir)
                model = experiment_tracker.load_model('final_model')

                experiment_dir = resample_dir.parent.parent.parent
                experiment_id = experiment_dir.name
                resample_id = resample_dir.name
                timestamp = resample_dir.parent.parent.name
                
                model_details.append(dict(index=model_i, experiment_id=experiment_id, resample_id=resample_id, timestamp=timestamp))
            except FileNotFoundError:
                continue
            for dataset_spec, output_dataset in tqdm(list(zip(dataset_specs, output_datasets)), desc='Dataset'):
                dataset = make_dataset(dataset_spec, model, resample_dir)
                test_predictions = model.predict_dataset(dataset)
                test_prob_predictions = model.predict_dataset_proba(dataset)
                n_predictions_per_example = test_predictions.shape[1]
                for i in range(n_predictions_per_example):
                    output_dataset.add_properties(f'binary_prediction_target_{i}_model_{model_i}', test_predictions[:, i])
                    output_dataset.add_properties(f'probability_prediction_target_{i}_model_{model_i}', test_prob_predictions[:, i])
            model_i += 1
    for output_dataset in output_datasets:
        properties = output_dataset.properties
        class_ratios = []
        for prop in properties:
            class_predictions = [v for k,v in prop.items() if 'binary_prediction' in k]
            class_ratio = np.mean(class_predictions)
            class_ratios.append(class_ratio)
        output_dataset.add_properties('class_ratio', class_ratios)
        output_path = output_dir / f'predictions_{output_dataset.identifier}.csv'
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        output_dataset.properties_to_csv(output_path)
    
    model_details_df = pd.DataFrame.from_records(model_details)
    model_details_df.to_csv(output_dir / 'models_details.csv', index=False)
    

            
def main():
    parser = ArgumentParser(description="Run the model evaluation on a dataset")
    parser.add_argument('dataset_spec', help="Dataset spec to use for evaluation", type=Path)
    parser.add_argument('experiment_roots', help="Root directories to search for experiments in", nargs='+', type=Path)
    parser.add_argument('--spec-identifier', help="If given, use the DatasetSpec with this idenfier, otherwise use the one where the intended use is TEST")
    parser.add_argument('--output-dir', help="Where to save the prediction outputs", type=Path, default=Path())
    args = parser.parse_args()

    dataset_spec_collection = load_dataset_specs(args.dataset_spec)
    if args.spec_identifier is not None:
        dataset_specs = dataset_spec_collection[args.spec_identifier]
    else:
        dataset_specs = dataset_spec_collection.by_intended_use_all(TEST)

    evaluate_models(dataset_specs, args.experiment_roots, args.output_dir)


if __name__ == '__main__':
    main()
    
    
