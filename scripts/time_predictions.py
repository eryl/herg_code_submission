from argparse import ArgumentParser
from pathlib import Path
from typing import List
from collections import defaultdict
import pickle
import time

import pandas as pd
from tqdm import tqdm
from riseqsar.dataset.smi_dataset import SMIDataset
import numpy as np

from riseqsar.experiment.experiment_tracker import find_top_level_resamples, ExperimentTracker
from riseqsar.dataset.dataset_specification import load_dataset_specs, DatasetSpec
from riseqsar.dataset.constants import TEST
    
        
def time_models(dataset_specs: List[DatasetSpec], experiment_roots, n_samples, output_dir: Path, rng):
    
    
    dataset_sampled_smiles = defaultdict(list)
    for dataset_spec in dataset_specs:    
        dataset = SMIDataset.from_dataset_spec(dataset_spec=dataset_spec)
        smiles_list = dataset.molecules
        identifier = dataset.identifier
        if len(smiles_list) < n_samples:
            dataset_sampled_smiles[identifier].extend(smiles_list)
        else:
            dataset_sampled_smiles[identifier] = rng.choice(smiles_list, n_samples, replace=False)

    dataset_measurements = defaultdict(list)
    for path_to_run in tqdm(experiment_roots, desc='Experiment roots'):
        top_level_resamples = find_top_level_resamples(path_to_run)

        for resample_dir in tqdm(top_level_resamples, desc='Resamples'):
            try:
                experiment_tracker = ExperimentTracker(resample_dir)
                model = experiment_tracker.load_model('final_model', force_cpu=False)

                experiment_dir = resample_dir.parent.parent.parent
            except FileNotFoundError:
                continue
            for dataset_identifier, sampled_smiles in dataset_sampled_smiles.items():
                experiment_id = experiment_dir.name
                resample_id = resample_dir.name
                timestamp = resample_dir.parent.parent.name
                
                times = []
                predictions = []
                for smiles in tqdm(sampled_smiles, desc="running predictions", leave=False):
                    t0 = time.time()
                    prediction = model.predict_proba(smiles)
                    t1 = time.time()
                    dt = t1-t0
                    times.append(dt)
                    # I honestly don't know if this is needed
                    predictions.append(prediction)
                measurement = dict(experiment_id=experiment_id, 
                                resample_id=resample_id, 
                                timestamp=timestamp, 
                                times_mean=np.mean(times), 
                                times_std=np.std(times), 
                                times_median=np.median(times),
                                n_samples=len(times))
                dataset_measurements[dataset_identifier].append(measurement)
    

    output_dir.mkdir(parents=True, exist_ok=True)
    for identifier, measurements in dataset_measurements.items():
        df = pd.DataFrame.from_records(measurements)
        output_path = output_dir / f'timings_{identifier}.csv'
        
        df.to_csv(output_path, index=False)
    
            
def main():
    parser = ArgumentParser(description="Run the model evaluation on a dataset")
    parser.add_argument('dataset_spec', help="Dataset spec to use for evaluation", type=Path)
    parser.add_argument('experiment_roots', help="Root directories to search for experiments in", nargs='+', type=Path)
    parser.add_argument('--spec-identifier', help="If given, use the DatasetSpec with this idenfier, otherwise use the one where the intended use is TEST")
    parser.add_argument('--output-dir', help="Where to save the prediction outputs", type=Path, default=Path())
    parser.add_argument('--n-molecules', help="How many molecules to sample", type=int, default=50)
    parser.add_argument('--random-seed', help="Constant to seed the random number generator with", type=int, default=1234)
    args = parser.parse_args()

    dataset_spec_collection = load_dataset_specs(args.dataset_spec)
    if args.spec_identifier is not None:
        dataset_specs = dataset_spec_collection[args.spec_identifier]
    else:
        dataset_specs = dataset_spec_collection.by_intended_use_all(TEST)

    rng = np.random.default_rng(args.random_seed)
    time_models(dataset_specs, args.experiment_roots, args.n_molecules, args.output_dir, rng)


if __name__ == '__main__':
    main()
    
    
