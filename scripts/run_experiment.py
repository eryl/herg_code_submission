import argparse
import argparse
from pathlib import Path
import subprocess
from itertools import product
import sys

from riseqsar.experiment.experiment_config import ExperimentSpecificationCollection, make_experiment_config
from riseqsar.experiment.experiment import run_experiment
from riseqsar.util import load_config, timestamp

def main():
    parser = argparse.ArgumentParser(description="Run an experiment given the experiment config file")
    parser.add_argument('experiment_config', type=Path, help="Path to a python file containing an ExperimentConfiguration object")
    parser.add_argument('--experiment-ids', help="Only run on the experiments with these indices in the experiments collection", type=int, nargs='+')
 
    args = parser.parse_args()

    experiment_config_path = args.experiment_config
    experiment_spec_collection = load_config(experiment_config_path, ExperimentSpecificationCollection)
    print(experiment_spec_collection) 

    cli_args = dict()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, list):
            value = [str(x) for x in value]
        cli_args[name] = value
    metadata = {'command_line_args': cli_args}

    root_directory = Path(experiment_spec_collection.output_dir) / experiment_spec_collection.name 

    for i, experiment_specification in enumerate(experiment_spec_collection.experiments):
        # This code was to try to run a subprocess with a different conda environment
        #cmd = ['conda', 'run', '-n', model_env.model_environment, 'python', str(Path('scripts/train_smiles_predictor.py')), model_env.model_config, dataset_spec, '--output-dir', experiment_config.output_dir]
        #cmd = ['conda', 'run', '-n', model_env.model_environment, 'python', '-c', 'from riseqsar.training.experiment import ExperimentTracker']
        # This will be replaced by a proper subprocess executor in the correct environment
        # cmd = ['python', str(Path('scripts/train_smiles_predictor.py')), model_env.model_config, dataset_spec, '--output-dir', experiment_config.output_dir]
        # print(cmd)
        # if num_workers is not None: 
        #     cmd.extend(['--num-workers', str(num_workers)])
        # completed_process = subprocess.run(cmd, shell=True)
        # print(completed_process)
        if args.experiment_ids:
            if i not in args.experiment_ids:
                print(f"Skipping experiment {experiment_specification.name} with id {i} not in the --experiments-id arguments")
                continue
            
        experiment_config = make_experiment_config(experiment_specification)
        artifacts = {'experiment_specification_collection': experiment_spec_collection,
                     'experiment_specification': experiment_specification,
                     'experiment_config': experiment_config}
        files = {'config_file.py': args.experiment_config,
                'model_specification.py': experiment_specification.model_spec_path,
                'dataset_spec.py': experiment_specification.dataset_spec_path}

        output_dir = root_directory / experiment_specification.name / timestamp()
        
        run_experiment(experiment_config=experiment_config, output_dir=output_dir, artifacts=artifacts, files=files, metadata=metadata)

if __name__ == '__main__':
    main()



