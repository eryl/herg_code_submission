import argparse
import csv
from pathlib import Path

from riseqsar.experiment.experiment import ExperimentTracker, find_experiment_top_level_models


def main():
    parser = argparse.ArgumentParser(description="Bundles experiments to packages suitable for serving")
    parser.add_argument('smi_file',
                        help="Path to SMI file, containing one SMILES per row",
                        type=Path)
    parser.add_argument('experiment_directories',
                        help="Path to experiment directories to scan for models. Will recursively go through all subdirectories",
                        type=Path, nargs='+')
    parser.add_argument('--output-dir', help="Save packaged models to this directory", type=Path, default='predictions')
    args = parser.parse_args()

    experiments_paths = []
    for experiment_directory in args.experiment_directories:
        experiments_paths.extend(find_experiment_top_level_models(experiment_directory))

    for experiment_path in experiments_paths:
        experiment_tracker = ExperimentTracker(experiment_path)
        model = experiment_tracker.load_model('final_model')
        predictions_output_dir = args.output_dir / experiment_tracker.get_identifier()
        predictions_output_dir.mkdir(parents=True, exist_ok=True)
        predictions_output_file = predictions_output_dir / args.smi_file.stem
        with open(args.smi_file) as smiles_fp, open(predictions_output_file.with_suffix('.csv'), 'w') as out_fp:
            csv_writer = csv.DictWriter(out_fp, fieldnames=['smiles', 'prediction'])
            csv_writer.writeheader()
            for smiles in smiles_fp:
                smiles = smiles.strip()
                prediction = model.predict(smiles)
                csv_writer.writerow({'smiles': smiles, 'prediction': prediction})


if __name__ == '__main__':
    main()