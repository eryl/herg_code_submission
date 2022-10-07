import argparse
import csv
from pathlib import Path
import multiprocessing

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.info')
from tqdm import tqdm

from riseqsar.dataset.dataset_specification import load_dataset_specs
from riseqsar.dataset.smi_dataset import SMIDataset

def check_smiles(indexed_smiles):
    index, smiles = indexed_smiles
    rd_mol = Chem.MolFromSmiles(smiles)
    is_valid = rd_mol is not None
    return index, is_valid


def main():
    parser = argparse.ArgumentParser(description="Script for filtering SMILES dataset for molecules which contain errors")
    parser.add_argument('dataset_specs', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.dataset_specs.parent / 'filtered_molecules'
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset_specs = load_dataset_specs(args.dataset_specs)
    with multiprocessing.Pool() as pool:
        for dataset_spec in dataset_specs:
            dataset = SMIDataset.from_dataset_specs(dataset_spec)
            file_name = dataset_spec.file_name
            name = file_name.with_suffix('').name
            correct_name = output_dir / (name + '_wellformed.csv')
            incorrect_name = output_dir / (name + '_malformed.csv')
            metadata_keys = dataset.get_metadata_keys()
            fieldnames = ['smiles', 'label', 'original_index'] + list(metadata_keys)

            with open(correct_name, 'w') as correct_fp, open(incorrect_name, 'w') as incorrect_fp:
                correct_csv_writer = csv.DictWriter(correct_fp, fieldnames=fieldnames)
                incorrect_csv_writer = csv.DictWriter(incorrect_fp, fieldnames=fieldnames)
                correct_csv_writer.writeheader()
                incorrect_csv_writer.writeheader()
                for i, is_valid in tqdm(pool.imap(check_smiles, enumerate(dataset.get_smiles()), chunksize=10),
                                        total=len(dataset), desc='Checking smiles'):
                    data_row = dict()
                    if dataset.has_properties():
                        smiles, label, properties = dataset[i]
                        data_row.update(properties)
                    else:
                        smiles, label = dataset[i]
                        data_row = dict(original_index=i, smiles=smiles, label=label)

                    if is_valid:
                        correct_csv_writer.writerow(data_row)
                    else:
                        incorrect_csv_writer.writerow(data_row)


if __name__ == '__main__':
    main()