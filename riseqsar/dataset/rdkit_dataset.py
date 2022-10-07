import pickle
from dataclasses import dataclass
import gzip
from typing import List, Optional
import multiprocessing
from pathlib import Path

import pandas as pd
import numpy as np

import rdkit.Chem as rdc
from tqdm import tqdm

from riseqsar.dataset.molecular_dataset import MolecularDataset, MolecularDatasetConfig
from riseqsar.dataset.dataset_specification import DatasetSpec, SDFDatasetSpec
from riseqsar.dataset.smi_dataset import SMIDatasetSpec, SMIDataset

def shuffle_atoms(rd_mol, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_atoms = list(range(rd_mol.GetNumAtoms()))
    rng.shuffle(n_atoms)
    m_rand = rdc.RenumberAtoms(rd_mol, n_atoms)
    return m_rand

def smiles_to_mol(smiles, standardize=True):
    from chembl_structure_pipeline import standardizer

    m = rdc.MolFromSmiles(smiles, sanitize=False)
    m.UpdatePropertyCache(strict=False)
    if standardize:
        m = standardizer.standardize_mol(m)
        m, exclude = standardizer.get_parent_mol(m)
    #m = checker.check_molblock(Chem.MolToMolBlock(m))
    #Chem.SanitizeMol(m,
                     # (Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                     #  Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                     #  Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                     #  Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                     #  Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                     #  Chem.SanitizeFlags.SANITIZE_SYMMRINGS),
                     # catchErrors=True)
    return m


def mol_to_smiles(mol_list, augment=False, canonical=True, rng=None):
    if augment:
        if rng is None:
            rng = np.random.default_rng()
        mol_list = [shuffle_atoms(rd_mol, rng) for rd_mol in mol_list]
        ## When augmenting the smiles, using canonical form makes no sense
        return [rdc.MolToSmiles(mol, canonical=False) for mol in mol_list]
    else:
        return [rdc.MolToSmiles(mol, canonical=canonical) for mol in mol_list]


def ensure_rd_mol(m):
    if isinstance(m, str):
        rd_mol = smiles_to_mol(m, standardize=True)
    elif isinstance(m, rdc.Mol):
        rd_mol = m
    else:
        raise NotImplementedError(f"Can't handle molecule {m} of type {type(m)}")
    return rd_mol

def ensure_rd_mols(molecules):
    rd_mols = []
    with multiprocessing.Pool() as pool:
        for i, rd_mol in tqdm(enumerate(pool.imap(ensure_rd_mol, molecules)), desc='Loading RDKit Molecules', total=len(molecules)):
            rd_mols.append(rd_mol)
    return rd_mols


class RDKitMolDatasetConfig(MolecularDatasetConfig):
    pass


class RDKitMolDataset(MolecularDataset):
    def get_smiles(self, augment=False):
        if augment is False:
            return self.smiles
        else:
            return mol_to_smiles(self.molecules, augment=True)

    @classmethod
    def from_dataset_spec(cls, *, dataset_spec: DatasetSpec, config: RDKitMolDatasetConfig=None, indices = None, tag=None, **kwargs):
        if config is None:
            config = RDKitMolDatasetConfig()
        file_name = Path(dataset_spec.file_name)
        base_name = file_name.with_suffix('').name
        dataset_filename = file_name.with_name(base_name + f'_rdkitdataset.pkl')

        if dataset_filename.exists():
            with open(dataset_filename, 'rb') as fp:
                dataset_fields = pickle.load(fp)
            molecules = dataset_fields['molecules']
            properties = dataset_fields['properties']
            target_lists = dataset_fields['target_lists']
        else:
            if isinstance(dataset_spec, SMIDatasetSpec):
                smi_dataset = SMIDataset.from_dataset_spec(dataset_spec=dataset_spec)
                molecules = ensure_rd_mols(smi_dataset.molecules)
                target_lists = smi_dataset.target_lists
                properties = smi_dataset.properties
                with open(dataset_filename, 'wb') as fp:
                    pickle.dump(dict(molecules=molecules,
                                     properties = properties,
                                     target_lists = target_lists),
                                fp)

            elif isinstance(dataset_spec, SDFDatasetSpec):
                sdf_path = dataset_spec.file_name
                metadata_properties = dataset_spec.metadata_properties
                if sdf_path.suffix == '.gz':
                    file = gzip.open(sdf_path)
                else:
                    file = open(sdf_path, 'rb')

                i = 0
                invalid_mols = []
                target_lists = {target_name: [] for target_name in dataset_spec.target_spec.keys()}

                supplier = rdc.ForwardSDMolSupplier(file)
                molecules = []
                properties = []
                for j, mol in tqdm(enumerate(supplier), desc='Parsing SDF'):
                    if mol is None:
                        invalid_mols.append(i)
                    else:
                        mol_props = dict()
                        for sdf_prop_name, prop_name in metadata_properties.items():
                            try:
                                mol_props[prop_name] = mol.GetProp(sdf_prop_name)
                            except KeyError:
                                mol_props[prop_name] = None
                        for target_name in dataset_spec.target_spec.keys():
                            try:
                                target_value = float(mol_props[target_name])
                            except ValueError:
                                target_value = mol_props[target_name]

                            target_lists[target_name].append(target_value)
                        molecules.append(mol)
                        properties.append(mol_props)
                    i += 1

                if invalid_mols:
                    invalid_path = file_name.with_name('invalid_mols_' + sdf_path.with_suffix('.txt').name)
                    invalid_mols_str = ','.join(str(m) for m in invalid_mols)
                    with open(invalid_path, 'w') as invalid_fp:
                        invalid_fp.write(f'Invalid molecules:\n {invalid_mols_str}\n')

                with open(dataset_filename, 'wb') as fp:
                    pickle.dump(dict(molecules=molecules,
                                     properties=properties,
                                     target_lists=target_lists),
                                fp)
            else:
                raise NotImplementedError(f"Loading dataset for {dataset_spec} has not been implemented")

        return cls(molecules=molecules,
                   properties=properties,
                   target_lists=target_lists,
                   config=config,
                   dataset_spec=dataset_spec,
                   **kwargs)


