import datetime
import multiprocessing
import os
from dataclasses import dataclass
from tempfile import mkstemp
from typing import List


import pandas as pd
import numpy as np
import rdkit.Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.info')
from rdkit import Chem
from chembl_structure_pipeline import standardizer
from chembl_structure_pipeline import checker

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

#from cachier import cachier
from tqdm import tqdm

from riseqsar.dataset.rdkit_dataset import RDKitMolDataset

@dataclass
class FeaturizerConfig:
    method: str = 'padel'


def make_featurizer(featurizer_config: FeaturizerConfig):
    if featurizer_config.method == 'padel':
        featurizer = PadelMolecularFeaturizer()
    elif featurizer_config.method == 'mordred':
        featurizer = MordredMolecularFeaturizer()
    else:
        raise NotImplementedError(f"Featurizer of type {featurizer_config.method} has not been implemented")
    return featurizer


def smiles_to_mol(smiles, standardize=False):
    m = Chem.MolFromSmiles(smiles, sanitize=True)
    if m is None:
        return None
    #m.UpdatePropertyCache(strict=True)
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

def ensure_rd_mol(m):
    if isinstance(m, str):
        rd_mol = smiles_to_mol(m, standardize=True)
    elif isinstance(m, rdkit.Chem.Mol):
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


class MolecularFeaturizer(object):
    def __init__(self, scaler_type='min_max'):
        self.fitted = False
        if scaler_type == 'min_max':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Scaler of type {scaler_type} is not supported")

    def calculate_descriptors(self, smiles_list):
        raise NotImplementedError("calculate_descriptors() has not been implemented")

    def is_fitted(self):
        return self.fitted

    def fit(self, rdkit_dataset: RDKitMolDataset):
        features = self.calculate_descriptors(rdkit_dataset)
        features_nan_filled = features.replace([np.inf, -np.inf], np.nan)
        self.valid_columns = features.columns[features_nan_filled.notna().all()]
        selected_features = features[self.valid_columns]
        self.scaler.fit(selected_features)
        normalized_features = pd.DataFrame(self.scaler.transform(selected_features.values),
                                           columns=selected_features.columns, index=None)
        self.fill_values = normalized_features.mean()
        self.fitted = True
        return normalized_features

    def featurize(self, data):
        if not self.fitted:
            raise RuntimeError("Featurizer has not been fitted, please call .fit() before using")


        features = self.calculate_descriptors(data)

        selected_features = features[self.valid_columns]
        normalized_features = pd.DataFrame(self.scaler.transform(selected_features.values),
                                           columns=selected_features.columns, index=None)
        na_filled = normalized_features.fillna(self.fill_values)
        return na_filled


#@cachier(stale_after=datetime.timedelta(days=3))
def calculate_mordred_descriptors(molecules, desc=None, ignore_3d=True, num_workers=16):
    # TODO: add caching here, per molecule. This way any molecule we've 
    # used before will already have its descriptors calculated.
    # This will need to first pick out any cached molecules, run 
    # the mordred calculation on the rest and then merge the 
    # results correctly.

    from mordred import Calculator, descriptors
    #mols = [smiles_to_mol(smiles, standardize=True) for smiles in tqdm(smiles_list, desc='Chembl-standardiziation')]
    if desc is None:
        desc = descriptors
    calculator = Calculator(desc, ignore_3D=ignore_3d)
    features = calculator.pandas(molecules, nproc=num_workers)
    return features


class MordredMolecularFeaturizer(MolecularFeaturizer):
    def __init__(self, desc=None, ignore_3d=True, **kwargs):
        super().__init__(**kwargs)
        self.desc = desc
        self.ignore_3d = ignore_3d

    def calculate_descriptors(self, data):
        if isinstance(data, str):
            rd_mols = ensure_rd_mols([data])
        elif isinstance(data, RDKitMolDataset):
            rd_mols = data.molecules
        elif isinstance(data, list):
            rd_mols = ensure_rd_mols(data)
        else:
            raise NotImplementedError(f"Can't handle molecules {data} of type {type(data)}")

        descriptors = calculate_mordred_descriptors(rd_mols, self.desc, self.ignore_3d)
        descriptors.fill_missing(inplace=True)
        return descriptors.astype(float)

    def calculate_descriptors_rdkit(self, rdkit_dataset: RDKitMolDataset):
        descriptors = calculate_mordred_descriptors(rdkit_dataset.molecules, self.desc, self.ignore_3d)
        descriptors.fill_missing(inplace=True)
        return descriptors.astype(float)


#@cachier(stale_after=datetime.timedelta(days=3))
def calculate_padel_descriptors(smiles_list, use_cli=True):
    if use_cli:
        from padelpy import padeldescriptor
        from tempfile import mkstemp
        smiles_fd, smiles_file = mkstemp(suffix='.smi')
        descriptors_fd, descriptors_file = mkstemp(suffix='.csv')
        os.close(smiles_fd)
        os.close(descriptors_fd)
        with open(smiles_file, 'w') as smiles_fp:
            smiles_fp.write('\n'.join(smiles_list))
        padeldescriptor(mol_dir=smiles_file,
                        d_file=descriptors_file,
                        #standardizenitro=True,
                        standardizenitro=False,
                        removesalt=True,
                        #detectaromaticity=True,
                        detectaromaticity=False,
                        convert3d=False,
                        d_2d=True,
                        d_3d=False,
                        retain3d=False,
                        fingerprints=False,
                        retainorder=True,
                        #sp_timeout=50
        )
        os.unlink(smiles_file)
        os.unlink(descriptors_file)
        raise NotImplementedError("parsing of padeldescriptors is not finnished")
    else:
        from padelpy import from_smiles

        molecular_descriptor_dicts = []
        fieldnames = set()
        for smiles in smiles_list:
            descriptors = from_smiles(smiles)
            molecular_descriptor_dicts.append((smiles, descriptors))
            fieldnames.update(descriptors.keys())
        descriptor_lists = {fieldname: [] for fieldname in fieldnames}
        for smiles, descriptors in molecular_descriptor_dicts:
            for name, value in descriptors.items():
                descriptor_lists[name].append(float(value))
        df = pd.DataFrame(descriptor_lists)
        return df


class PadelMolecularFeaturizer(MolecularFeaturizer):
    def calculate_descriptors(self, rdkit_dataset: RDKitMolDataset):
        smiles_list = rdkit_dataset.get_smiles()
        return calculate_padel_descriptors(tuple(smiles_list))


if __name__ == '__main__':
    smiles_list = ['O=Cc1ccc(O)c(OC)c1', 'CN1CCC[C@H]1c2cccnc2']
    target = [0, 1]
    featurizer = PadelMolecularFeaturizer()
    featurizer.fit(smiles_list)
    f = featurizer.featurize(smiles_list[0])
    print(f)