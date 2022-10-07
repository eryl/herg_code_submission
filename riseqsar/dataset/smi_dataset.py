import pickle
from typing import Union, List, Literal, Optional
from pathlib import Path

import pandas as pd

from riseqsar.dataset.dataset_specification import DatasetSpec
from riseqsar.dataset.molecular_dataset import MolecularDataset

class SMIDatasetSpec(DatasetSpec):
    def __init__(self,
                 *args,
                 smiles_col: Union[int, str],
                 target_col: Union[int, str, List[Union[int, str]]],
                 skip_header: bool = False,
                 sep: str = ',',
                 comment: Optional[str] = None,
                 skip_rows: Optional[int] = None,
                 **kwargs):
        super(SMIDatasetSpec, self).__init__(*args, **kwargs)
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.skip_header = skip_header
        self.sep = sep
        self.comment = comment
        self.skip_rows = skip_rows

    def load_default_dataset(self):
        return SMIDataset.from_dataset_specs(self)


class SMIDataset(MolecularDataset):
    def __init__(self, *args, smiles_list, **kwargs):
        super(SMIDataset, self).__init__(*args, **kwargs)
        self.smiles_list = smiles_list
        if self.indices is not None:
            new_smiles_list = [smiles_list[i] for i in self.indices]
            self.smiles_list = new_smiles_list

    def get_smiles(self):
        return self.molecules

    @classmethod
    def from_dataset_spec(cls, *, dataset_spec: SMIDatasetSpec, indices = None, config=None, tag=None, rng = None, ** kwargs):
        file_name = Path(dataset_spec.file_name)
        base_name = file_name.with_suffix('').name
        dataset_filename = file_name.with_name(base_name + f'_smidataset.pkl')

        if dataset_filename.exists():
            with open(dataset_filename, 'rb') as fp:
                dataset_fields = pickle.load(fp)
            molecules = dataset_fields['molecules']
            properties = dataset_fields['properties']
            target_lists = dataset_fields['target_lists']
            return cls(molecules=molecules,
                       smiles_list=molecules,
                       properties=properties,
                       target_lists=target_lists,
                       config=config,
                       dataset_spec=dataset_spec)
        else:
            if dataset_spec.skip_header:
                header = None
            else:
                header = 'infer'
            data = pd.read_csv(dataset_spec.file_name, sep=dataset_spec.sep, header=header)
            remaining_columns = list(enumerate(data.columns))

            smiles_col = dataset_spec.smiles_col
            label_cols = dataset_spec.target_col
            try:
                smiles_col = int(smiles_col)
                remaining_columns = [(idx, name) for idx, name in remaining_columns if idx != smiles_col]
                smiles_list = data.iloc[:, smiles_col]
            except ValueError:
                smiles_list = data[smiles_col]
                remaining_columns = [(idx, name) for idx, name in remaining_columns if name != smiles_col]

            #Add support for incomplete smiles lists
            smiles_list_valid_idx = [i for i,smiles in enumerate(smiles_list) if isinstance(smiles, str)]
            smiles_list = smiles_list.iloc[smiles_list_valid_idx]
            data = data.iloc[smiles_list_valid_idx]
            
            if label_cols is not None:
                try:
                    label_cols = int(label_cols)
                    remaining_columns = [(idx, name) for idx, name in remaining_columns if idx != label_cols]
                    labels = data.iloc[:, label_cols]
                except ValueError:
                    labels = data[label_cols]
                    remaining_columns = [(idx, name) for idx, name in remaining_columns if name != label_cols]
            else:
                labels = [float('nan')]*len(smiles_list)

            #metadata = data.iloc[:, [idx for idx, name in remaining_columns]]
            metadata = data  # Keep all data for now as properties
            if 0 in metadata.shape:
                properties = None
            else:
                properties = metadata.to_dict('records')

            molecules = smiles_list.tolist()
            target_lists = {'label': labels}
            with open(dataset_filename, 'wb') as fp:
                pickle.dump(dict(molecules=molecules,
                                 properties=properties,
                                 target_lists=target_lists),
                            fp)

            return cls(molecules=molecules,
                       smiles_list=smiles_list,
                       properties=properties,
                       target_lists=target_lists,
                       config=config,
                       dataset_spec=dataset_spec,
                       tag=tag)

