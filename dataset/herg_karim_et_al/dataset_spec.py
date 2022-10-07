from riseqsar.dataset.dataset_specification import DatasetSpec, DatasetSpecCollection
from riseqsar.dataset.smi_dataset import SMIDatasetSpec
from riseqsar.dataset.constants import TEST


test_dataset = SMIDatasetSpec(identifier='herg_karim_et_tal',
                              file_name='karim_et_al_803_new_compounds.csv',
                              smiles_col=1,
                              target_col=0,
                              intended_use=TEST,
                              target_spec = {'label': 'classification'},
                              dataset_info = '',
                              dataset_endpoint = 'hERG')

datasets = DatasetSpecCollection(dataset_specs=[test_dataset], cv_split_dataset=None)
