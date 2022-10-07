from riseqsar.dataset.dataset_specification import DatasetSpec, DatasetSpecCollection
from riseqsar.dataset.smi_dataset import SMIDatasetSpec
from riseqsar.dataset.constants import TRAIN, TEST

training_dataset = SMIDatasetSpec(identifier='herg_ogura_train_filtered',
                                  file_name='train_set_smiles_toxicity_class_wellformed.csv',
                                  smiles_col=0, target_col=1,
                                  intended_use=TRAIN,
                                  target_spec = {'label': 'classification'},
                                  dataset_info = '',
                                  dataset_endpoint = 'hERG')

test_dataset = SMIDatasetSpec(identifier='herg_ogura_test_filtered',
                              file_name='test_set_smiles_toxicity_class_wellformed.csv',
                              smiles_col=0,
                              target_col=1,
                              intended_use=TEST,
                              target_spec = {'label': 'classification'},
                              dataset_info = '',
                              dataset_endpoint = 'hERG')

datasets = DatasetSpecCollection(dataset_specs=[training_dataset, test_dataset], cv_split_dataset=training_dataset)
