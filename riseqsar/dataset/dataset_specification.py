from collections import defaultdict
from typing import List, Union, Optional, Dict, TypeVar
from pathlib import Path

from riseqsar.util import load_config
from riseqsar.dataset.resampling import ResamplingConfig

TargetSpec = Union['regression', 'classification']


class MissingDatasetSpecError(Exception):
    def __init__(self, spec_id, message):
        self.spec_id = spec_id
        self.message = message


class DatasetSpec:
    def __init__(self,
                 *,
                 identifier: str,
                 file_name: Union[Path, str],
                 intended_use: str,
                 target_spec: Dict[str, TargetSpec],
                 dataset_info: str,
                 dataset_endpoint: str,
                 indices: Optional[List[int]] = None):
        self.identifier = identifier
        self.file_name = file_name
        self.intended_use = intended_use
        self.target_spec = target_spec
        self.dataset_info = dataset_info
        self.dataset_endpoint = dataset_endpoint
        self.indices = indices

    def load_default_dataset(self):
        raise NotImplementedError("DatasetSpec.load_dataset()")

    def make_resample(self, resample_config: ResamplingConfig):
        pass


class SDFDatasetSpec(DatasetSpec):
    metadata_properties: Optional[Dict[str, str]] = None  # The properties of i


class DatasetSpecCollection(object):
    def __init__(self, dataset_specs: List[DatasetSpec], cv_split_dataset=None):
        self.dataset_specs = dataset_specs
        self.intended_uses = defaultdict(list)
        self.cv_split_dataset = cv_split_dataset
        for dataset_spec in dataset_specs:
            self.intended_uses[dataset_spec.intended_use].append(dataset_spec)

    def __iter__(self):
        return iter(self.dataset_specs)

    def __getitem__(self, item):
        for spec in self.dataset_specs:
            if spec.identifier == item:
                return spec
        raise MissingDatasetSpecError(item, f'Dataset with identifier {item} not found')

    def split_by_intended_use(self):
        return self.intended_uses

    def by_intended_use(self, intended_use: str) -> DatasetSpec:
        all_by_intended_use = self.by_intended_use_all(intended_use)
        if len(all_by_intended_use) > 1:
            raise MissingDatasetSpecError(intended_use, f"Multiple dataset with intended use {intended_use}")
        elif len(all_by_intended_use) == 0:
            raise MissingDatasetSpecError(intended_use, f"No dataset with intended use {intended_use}")
        return all_by_intended_use[0]

    def by_intended_use_all(self, intended_use: str) -> List[DatasetSpec]:
        return self.intended_uses[intended_use]

    def set_root(self, root: Path):
        for dataset_spec in self.dataset_specs:
            smiles_path = root / dataset_spec.file_name
            dataset_spec.file_name = smiles_path

    @classmethod
    def from_path(cls, spec_path: Path):
        spec_path = Path(spec_path)
        spec = load_config(spec_path, DatasetSpecCollection)
        spec.set_root(spec_path.parent)
        return spec

    def make_resamples(self, cv_config: ResamplingConfig):
        # If there are predefined cross validation splits in this dataset specification, use that. Otherwise do it
        # dynamically
        if self.cv_split_dataset is None:
            raise MissingDatasetSpecError('cv_split_dataset', "No cv_split_datasets defined for this dataset specs")
        return make_resample_split(self.cv_split_dataset, cv_config)


def load_dataset_specs(spec_path: Path):
    """ Loads a DatasetSpecs from the given file path and replaces all the file names of the individual
    dataset specifications with correct Path objects (using the path relative to the parent of the dataset spec file)"""
    return DatasetSpecCollection.from_path(spec_path)

