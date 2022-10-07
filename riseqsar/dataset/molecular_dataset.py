from collections import Counter, defaultdict
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from riseqsar.dataset.dataset_specification import DatasetSpec
from riseqsar.dataset.resampling import ResamplingConfig, CrossvalidationConfig, SubsamplingConfig

class MolecularDatasetConfig:
    pass

class MolecularDataset(object):
    def __init__(self, *,
                 molecules,
                 properties,
                 target_lists,
                 config: MolecularDatasetConfig,
                 dataset_spec: DatasetSpec,
                 identifier=None,
                 tag=None):
        if identifier is None:
            identifier = dataset_spec.identifier
        self.identifier = identifier
        self.dataset_spec = dataset_spec
        self.molecules = molecules
        self.config = config
        self.properties = properties
        self.target_lists = target_lists
        self.indices = dataset_spec.indices
        self.tag = tag
        if self.indices is not None:
            self.molecules = [molecules[i] for i in self.indices]
            if properties is not None:
                self.properties = [properties[i] for i in self.indices]
            self.target_lists = {target_name: [target_values[i] for i in self.indices]
                                 for target_name, target_values in target_lists.items()}

    def properties_to_csv(self, path):
        """Dumps the properties of this dataset to the given path as a CSV"""
        df = pd.DataFrame.from_records(self.properties)
        df.to_csv(path, index=False)

    def make_copy(self, dataset_spec=None, identifier=None, tag=None):
        "Return a copy of this dataset"
        if dataset_spec is None:
            dataset_spec = copy.deepcopy(self.dataset_spec)
        dataset_class = type(self)
        dataset_copy = dataset_class(molecules = copy.deepcopy(self.molecules), 
                                     properties = copy.deepcopy(self.properties), 
                                     target_lists = copy.deepcopy(self.target_lists), 
                                     config = copy.deepcopy(self.config),
                                     dataset_spec = dataset_spec,
                                     identifier = identifier,
                                     tag=tag)
        return dataset_copy


    def __getitem__(self, item):
        #if self.properties is not None:
#            return self.molecules[item], *[target_values[item] for target_values in self.target_lists.values()], self.properties[item]
#        else:
        return self.molecules[item], *[target_values[item] for target_values in self.target_lists.values()],

    def get_metadata_keys(self):
        if self.properties is not None:
            return self.properties[0].keys()
        else:
            return []
    
    def has_properties(self):
        return self.properties is not None

    def add_properties(self, prop_name, new_properties):
        """Adds properties to the dataset"""
        if not self.has_properties():
            self.properties = [dict() for i in range(len(self))]
        assert len(new_properties) == len(self), "There needs to be as many new properties as there are molecules in this dataset"
        for prop_dict, prop in zip(self.properties, new_properties):
            prop_dict[prop_name] = prop

    def get_samples_weights(self):
        targets = self.get_only_targets()
        target_class_count = Counter(targets)
        class_weight = {target_class: 1 / count for target_class, count in sorted(target_class_count.items())}
        samples_weight = np.array([class_weight[t] for t in targets])
        return samples_weight

    def get_only_targets(self):
        '''Returns the target values, after checking that there is only one'''
        targets = self.get_targets()
        if len(targets) != 1:
            raise ValueError('There is not exactly one target, can not get an only target')
        target_name, target_values = targets[0]
        return target_values

    def get_targets(self):
        # .items() might return an iterator over special dict_item object, the following instead allows the user to
        # directly select a target by index (e.g. 0 to select the first)
        return [(target_name, target_value) for target_name, target_value in self.target_lists.items()]

    def __len__(self):
        return len(self.molecules)



    def get_smiles(self):
        raise NotImplementedError(f'get_smiles has not been implemented for {self.__class__.__name__}')

    def merge(self, other_dataset):
        '''Merge two datasets. This does not check for duplicate entries'''
        molecules = self.molecules + other_dataset.molecules
        properties = self.properties + other_dataset.properties
        target_lists = copy.deepcopy(self.target_lists)
        for target, target_values in other_dataset.get_targets():
            target_lists[target].extend(target_values)

        identifier = '_'.join(sorted({self.get_identifier(), other_dataset.get_identifier()}))
        tag = '_'.join(sorted({self.get_tag(), other_dataset.get_tag()}))

        # Since this dataset might have been derived from selected of the dataset, we have to fix that here
        dataset_spec = copy.deepcopy(self.dataset_spec)
        dataset_spec.file_name = None
        dataset_spec.identifier = identifier
        dataset_spec.indices = None

        return type(self)(molecules=molecules,
                          properties=properties,
                          target_lists=target_lists,
                          config=self.config,
                          dataset_spec=dataset_spec,
                          identifier=identifier,
                          tag=tag)

    def select(self, indices):
        dataset_spec = copy.deepcopy(self.dataset_spec)
        dataset_spec.indices = indices
        return type(self)(molecules=copy.deepcopy(self.molecules),
                          properties=copy.deepcopy(self.properties),
                          target_lists=copy.deepcopy(self.target_lists),
                          config=self.config,
                          dataset_spec=dataset_spec)

    def __str__(self):
        if self.tag is None:
            return self.dataset_spec.identifier
        else:
            return f'{self.dataset_spec.identifier}_{self.tag}'

    def get_identifier(self):
        return self.identifier

    def get_tag(self):
        return self.tag

    def set_tag(self, tag):
        self.tag = tag

    @classmethod
    def from_dataset_spec(cls, *, dataset_spec: DatasetSpec, indices=None, config=None, tag=None, rng=None, **kwargs):
        raise NotImplementedError("MolecularDataset.from_dataset_spec()")

    def make_resamples(self, resample_config: ResamplingConfig, tag=None, rng=None):
        from riseqsar.dataset.dataset_specification import DatasetSpecCollection

        if rng is None:
            rng = np.random.default_rng()

        if isinstance(resample_config, CrossvalidationConfig):
            # TODO: Implement cross-validation resampling
            raise NotImplementedError(f'Resample strategy config of type {type(resample_config)} needs to be rewritten')
            if resample_config.mol_sample_strategy == 'stratified':
                dataset = self
                if len(dataset.target_lists) != 1:
                    raise ValueError('Cannot perform stratified split with more than one target')
                target_values = dataset.get_only_targets()
                labels = np.array(target_values)

                random_seed = rng.integers(0, 2**32-1).item()
                skf = StratifiedKFold(n_splits=resample_config.n_subsamples, random_state=random_seed, shuffle=True)

                # run 5 fold CV
                used_indices = set()
                for i, (visible_indices, heldout_indices) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                    if used_indices.intersection(heldout_indices):
                        raise RuntimeError(f"There is overlap in heldout indices: {used_indices.intersection(heldout_indices)}")
                    used_indices.update(heldout_indices)
                    visible_labels = labels[visible_indices]
                    random_seed = rng.integers(0, 2**32-1).item()
                    train_indices, dev_indices = train_test_split(visible_indices,
                                                                test_size=resample_config.dev_ratio,
                                                                stratify=visible_labels, random_state=random_seed)
                    #print(f"Fold {i} stats")
                    for name, split_indices in [('train', train_indices), ('val', dev_indices), ('test', heldout_indices)]:
                        split_labels = labels[split_indices]
                        total = len(split_labels)
                        count = Counter(split_labels)
                        split_ratios = ','.join(f'{label}:{label_count / total}' for label, label_count in sorted(count.items()))
                        #print(f'Ratios for {name}: {split_ratios}')

                    train_dataset_spec = copy.deepcopy(dataset_spec)
                    train_dataset_spec.indices = sorted(train_indices.tolist())
                    train_dataset_spec.intended_use = TRAIN
                    dev_dataset_spec = copy.deepcopy(dataset_spec)
                    dev_dataset_spec.indices = sorted(dev_indices.tolist())
                    dev_dataset_spec.intended_use = DEV
                    test_dataset_spec = copy.deepcopy(dataset_spec)
                    test_dataset_spec.indices = sorted(heldout_indices.tolist())
                    test_dataset_spec.intended_use = TEST
                    yield DatasetSpecCollection([train_dataset_spec, dev_dataset_spec, test_dataset_spec])

        elif isinstance(resample_config, SubsamplingConfig):
            if resample_config.mol_sample_strategy == 'stratified':
                dataset = self
                if len(dataset.target_lists) != 1:
                    raise ValueError('Cannot perform stratified split with more than one target')
                target_values = dataset.get_only_targets()
                labels = np.array(target_values)

                labeled_indices = defaultdict(list)
                for i,label in enumerate(labels):
                    labeled_indices[label].append(i)
                
                for i in range(resample_config.n_subsamples):
                    shuffled_labeled_indices = {label: rng.permuted(indices) for label, indices in labeled_indices.items()}
                    subsamples = {intended_use: [] for intended_use in resample_config.subsampling_ratios.keys()}

                    for label, shuffled_indices in shuffled_labeled_indices.items():
                        start = 0
                        for intended_use, sample_ratio in resample_config.subsampling_ratios.items():
                            sample_n = int(len(shuffled_indices)*sample_ratio)
                            end = start + sample_n
                            sample = shuffled_indices[start:end]
                            subsamples[intended_use].extend(sample)
                            start = end
                    
                    datasets = dict()
                    for intended_use, sampled_indices in subsamples.items():
                        subsample_dataset_spec = copy.deepcopy(self.dataset_spec)
                        subsample_dataset_spec.indices = sorted(sampled_indices)
                        subsample_dataset_spec.intended_use = intended_use
                        subsample_dataset = self.make_copy(subsample_dataset_spec, tag=tag)
                        datasets[intended_use] = subsample_dataset

                    indices = [set(dataset.dataset_spec.indices) for dataset in datasets.values()]
                    from itertools import combinations
                    for d1, d2 in combinations(indices, 2):
                        assert(len(d1.intersection(d2)) == 0)
                    

                    yield datasets
            else:
                raise NotImplementedError(f"Resample strategy {resample_config.mol_sample_strategy} has not been implemented")

        else:
            raise NotImplementedError(f'Resample strategy config of type {type(resample_config)} is not supported')
            # # Random uniform sampling
            # indices = np.arange(len(dataset))
            # rng = np.random.default_rng(seed)
            # rng.shuffle(indices)
            # fold_length = int(math.ceil(len(dataset)/n_folds))
            # for i in range(n_folds):
            #     start = i*fold_length
            #     end = start + fold_length
            #     heldout_indices = indices[start:end]
            #     visible_indices = np.concatenate([indices[:start], indices[end:]])  #  Hopefully this copies the indices, make sure it does!
            #     # For now dev-indices are just uniformly sampled from the visible indices. An alternative could be to use
            #     # the folds to ensure no overlap over dev sets, but that would assume the dev-ratio is  less than 1/n_folds
            #     rng.shuffle(visible_indices)
            #     n_dev_examples = int(math.ceil(len(visible_indices)*dev_ratio))
            #     dev_indices = visible_indices[:n_dev_examples]
            #     train_indices = visible_indices[n_dev_examples:]
            #     splits = Splits(train_indices=sorted(train_indices.tolist()),
            #                     dev_indices=sorted(dev_indices.tolist()),
            #                     test_indices=sorted(heldout_indices.tolist()),
            #                     dataset_specs=dataset_specs_path,
            #                     dataset_identifier=dataset_identifier,
            #                     dataset_length=len(train_indices) + len(dev_indices) + len(heldout_indices),
            #                     random_seed=seed,
            #                     fold_i=i,
            #                     n_folds=n_folds,
            #                     dev_ratio=dev_ratio,
            #                     stratified=stratified)
            #     yield splits
            pass

