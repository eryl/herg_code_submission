import copy
from dataclasses import dataclass, field
from distutils.command.config import config
from typing import Literal, List, Tuple, Sequence, Union, Dict, Optional
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np

from riseqsar.dataset.constants import TRAIN, DEV, TEST



class ResamplingConfig:
    def __init__(self, *, n_subsamples: int, 
                          random_seed: int,  
                          resample_name_prefix: Optional[str] = None,
                          mol_sample_strategy: Literal['uniform', 'stratified', 'scaffold'] = 'uniform'):
        self.n_subsamples = n_subsamples
        self.random_seed = random_seed
        self.resample_name_prefix = resample_name_prefix
        self.mol_sample_strategy = mol_sample_strategy


class CrossvalidationConfig(ResamplingConfig):
    pass


class SubsamplingConfig(ResamplingConfig):
    def __init__(self, *, subsampling_ratios: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.subsampling_ratios = subsampling_ratios


# def make_resample_split(dataset_spec: DatasetSpec, resample_config: ResamplingConfig, rng=None):
#     from riseqsar.dataset.dataset_specification import DatasetSpecCollection

#     if rng is None:
#         rng = np.random.default_rng(resample_config.random_seed)

#     if isinstance(resample_config, CrossvalidationConfig):
#         if resample_config.mol_sample_strategy == 'stratified':
#             dataset = dataset_spec.load_default_dataset()
#             if len(dataset.target_lists) != 1:
#                 raise ValueError('Cannot perform stratified split with more than one target')
#             target_values = dataset.get_only_targets()
#             labels = np.array(target_values)

#             random_seed = rng.integers(0, 2**32-1).item()
#             skf = StratifiedKFold(n_splits=resample_config.n_subsamples, random_state=random_seed, shuffle=True)

#             # run 5 fold CV
#             used_indices = set()
#             for i, (visible_indices, heldout_indices) in enumerate(skf.split(np.zeros(len(labels)), labels)):
#                 if used_indices.intersection(heldout_indices):
#                     raise RuntimeError(f"There is overlap in heldout indices: {used_indices.intersection(heldout_indices)}")
#                 used_indices.update(heldout_indices)
#                 visible_labels = labels[visible_indices]
#                 random_seed = rng.integers(0, 2**32-1).item()
#                 train_indices, dev_indices = train_test_split(visible_indices,
#                                                             test_size=resample_config.dev_ratio,
#                                                             stratify=visible_labels, random_state=random_seed)
#                 #print(f"Fold {i} stats")
#                 for name, split_indices in [('train', train_indices), ('val', dev_indices), ('test', heldout_indices)]:
#                     split_labels = labels[split_indices]
#                     total = len(split_labels)
#                     count = Counter(split_labels)
#                     split_ratios = ','.join(f'{label}:{label_count / total}' for label, label_count in sorted(count.items()))
#                     #print(f'Ratios for {name}: {split_ratios}')

#                 train_dataset_spec = copy.deepcopy(dataset_spec)
#                 train_dataset_spec.indices = sorted(train_indices.tolist())
#                 train_dataset_spec.intended_use = TRAIN
#                 dev_dataset_spec = copy.deepcopy(dataset_spec)
#                 dev_dataset_spec.indices = sorted(dev_indices.tolist())
#                 dev_dataset_spec.intended_use = DEV
#                 test_dataset_spec = copy.deepcopy(dataset_spec)
#                 test_dataset_spec.indices = sorted(heldout_indices.tolist())
#                 test_dataset_spec.intended_use = TEST
#                 yield DatasetSpecCollection([train_dataset_spec, dev_dataset_spec, test_dataset_spec])

#     elif isinstance(resample_config, SubsamplingConfig):
#         if resample_config.mol_sample_strategy == 'stratified':
#             dataset = dataset_spec.load_default_dataset()
#             if len(dataset.target_lists) != 1:
#                 raise ValueError('Cannot perform stratified split with more than one target')
#             target_values = dataset.get_only_targets()
#             labels = np.array(target_values)

#             labeled_indices = defaultdict(list)
#             for i,label in enumerate(labels):
#                 labeled_indices[label].append(i)
            
#             for i in range(resample_config.n_subsamples):
#                 shuffled_labeled_indices = {label: rng.permuted(indices) for label, indices in labeled_indices.items()}
#                 subsamples = {intended_use: [] for intended_use in resample_config.subsampling_ratios.keys()}

#                 for label, shuffled_indices in shuffled_labeled_indices.items():
#                     start = 0
#                     for intended_use, sample_ratio in resample_config.subsampling_ratios.items():
#                         sample_n = int(len(shuffled_indices)*sample_ratio)
#                         end = start + sample_n
#                         sample = shuffled_indices[start:end]
#                         subsamples[intended_use].extend(sample)  

#                 dataset_specs = []
#                 for intended_use, sampled_indices in subsamples.items():
#                     subsample_dataset_spec = copy.deepcopy(dataset_spec)
#                     subsample_dataset_spec.indices = sorted(sampled_indices)
#                     subsample_dataset_spec.intended_use = intended_use
#                     dataset_specs.append(subsample_dataset_spec)
                
#                 yield DatasetSpecCollection(dataset_specs)
#         else:
#             raise NotImplementedError(f"Resample strategy {resample_config.mol_sample_strategy} has not been implemented")

#     else:
#         raise NotImplementedError(f'Resample strategy config of type {type(resample_config)} is not supported')
#         # # Random uniform sampling
#         # indices = np.arange(len(dataset))
#         # rng = np.random.default_rng(seed)
#         # rng.shuffle(indices)
#         # fold_length = int(math.ceil(len(dataset)/n_folds))
#         # for i in range(n_folds):
#         #     start = i*fold_length
#         #     end = start + fold_length
#         #     heldout_indices = indices[start:end]
#         #     visible_indices = np.concatenate([indices[:start], indices[end:]])  #  Hopefully this copies the indices, make sure it does!
#         #     # For now dev-indices are just uniformly sampled from the visible indices. An alternative could be to use
#         #     # the folds to ensure no overlap over dev sets, but that would assume the dev-ratio is  less than 1/n_folds
#         #     rng.shuffle(visible_indices)
#         #     n_dev_examples = int(math.ceil(len(visible_indices)*dev_ratio))
#         #     dev_indices = visible_indices[:n_dev_examples]
#         #     train_indices = visible_indices[n_dev_examples:]
#         #     splits = Splits(train_indices=sorted(train_indices.tolist()),
#         #                     dev_indices=sorted(dev_indices.tolist()),
#         #                     test_indices=sorted(heldout_indices.tolist()),
#         #                     dataset_specs=dataset_specs_path,
#         #                     dataset_identifier=dataset_identifier,
#         #                     dataset_length=len(train_indices) + len(dev_indices) + len(heldout_indices),
#         #                     random_seed=seed,
#         #                     fold_i=i,
#         #                     n_folds=n_folds,
#         #                     dev_ratio=dev_ratio,
#         #                     stratified=stratified)
#         #     yield splits



