from typing import Mapping, Sequence, Optional
from riseqsar.dataset.rdkit_dataset import RDKitMolDataset, RDKitMolDatasetConfig

class SequenceDatasetConfig(RDKitMolDatasetConfig):
    def __init__(self, *args,
                 tokenizer_class: type,
                 tokenizer_args: Optional[Sequence]=None,
                 tokenizer_kwargs: Optional[Mapping]=None, **kwargs):
        super(SequenceDatasetConfig, self).__init__(*args, **kwargs)
        if tokenizer_args is None:
            tokenizer_args = tuple()
        if tokenizer_kwargs is None:
            tokenizer_kwargs = dict()
        self.tokenizer_class = tokenizer_class
        self.tokenizer_args = tokenizer_args
        self.tokenizer_kwargs = tokenizer_kwargs


class SequenceDataset(RDKitMolDataset):
    """Dataset which represent molecules as tokenized sequences"""
    def __init__(self, *args, tokenizer, **kwargs):
        super(SequenceDataset, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
