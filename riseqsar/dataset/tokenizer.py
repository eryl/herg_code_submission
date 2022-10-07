from typing import Literal

import numpy as np

class BaseTokenizer(object):
    def __init__(self, *, alignment: Literal['left', 'right'], padding_idx=0):
        self.padding_idx = padding_idx
        self.fitted = False
        self.alignment = alignment

    def is_fitted(self):
        return self.fitted

    def get_num_embeddings(self):
        raise NotImplementedError('BaseTokenizer.get_num_embeddings() has not been implemented')

    def fit(self, smiles_list):
        raise NotImplementedError('BaseTokenizer.fit() has not been implemented')

    def tokenize(self, smiles):
        raise NotImplementedError('BaseTokenizer.tokenize() has not been implemented')

    def tokenize_and_pad(self, smiles_list, dtype=np.long):
        """Tokenizes the given smiles list and packs into a matrix with
        shape (n_smiles, max_tokenized_smiles_length). Returns a tuple of the tokenized
        smiles and a boolean mask which is False where there are no valid tokens."""
        tokenized_smiles = [self.tokenize(smiles) for smiles in smiles_list]
        n_smiles = len(smiles_list)
        max_len = max(len(smiles) for smiles in tokenized_smiles)
        smiles_batch = np.full((n_smiles, max_len), self.padding_idx, dtype=dtype)
        #mask = np.zeros((n_smiles, max_len), dtype=np.bool)
        for i, smiles in enumerate(tokenized_smiles):
            k = len(smiles)
            if self.alignment == 'left':
                smiles_batch[i, :k] = smiles
                #mask[i, :k] = True
            elif self.alignment == 'right':
                smiles_batch[i, max_len-k:] = smiles
                #mask[i, max_len-k:] = True
            else:
                raise ValueError(f"Alignment {self.alignment} not recognized")
        mask = smiles_batch != self.padding_idx
        return smiles_batch, mask



class BytesTokenizer(BaseTokenizer):
    """Simple tokenizer which just takes the bytes value of the character"""
    def __init__(self, *args, **kwargs):
        super(BytesTokenizer, self).__init__(*args, **kwargs)
        self.fitted = True

    def fit(self, *args, **kwargs):
        pass

    def tokenize(self, smiles):
        tokenized_smiles = [ord(c) for c in smiles]
        if max(tokenized_smiles) > 127:
            raise ValueError("BytesTokenizer only supports ASCII characters")
        return tokenized_smiles

    def get_num_embeddings(self):
        return 128

