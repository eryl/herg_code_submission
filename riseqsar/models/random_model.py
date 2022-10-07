from riseqsar.models.molecular_predictor import MolecularPredictor
from riseqsar.dataset.smi_dataset import SMIDataset

import numpy as np

class RandomModel():
    def __init__(self, a, b, random_state=None):
        self.a = a
        self.b = b
        self.rng = np.random.default_rng(random_state)

    def fit(self, *args, **kwargs):
        print("RandomModel.fit()")

    def predict(self, n):
        p = self.rng.beta(self.a, self.b, n)
        return self.rng.binomial(1, p, n)

    def predict_proba(self, n):
        return self.rng.beta(self.a, self.b, n)


class RandomPredictor(MolecularPredictor):
    dataset_class = SMIDataset
    def __init__(self, *args, **kwargs):
        super(RandomPredictor, self).__init__(*args, **kwargs)

    def fit(self, *, train_dataset: SMIDataset, **kwargs):
        labels = train_dataset.get_only_targets()
        a = sum(labels)
        b = len(labels) - a
        self.model = RandomModel(a, b, random_state=self.random_state)

    def predict_dataset_proba(self, dataset: SMIDataset):
        return self.model.predict_proba(len(dataset))