from riseqsar.models.molecular_predictor import DescriptorbasedPredictor
import numpy as np

class DummyModel(object):
    def fit(self, *args, **kwargs):
        print("DummyModel fit()")

    def predict(self, x):
        return np.zeros(x.shape[0])

    def predict_proba(self, x):
        return np.zeros(x.shape[0])


class DummyPredictor(DescriptorbasedPredictor):
    def __init__(self, *args, **kwargs):
        super(DummyPredictor, self).__init__(*args, **kwargs)
        self.model = DummyModel()