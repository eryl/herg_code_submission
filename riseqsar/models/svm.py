import copy
import pickle
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore')

import numpy as np

#from sklearn.svm import SVC
from thundersvm import SVC, SVR

from riseqsar.models.molecular_predictor import MolecularPredictorConfig
from riseqsar.dataset.featurized_dataset import FeaturizedDataset
from riseqsar.models.descriptor_based_predictor import DescriptorbasedPredictor

def deserializes(model_bytes):
    model = pickle.loads(model_bytes)
    with NamedTemporaryFile('wb') as fp:
        fp.write(model.model)
        model.model = SVC(random_state=model.random_state, probability=1, **asdict(model.config))
        model.model.load_from_file(fp.name)
    return model


@dataclass
class SVMClassifierConfig(MolecularPredictorConfig):
    gpu_id: int
    C: float
    gamma: float
    kernel: str ='rbf'
    class_weight: str ='balanced'

class SVMClassifier(DescriptorbasedPredictor):
    def __init__(self, *args, config: SVMClassifierConfig, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.config = config
        self.model = SVC(random_state=self.random_state, probability=1, **asdict(self.config))

    def fit(self, *, train_dataset: FeaturizedDataset, dev_dataset=None,
            experiment_tracker=None, evaluation_metrics=None):
        if self.featurizer is None:
            self.featurizer = train_dataset.featurizer

        featurized_mols = train_dataset.features
        featurized_mols_ndarray = featurized_mols.values

        if len(train_dataset.get_targets()) != 1:
            raise ValueError("SVMClassifier can not be used for multiple targets")

        target_name, target_values = train_dataset.get_targets()[0]
        self.model.fit(featurized_mols_ndarray, target_values)

        # Stupid thunder SVM orders the classes in the prediction according to the first training example. We need to
        # figure out which column is the prediction for the positive class. We assume the probabilities are calibrated
        # and can then match the highest probability to the class prediction

        tr_pred = self.model.predict_proba(featurized_mols_ndarray[:1])
        class_prediction_0 = self.model.predict(featurized_mols_ndarray[:1])  # We only compare to the first sample
        class_0_col = np.argmax(tr_pred[:1])  # Which column had the highest probability, we're assuming that is the one the model predicts
        # We wan't to know which column is the positive class. If the class prediction is positive, the column with highest mass is chosen.
        if class_prediction_0 > 0:  # The class was predicted positive
            self.positive_col = class_0_col
        else:  # Predicted negative, positive class is the other one
            self.positive_col = 1 - class_0_col

    def predict_proba_featurized(self, featurized_mols):
        pred_prediction = self.model.predict_proba(featurized_mols)
        positive_prediction = pred_prediction[:, self.positive_col]
        return positive_prediction

    def save(self, output_dir: Path, tag=None):
        if tag is None:
            model_name = f'{self.__class__.__name__}.pkl'
        else:
            model_name = f'{self.__class__.__name__}_{tag}.pkl'
        model = copy.deepcopy(self)
        # The ThunderSVM model has some internal pointers which can't be pickled. These can be deleted after fitting (I think...)
        del model.model.predict_label_ptr
        del model.model.predict_pro_ptr
        with open(output_dir / model_name, 'wb') as fp:
            pickle.dump(model, fp)

    def serialize(self, working_dir: Path, tag=None):
        """Returns a factory function for recreating this model as well as the state required to do so"""
        tmp_svm_path = working_dir / 'tmp_smv_model.svm'
        self.model.save_to_file(str(tmp_svm_path))
        with open(tmp_svm_path, 'rb') as fp:
            model_bytes = fp.read()
        tmp_svm_path.unlink()
        self_copy = copy.deepcopy(self)
        self_copy.model = model_bytes
        self_bytes = pickle.dumps(self_copy)
        return deserializes, self_bytes

