from copy import deepcopy
import gc
from dataclasses import dataclass, field
from typing import Union, List, Optional, Sequence, Mapping
from collections import Counter
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from riseqsar.models.molecular_predictor import MolecularPredictor
from riseqsar.dataset.molecular_dataset import MolecularDataset
from riseqsar.dataset.constants import TRAIN, DEV, TEST
from riseqsar.dataset.featurized_dataset import FeaturizedDataset, FeaturizedDatasetConfig
from riseqsar.experiment.experiment import ExperimentTracker
from riseqsar.experiment.minibatch_trainer import minibatch_train, MiniBatchTrainerConfig
from riseqsar.evaluation.performance import EvaluationMetric, Performance, setup_performance
from riseqsar.evaluation.constants import ROC_AUC, BCE

@dataclass
class DataloaderConfig:
    shuffle: bool = None
    pin_memory: bool = True
    drop_last: bool = False
    timeout: float = 1
    persistent_workers: bool = True


class DNNConfig:
    def __init__(self, *,
                 trainer_config: MiniBatchTrainerConfig,
                 encoder_class: type,
                 decoder_class: type,
                 optim_class: type,
                 # The dimensionality of the interface between encoder and decoder. Will be used as output_dim for the encoder,
                 # and input dim for the decoder,
                 hidden_dim: int,
                 batch_size: int,
                 weighted_sampler: bool = True,
                 # Number of data loader workers
                 num_dl_workers: int = 0,
                 scheduler_class: Optional[type] = None,
                 train_encoder: bool = True,
                 output_param_norms: bool = False,
                 output_gradients: bool = False,
                 gradient_clipping: Optional[float] = None,
                 update_iterations: int = 1,
                 encoder_args: Sequence = None,
                 encoder_kwargs: Mapping = None,
                 decoder_args: Sequence = None,
                 decoder_kwargs: Mapping = None,
                 optim_args: Sequence = None,
                 optim_kwargs: Mapping = None,
                 scheduler_args: Sequence = None,
                 scheduler_kwargs: Mapping = None,
                 device: str = 'cpu',
                 train_dataloader_config: DataloaderConfig = None,
                 dev_dataloader_config: DataloaderConfig = None,
                 test_dataloader_config: DataloaderConfig = None
                 ):
        if encoder_args is None:
            encoder_args = tuple()
        if encoder_kwargs is None:
            encoder_kwargs = dict()
        if decoder_args is None:
            decoder_args = tuple()
        if decoder_kwargs is None:
            decoder_kwargs = dict()
        if optim_args is None:
            optim_args = tuple()
        if optim_kwargs is None:
            optim_kwargs = dict()
        if scheduler_args is None:
            scheduler_args = tuple()
        if scheduler_kwargs is None:
            scheduler_kwargs = dict()
        if train_dataloader_config is None:
            train_dataloader_config = DataloaderConfig(shuffle=True, drop_last=False)
        if dev_dataloader_config is None:
            dev_dataloader_config = DataloaderConfig(shuffle=False, drop_last=False)
        if test_dataloader_config is None:
            test_dataloader_config = DataloaderConfig(shuffle=False, drop_last=False)
        
        self.trainer_config = trainer_config
        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.optim_class = optim_class
        # The dimensionality of the interface between encoder and decoder. Will be used as output_dim for the encoder, 
        # and input dim for the decoder, 
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.weighted_sampler = weighted_sampler
        self.num_dl_workers = num_dl_workers
        self.scheduler_class = scheduler_class
        self.train_encoder = train_encoder
        self.output_param_norms = output_param_norms
        self.output_gradients = output_gradients
        self.gradient_clipping = gradient_clipping
        self.update_iterations = update_iterations
        self.encoder_args = encoder_args
        self.encoder_kwargs = encoder_kwargs
        self.decoder_args = decoder_args
        self.decoder_kwargs = decoder_kwargs
        self.optim_args = optim_args
        self.optim_kwargs = optim_kwargs
        self.scheduler_args = scheduler_args
        self.scheduler_kwargs = scheduler_kwargs
        self.device = device
        self.train_dataloader_config = train_dataloader_config
        self.dev_dataloader_config = dev_dataloader_config
        self.test_dataloader_config = test_dataloader_config


class DeepNeuralNetwork(MolecularPredictor):
    def __init__(self, *args, config: DNNConfig, encoder=None, decoder=None, **kwargs):
        super(DeepNeuralNetwork, self).__init__(*args, config=config, **kwargs)

        # Note that the network is not actually initialized here. The reason is that we often like to 
        # create the network conditioned on the dataset (e.g. set the input dimension based on the dataset, 
        # not hardcode it to the instantiation of the class)
        # To initialize the network, call the method initialize_network()

        self.config = config
        self.device = torch.device(self.config.device)
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_random_state = self.rng.integers(0, 2 ** 32 - 1)
        self.decoder_random_state = self.rng.integers(0, 2 ** 32 - 1)
        torch.manual_seed(self.rng.integers(0, 2**32-1))
        self.initialization_params = dict(threshold=None)

    def set_device(self, device: str):
        device = torch.device(device)
        self.device = device
        self.model.to(self.device)

    def fit_threshold(self, dataset: MolecularDataset):
        threshold = super().fit_threshold(dataset)
        self.initialization_params['threshold'] = threshold

    def setup_initialization_params(self, train_dataset):
        pass

    def initialize_network(self):
        self.model.to(self.device)

        params = []
        if self.config.train_encoder:
            params.extend(self.encoder.parameters())
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False

        params.extend(self.decoder.parameters())
        self.params = params

    def setup_dataloader(self, *, dataset: MolecularDataset, is_training: bool): 
        if is_training:
            if self.config.weighted_sampler:
                samples_weight = dataset.get_samples_weights()
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

                dataloader = DataLoader(dataset,
                                        batch_size=self.config.batch_size,
                                        sampler=sampler,
                                        drop_last=False,
                                        num_workers=self.config.num_dl_workers,
                                        pin_memory=True)
                return dataloader

        dataloader = DataLoader(dataset,
                                batch_size=self.config.batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=self.config.num_dl_workers,
                                pin_memory=True)
        return dataloader


    def fit(self, *, train_dataset: FeaturizedDataset, evaluation_metrics: List[EvaluationMetric], dev_dataset=None, experiment_tracker: ExperimentTracker=None):
        initial_performance = setup_performance(evaluation_metrics)

        # The reason the initialization of the network is delayed to here is that we would like to 
        # be able to create it conditioned on the dataset just like the scikit learn models
        self.setup_initialization_params(train_dataset=train_dataset)
        self.initialize_network()

        # TODO: We should extend the framework to decide loss function based on target dynamically
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.optimizer = self.config.optim_class(self.params, *self.config.optim_args, **self.config.optim_kwargs)

        if self.config.scheduler_class is not None:
            self.scheduler = self.config.scheduler_class(self.optimizer,
                                                         *self.config.scheduler_args,
                                                         **self.config.scheduler_kwargs)
        else:
            self.scheduler = None
        self.updates = 0
        self.update_batch_loss = []
        
        #self.autotune_batch_size(initial_performance=initial_performance, experiment_tracker=experiment_tracker, train_dataset=train_dataset, dev_dataset=dev_dataset)

        train_dataloader = self.setup_dataloader(dataset=train_dataset, is_training=True)
        dev_dataloader = self.setup_dataloader(dataset=dev_dataset, is_training=False)

        best_performance, best_model_reference = minibatch_train(model=self,
                                                                 training_dataset=train_dataloader,
                                                                 dev_dataset=dev_dataloader,
                                                                 experiment_tracker=experiment_tracker,
                                                                 training_config=self.config.trainer_config,
                                                                 scheduler=self.scheduler,
                                                                 initial_performance=initial_performance)
        # Since the experiment tracker will serialize _this_ model (self), we pick out the model attribute of it since
        # that holds all the important learnt stuff
        self.model = experiment_tracker.load_model(best_model_reference).model

    def autotune_batch_size(self, *, initial_performance, train_dataset: MolecularDataset, dev_dataset: MolecularDataset, experiment_tracker: ExperimentTracker):
        # We make a special starting model
        initial_batch_size = self.config.batch_size
        experiment_tracker.log_model('initial_model', self)
        current_batch_size = self.config.batch_size # First try the batch size set by the config
        training_config = deepcopy(self.config.trainer_config)

        while current_batch_size > 1:
            try:
                #garbage_collection_cuda()
                perf = deepcopy(initial_performance)
                self.model = experiment_tracker.load_model('initial_model').model
                
                self.config.batch_size = current_batch_size
                train_dataloader = self.setup_dataloader(dataset=train_dataset, is_training=True)
                dev_dataloader = self.setup_dataloader(dataset=dev_dataset, is_training=False)
                training_config.max_epochs = 1
                training_config.keep_snapshots = 'none'
            
            
                minibatch_train(model=self,
                                training_dataset=train_dataloader,
                                dev_dataset=dev_dataloader,
                                experiment_tracker=experiment_tracker,
                                training_config=training_config,
                                scheduler=self.scheduler,
                                initial_performance=perf)
                break

            except RuntimeError as exception:
                # Only these errors should trigger an adjustment
                if is_oom_error(exception):
                    print(f"batch size {current_batch_size} caused a memory error, decreasing to {current_batch_size//2}")
                    current_batch_size = current_batch_size // 2
                    garbage_collection_cuda()
                    continue
                else:
                    raise  # some other error not memory related
                    
        print(f"Setting batch size to {current_batch_size}")
        self.config.batch_size = current_batch_size
        self.model = experiment_tracker.load_model('initial_model').model


    def predict(self, smiles: str):
        raise NotImplementedError()

    def predict_on_batch(self, batch):
        raise NotImplementedError()

    def loss_on_batch(self, batch):
        raise NotImplementedError()

    def fit_minibatch(self, batch):
        self.model.train(True)

        # Whether y is non-null or not. The labels are encoded with values -1, 0, 1. 0 Indicates a missing value,
        # while -1 and 1 indicate the values of the binary label
        # loss_mask = torch.nonzero(y, as_tuple=True)
        # y = (y + 1) / 2  # Rescale so that the binary labels are 0 and 1
        # Loss matrix
        # loss_mat = self.loss(pred[loss_mask], y[loss_mask])

        loss, y, pred = self.loss_on_batch(batch)

        if self.config.update_iterations is not None and self.updates == 0:
            self.optimizer.zero_grad()

        self.updates += len(pred)


        loss.backward()
        return_dict = {}
        for i, group in enumerate(self.optimizer.param_groups):
            lr = group['lr']
            try:
                lr = lr[0]
            except IndexError:
                pass
            except TypeError:
                pass
            return_dict[f'learning_rate_{i:02}'] = lr
        return_dict['training_bce'] = loss.detach().cpu().numpy()
        self.update_batch_loss.append(loss.detach().cpu().item())

        if self.config.update_iterations is None or self.updates >= self.config.update_iterations:
            if self.config.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)

            self.optimizer.step()
            self.updates = 0
            update_loss = np.mean(self.update_batch_loss)
            self.update_batch_loss.clear()
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step_minibatch'):
                    self.scheduler.step_minibatch(update_loss)
                # return_dict['learning_rate'] = self.scheduler.get_last_lr()

            # If we're accumulating gradients, it only makes sense to output them when we're doing the step
            # (otherwise we'll have confusing sequences of cumulating gradients)
            if self.config.output_gradients:
                grad_norms = dict()
                for i, parameter in enumerate(self.model.parameters()):
                    grad = parameter.grad
                    gnorm = None
                    if grad is not None:
                        gnorm = grad.norm().item()
                    grad_norms[f'{i}_{parameter.name}'] = gnorm
                return_dict['gradient_norms'] = grad_norms
            if self.config.output_param_norms:
                param_norms = {f'{i}_{parameter.name}': parameter.detach().norm().cpu().item() for i, parameter in
                               enumerate(self.model.parameters())}
                return_dict['parameter_norms'] = param_norms

        return return_dict

    def evaluate_dataset(self, dataset):
        self.model.eval()
        with torch.no_grad():
            losses = []
            y_true = []
            y_scores = []
            metadata = []

            for batch in tqdm(dataset, desc='Evaluating dataset', total=len(dataset)):
                loss, y, pred = self.loss_on_batch(batch)

                y_true.append(y.detach().cpu().numpy())
                y_scores.append(pred.detach().cpu().numpy())

                # Whether y is non-null or not. The labels are encoded with values -1, 0, 1. 0 Indicates a missing value,
                # while -1 and 1 indicate the values of the binary label
                # loss_mask = torch.nonzero(y, as_tuple=True)
                # y = (y + 1) / 2  # Rescale so that the binary labels are 0 and 1
                loss_mat = self.loss(pred, y)
                # loss matrix after removing null target
                # loss = loss_mat[loss_mask].mean()
                loss = loss_mat.mean()
                losses.append(loss.detach().cpu().numpy())

            y_true = np.concatenate(y_true, axis=0)
            y_scores = np.concatenate(y_scores, axis=0)

            roc_list = []
            for i in range(y_true.shape[1]):
                # AUC is only defined when there is at least one positive data.
                # if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                #    is_valid = y_true[:, i] ** 2 > 0
                # roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
                roc_list.append(roc_auc_score(y_true[:, i], y_scores[:, i]))

            if len(roc_list) < y_true.shape[1]:
                print("Some target is missing!")
                print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

            auc = np.mean(roc_list)
            return {ROC_AUC: auc,
                    BCE: np.mean(losses),
                    #'logits': y_scores,
                    #'y': y_true,
                    #'metadata': metadata,
            }

    def predict_dataset_proba(self, dataset: MolecularDataset):
        self.model.eval()
        with torch.no_grad():
            dataloader = self.setup_dataloader(dataset=dataset, is_training=False)
            batch_predictions = [self.predict_on_batch(batch).detach().cpu() for batch in dataloader]
            dataset_predictions = torch.cat(batch_predictions, dim=0)
            dataset_probas = torch.sigmoid(dataset_predictions)
            return dataset_probas.detach().cpu().numpy()

    

    def serialize(self, working_dir, tag=None):
        # We want to handle things a bit differently. Essentially first 
        # create the regular torch object, but also save away the config 
        # used to create the networks
        return super().serialize(working_dir, tag)


    def save(self, output_dir: Path, tag=None):
        if tag is None:
            model_name = f'{self.__class__.__name__}.pkl'
        else:
            model_name = f'{self.__class__.__name__}_{tag}.pkl'
        with open(output_dir / model_name, 'wb') as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, load_dir, tag=None):
        if tag is None:
            model_name = f'{cls.__name__}.pkl'
        else:
            model_name = f'{cls.__name__}_{tag}.pkl'
        with open(load_dir / model_name, 'rb') as fp:
            model = pickle.load(fp)
        return model

    def serialize(self, working_dir, tag=None):
        """Returns a factory function for recreating this model as well as the state required to do so"""
        from io import BytesIO
        model_bytes = BytesIO()
        torch.save(self.model.state_dict(), model_bytes)
        model_factory = load_dnn
        model_data = pickle.dumps(dict(model_class=type(self), 
                                       model_config=self.config, 
                                       model_bytes=model_bytes.getvalue(),
                                       initialization_params=self.initialization_params))
        return model_factory, model_data


def load_dnn(model_bytes) -> DeepNeuralNetwork:
    # Factory function for loading a DeepNeuralNetowrk
    from io import BytesIO
        
    model_data = pickle.loads(model_bytes)
    model_class = model_data['model_class']
    model_config = model_data['model_config']
    initialization_params = model_data['initialization_params']
    model = model_class(config=model_config)
    model.initialization_params.update(initialization_params)
    model.initialize_network()
    model.threshold = initialization_params['threshold']

    model_bytes = torch.load(BytesIO(model_data['model_bytes']))  
    model.model.load_state_dict(model_bytes)
    
    return model


def is_oom_error(exception: BaseException) -> bool:
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    oom_error = (isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0])
    illegal_memory_access = (isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA error: an illegal memory access was encountered" in exception.args[0])
    return oom_error or illegal_memory_access


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    cudnn_not_supported = (isinstance(exception, RuntimeError) 
              and len(exception.args) == 1
              and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0])
    cudnn_failed = (isinstance(exception, RuntimeError) 
              and len(exception.args) == 1
              and "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED" in exception.args[0])
    return (cudnn_not_supported or cudnn_failed)


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def garbage_collection_cuda() -> None:
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    try:
        # This is the last thing that should cause an OOM error, but seemingly it can.
        torch.cuda.empty_cache()
    except RuntimeError as exception:
        if not is_oom_error(exception):
            # Only handle OOM errors
            raise