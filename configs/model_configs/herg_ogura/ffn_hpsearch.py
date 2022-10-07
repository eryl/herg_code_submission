from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from riseqsar.models.neural_networks.feedforward_network import FeedForwardNetwork, FeedForwardNetworkConfig, DeepNeuralNetworkDescriptorbasedPredictor
from riseqsar.models.neural_networks.dnn import DNNConfig, DeepNeuralNetwork
from riseqsar.experiment.hyperparameter_optimization import HyperParameterCatergorical, HyperParameterInteger, HyperParameterFunction, HyperParameterLogUniform
from riseqsar.featurizer import FeaturizerConfig
from riseqsar.dataset.featurized_dataset import FeaturizedDatasetConfig
from riseqsar.experiment.minibatch_trainer import MiniBatchTrainerConfig
from riseqsar.models.model_specification import ModelSpecification


### Dataset Config ###
featurizer_config = FeaturizerConfig(method='mordred')
dataset_config = FeaturizedDatasetConfig(featurizer_config=featurizer_config)


### Hyper Parameter Setup ###
learning_rate = HyperParameterLogUniform(name='learning_rate', low=1e-5, high=1e-2)
normalization = HyperParameterCatergorical(name='normalization', choices=[True, False])
residual_connections = HyperParameterCatergorical(name='residual_connections', choices=[True, False])
#n_encoder_layers = HyperParameterInteger(name='n_encoder_layers', low=1, high=20)
n_encoder_layers = HyperParameterCatergorical(name='n_encoder_layers', choices=list(range(2,9)))


# Remember, this function will be executed during HP search, when the total
# number of layers has been fixed. We have to make it a named function for picklings sake
#def set_encoder_layers(trial_or_study):
#    return n_layers_total.get_value(trial_or_study) - 1

#n_encoder_layers = HyperParameterFunction(name='n_encoder_layers', function=set_encoder_layers)

hidden_dim = HyperParameterCatergorical(name='hidden_dim', choices=[64, 128, 256, 512])
dropout_rate = HyperParameterCatergorical(name='dropout_rate', choices=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


encoder_config = FeedForwardNetworkConfig(n_layers=n_encoder_layers,
                                          hidden_dim=hidden_dim,
                                          normalization=normalization,
                                          residual_connections=residual_connections,
                                          dropout_rate=dropout_rate)

decoder_config = FeedForwardNetworkConfig(n_layers=1,
                                          hidden_dim=hidden_dim,
                                          normalization=normalization,
                                          residual_connections=residual_connections,
                                          dropout_rate=dropout_rate)

minibatch_training_config = MiniBatchTrainerConfig(max_epochs=100,
                                                   keep_snapshots='best',
                                                   do_pre_eval=False,
                                                   early_stopping_patience=20)


model_config = DNNConfig(encoder_class=FeedForwardNetwork,
                       decoder_class=FeedForwardNetwork,
                       hidden_dim=hidden_dim,
                       trainer_config=minibatch_training_config,
                       encoder_kwargs=dict(config=encoder_config),
                       decoder_kwargs=dict(config=decoder_config),
                       update_iterations=512,
                       batch_size=512,
                       optim_class=AdamW,
                       scheduler_class=ReduceLROnPlateau,
                       scheduler_kwargs=dict(mode='max', factor=0.1, patience=10, verbose=True),
                       train_encoder=True,
                       optim_kwargs=dict(lr=learning_rate, weight_decay=1e-6),
                       output_gradients=True,
                       device='cuda:0',
                       num_dl_workers=6)

model_specification = ModelSpecification(model_class=DeepNeuralNetworkDescriptorbasedPredictor, model_config=model_config, dataset_config=dataset_config)
