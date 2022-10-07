from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from riseqsar.models.model_specification import ModelSpecification
from riseqsar.models.neural_networks.feedforward_network import FeedForwardNetwork, FeedForwardNetworkConfig
from riseqsar.models.neural_networks.graph_neural_network import GraphDeepNeuralNetworkConfig, GraphDeepNeuralNetworkPredictor, GraphDeepNeuralNetwork, GNNEncoderConfig, GNNEncoder
from riseqsar.experiment.hyperparameter_optimization import HyperParameterCatergorical, HyperParameterInteger, \
    HyperParameterLogUniform

from riseqsar.featurizer import FeaturizerConfig
from riseqsar.dataset.graph_dataset import MolecularGraphDatasetConfig
from riseqsar.experiment.minibatch_trainer import MiniBatchTrainerConfig


### Hyper Parameter Setup ###
learning_rate = HyperParameterLogUniform(name='learning_rate', low=1e-5, high=1e-2)
normalization = HyperParameterCatergorical(name='normalization', choices=[True, False])
residual_connections = HyperParameterCatergorical(name='residual_connections', choices=[True, False])
#n_encoder_layers = HyperParameterInteger(name='n_encoder_layers', low=1, high=20)
n_encoder_layers = HyperParameterInteger(name='n_encoder_layers', low=2, high=9)

# Remember, this function will be executed during HP search, when the total
# number of layers has been fixed. We have to make it a named function for picklings sake
#def set_encoder_layers(trial_or_study):
#    return n_layers_total.get_value(trial_or_study) - 1

#n_encoder_layers = HyperParameterFunction(name='n_encoder_layers', function=set_encoder_layers)

hidden_dim = HyperParameterCatergorical(name='hidden_dim', choices=[64, 128, 256, 512])
dropout_rate = HyperParameterCatergorical(name='dropout_rate', choices=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# learning_rate = 1e-4
# dropout_rate = 0.1

# hidden_dim = 64
# n_encoder_layers = 4
# normalization = True
# residual_connections = False

embedding_dim = 128

encoder_config = GNNEncoderConfig(num_layers=n_encoder_layers,
                                  d_model=hidden_dim,
                                  ffn_hidden_dim=hidden_dim,
                                  dropout=dropout_rate,
                                  )

decoder_config = FeedForwardNetworkConfig(n_layers=1,
                                          hidden_dim=hidden_dim,
                                          normalization=normalization,
                                          residual_connections=residual_connections,
                                          dropout_rate=dropout_rate)

minibatch_training_config = MiniBatchTrainerConfig(max_epochs=100,
                                                   keep_snapshots='best',
                                                   do_pre_eval=False,
                                                   early_stopping_patience=20)

model_config = GraphDeepNeuralNetworkConfig(encoder_class=GNNEncoder,
                                            decoder_class=FeedForwardNetwork,
                                            hidden_dim=hidden_dim,
                                            trainer_config=minibatch_training_config,
                                            encoder_kwargs=dict(config=encoder_config),
                                            decoder_kwargs=dict(config=decoder_config),
                                            update_iterations=512,
                                            batch_size=128,
                                            optim_class=AdamW,
                                            scheduler_class=ReduceLROnPlateau,
                                            scheduler_kwargs=dict(mode='max', factor=0.1, patience=10, verbose=True),
                                            train_encoder=True,
                                            optim_kwargs=dict(lr=learning_rate, weight_decay=1e-6),
                                            output_gradients=True,
                                            device='cuda:0',
                                            embedding_dim=embedding_dim,
                                            num_dl_workers=8)

dataset_config = MolecularGraphDatasetConfig()

model_specification = ModelSpecification(model_class=GraphDeepNeuralNetworkPredictor, model_config=model_config, dataset_config=dataset_config)
