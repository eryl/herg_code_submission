from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from riseqsar.models.neural_networks.recurrent_networks import RecurrentNetwork, RecurrentNetworkConfig
from riseqsar.models.neural_networks.feedforward_network import FeedForwardNetwork, FeedForwardNetworkConfig
from riseqsar.models.neural_networks.sequence_neural_network import SequenceNeuralNetworkConfig, SequenceNeuralNetwork
from riseqsar.models.model_specification import ModelSpecification
from riseqsar.experiment.hyperparameter_optimization import HyperParameterCatergorical, HyperParameterInteger, HyperParameterFunction, HyperParameterLogUniform

from riseqsar.dataset.tokenizer import BytesTokenizer
from riseqsar.experiment.minibatch_trainer import MiniBatchTrainerConfig


### Hyper Parameter Setup ###
learning_rate = HyperParameterLogUniform(name='learning_rate', low=1e-5, high=1e-2)
normalization = HyperParameterCatergorical(name='normalization', choices=[True, False])
residual_connections = HyperParameterCatergorical(name='residual_connections', choices=[True, False])
#bidirectional_rnn = HyperParameterCatergorical(name='bidirectional_rnn', choices=[True, False])
n_layers_total = HyperParameterInteger(name='n_layers_total', low=2, high=8)

# Remember, this function will be executed during HP search, when the total
# number of layers will have been fixed. We have to make it a named function for picklings sake
def set_encoder_layers(trial_or_study):
    return n_layers_total.get_value(trial_or_study) - 1

n_encoder_layers = HyperParameterFunction(name='n_encoder_layers', function=set_encoder_layers)

hidden_dim = HyperParameterCatergorical(name='hidden_dim', choices=[64, 128, 256, 512])

dropout_rate = HyperParameterCatergorical(name='dropout_rate', choices=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

bidirectional_rnn = False

encoder_config = RecurrentNetworkConfig(n_layers=n_encoder_layers,
                                         hidden_dim=hidden_dim,
                                         bidirectional=bidirectional_rnn,
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

dnn_config = SequenceNeuralNetworkConfig(encoder_class=RecurrentNetwork,
                                         decoder_class=FeedForwardNetwork,
                                         tokenizer_class=BytesTokenizer,
                                         hidden_dim=128,
                                         embedding_dim=128,
                                         batch_size=512,
                                         trainer_config=minibatch_training_config,
                                         encoder_kwargs=dict(config=encoder_config),
                                         decoder_kwargs=dict(config=decoder_config),
                                         tokenizer_kwargs=dict(alignment='right', padding_idx=0),
                                         update_iterations=512,
                                         optim_class=AdamW,
                                         scheduler_class=ReduceLROnPlateau,
                                         scheduler_kwargs=dict(mode='max', factor=0.1, patience=10, verbose=True),
                                         train_encoder=True,
                                         optim_kwargs=dict(lr=learning_rate, weight_decay=1e-6),
                                         output_gradients=False,
                                         device='cuda:0',
                                         num_dl_workers=10)

model_specification = ModelSpecification(model_class=SequenceNeuralNetwork, model_config=dnn_config, dataset_config=None)

