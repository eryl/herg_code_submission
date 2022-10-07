from riseqsar.dataset.constants import TRAIN, DEV, TEST

from riseqsar.evaluation.calculate_performance import calculate_performance
from riseqsar.experiment.experiment_tracker import ExperimentTracker
from riseqsar.experiment.experiment_config import ExperimentConfig


def training_sequence(*,
                      experiment_config: ExperimentConfig,
                      training_dataset,
                      dev_dataset,
                      experiment_tracker: ExperimentTracker,
                      test_dataset=None,
                      save_model: bool = True):
    model_specification = experiment_config.model_specification
    model = model_specification.model_class(config=model_specification.model_config, rng=experiment_config.model_rng)
    
    experiment_tracker.log_artifact('training_dataset_spec', training_dataset.dataset_spec)
    experiment_tracker.log_artifact('dev_dataset_spec', dev_dataset.dataset_spec)
    
    model.fit(train_dataset=training_dataset,
              dev_dataset=dev_dataset,
              experiment_tracker=experiment_tracker,
              evaluation_metrics=experiment_config.evaluation_metrics)

    model.fit_threshold(dev_dataset)
    experiment_tracker.log_scalar('threshold', model.threshold)
    experiment_tracker.log_artifact('model_specification', model_specification)
    experiment_tracker.log_json('model_specification', model_specification)
    
    if save_model:
        experiment_tracker.log_model('final_model', model)
        ## Sanity check: load the final model to make sure serialization works properly
        final_model = experiment_tracker.load_model('final_model')
    else:
        final_model = model
        
    # TODO: This only handles single target tasks, to handle multiple tasks (e.g. Tox21) we need to refactor this
    training_predictions = final_model.predict_dataset_proba(training_dataset).squeeze()
    dev_predictions = final_model.predict_dataset_proba(dev_dataset).squeeze()

    train_performance = calculate_performance(true_class=training_dataset.get_only_targets(),
                                              prediction_scores=training_predictions,
                                              experiment_tracker=experiment_tracker,
                                              dataset_name=training_dataset.get_identifier(),
                                              tag=training_dataset.get_tag(),
                                              threshold=final_model.threshold)
    dev_performance = calculate_performance(true_class=dev_dataset.get_only_targets(),
                                            prediction_scores=dev_predictions,
                                            experiment_tracker=experiment_tracker,
                                            dataset_name=dev_dataset.get_identifier(),
                                            tag=dev_dataset.get_tag(),
                                            threshold=final_model.threshold)
    performance = {TRAIN: train_performance,
                   DEV: dev_performance}

    if test_dataset is not None:
        experiment_tracker.log_artifact('test_dataset_spec', test_dataset.dataset_spec)
        test_predictions = final_model.predict_dataset_proba(test_dataset)
        test_performance = calculate_performance(true_class=test_dataset.get_only_targets(),
                                                 prediction_scores=test_predictions,
                                                 experiment_tracker=experiment_tracker,
                                                 dataset_name=test_dataset.get_identifier(),
                                                 tag=test_dataset.get_tag(),
                                                 threshold=final_model.threshold)
        performance[TEST] = test_performance

    return final_model, performance


