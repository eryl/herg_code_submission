import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Union, Literal, Optional

from tqdm import trange, tqdm
import numpy as np

from riseqsar.experiment.experiment import ExperimentTracker
from riseqsar.evaluation.performance import PerformanceCollection

LATEST_MODEL = 'latest_model'
BEST_MODEL = 'best_model'

@dataclass
class MiniBatchTrainerConfig:
    max_epochs: int = 1
    # Determines what checkpoints to keep. 'all' keeps all checkpoints, 'best' only keeps the all time best. 'frontier' keeps the history of best models.
    keep_snapshots: Literal['all', 'best', 'frontier'] = 'none'
    eval_time: Optional[int] = None
    eval_iterations: Optional[int] = None
    eval_epochs: int = 1
    model_format_string: Optional[str] = None
    do_pre_eval: bool = False
    early_stopping_patience: Optional[int] = None



def minibatch_train(*,
                    model, training_dataset, dev_dataset,
                    experiment_tracker: ExperimentTracker,
                    training_config: MiniBatchTrainerConfig,
                    initial_performance: PerformanceCollection,
                    scheduler=None):
    epoch = 0
    best_model_path = None

    def sigint_handler(signal, frame):
        model_reference = f'{type(model).__name__}_aborted'
        checkpoint(model=model,
                   model_reference=model_reference,
                   experiment_tracker=experiment_tracker,
                   is_best=False, keep_snapshots='all')
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    # Since we call evaluate_models from som many places below, we summarize the common arguments in a dict
    eval_kwargs = dict(model=model,
                       evaluation_dataset=dev_dataset,
                       experiment_tracker=experiment_tracker,
                       keep_snapshots=training_config.keep_snapshots,
                       scheduler=scheduler)

    best_performance = initial_performance
    best_model_reference = None
    if training_config.do_pre_eval:
        best_performance, best_model_reference, _ = evaluate_model(best_performance=best_performance,
                                                                best_model_reference=best_model_reference,
                                                                epoch=0,
                                                                no_improvement_count = 0,
                                                                **eval_kwargs)

    # These variables will be used to control when to do evaluation
    eval_timestamp = time.time()
    eval_epoch = 0
    eval_iteration = 0
    needs_final_eval = True
    no_improvement_count = 0

    for epoch in trange(training_config.max_epochs, desc='Epochs'):
        ## This is the main training loop
        for i, batch in enumerate(tqdm(training_dataset, desc='Training batch')):
            needs_final_eval = True
            epoch_fraction = epoch + i / len(training_dataset)
            training_results = model.fit_minibatch(batch)

            experiment_tracker.log_scalar('epoch', epoch_fraction)
            if training_results is not None:
                experiment_tracker.log_scalars(training_results)

            # eval_time and eval_iterations allow the user to control how often to run evaluations
            eval_time_dt = time.time() - eval_timestamp
            eval_iteration += 1

            if ((training_config.eval_time is not None
                 and training_config.eval_time > 0
                 and eval_time_dt >= training_config.eval_time)
                    or
                (training_config.eval_iterations is not None
                 and training_config.eval_iterations > 0
                 and eval_iteration >= training_config.eval_iterations)):
                best_performance, best_model_reference, no_improvement_count = evaluate_model(best_performance=best_performance,
                                                                        best_model_reference=best_model_reference,
                                                                        epoch=epoch_fraction,
                                                                        no_improvement_count=no_improvement_count,
                                                                        **eval_kwargs)
                eval_timestamp = time.time()
                eval_iteration = 0
                needs_final_eval = False
            experiment_tracker.progress()
            # End of training loop

        eval_epoch += 1
        if (training_config.eval_epochs is not None
                and training_config.eval_epochs > 0
                and eval_epoch >= training_config.eval_epochs):
            best_performance, best_model_reference, no_improvement_count = evaluate_model(best_performance=best_performance,
                                                                                          best_model_reference=best_model_reference,
                                                                                          epoch=epoch,
                                                                                          no_improvement_count=no_improvement_count,
                                                                                          **eval_kwargs)
            eval_epoch = 0
            needs_final_eval = False

        if training_config.early_stopping_patience is not None and no_improvement_count >= training_config.early_stopping_patience:
            print("Eearly stopping")
            break
    # End of epoch

    # Done with the whole training loop. If we ran the evaluate_model at the end of the last epoch, we shouldn't do
    # it again
    if needs_final_eval:
        best_performance, best_model_reference, no_improvement_count = evaluate_model(best_performance=best_performance,
                                                                best_model_reference=best_model_reference,
                                                                epoch=epoch,
                                                                no_improvement_count=no_improvement_count,
                                                                **eval_kwargs)
    return best_performance, best_model_reference


def evaluate_model(*,
                   model,
                   evaluation_dataset,
                   best_performance,
                   best_model_reference,
                   epoch,
                   no_improvement_count,
                   experiment_tracker: ExperimentTracker,
                   scheduler=None,
                   keep_snapshots='best'):
    gathered_evaluation_results = model.evaluate_dataset(evaluation_dataset)
    evaluation_results = dict()

    for k, v in gathered_evaluation_results.items():
        if v:
            try:
                evaluation_results[k] = np.mean(v)
            except TypeError:
                print("Not logging result {}, can't aggregate data type".format(k))

    new_performance = best_performance.update(evaluation_results)
    is_best = new_performance.cmp(best_performance)
    experiment_tracker.log_scalars(evaluation_results)
    if scheduler is not None:
        prime_performance = new_performance.get_performance(new_performance.get_metrics()[0])
        scheduler.step(prime_performance.value)
    string_results = '__'.join([f'{k}:{v}' for k,v in evaluation_results.items()])
    model_reference = f'{type(model).__name__}_epoch:{epoch}_{string_results}'
    checkpoint(model=model,
               model_reference=model_reference,
               experiment_tracker=experiment_tracker,
               is_best=is_best,
               keep_snapshots=keep_snapshots)
    if is_best:
        best_performance = new_performance
        experiment_tracker.log_scalars({f'best_{k}': v for k, v in best_performance.items()})
        best_model_reference = model_reference
        no_improvement_count = 0
    else:
        no_improvement_count += 1
    return best_performance, best_model_reference, no_improvement_count


def checkpoint(*,
               model,
               model_reference,
               experiment_tracker: ExperimentTracker,
               is_best: bool,

               keep_snapshots: Literal['all', 'best', 'frontier'] = 'best'):
    experiment_tracker.log_model(model_reference, model)

    # If we shouldn't keep all snapshots, we have to remove the latest model unless it's also the best
    if keep_snapshots != 'all' and experiment_tracker.reference_exists(LATEST_MODEL):
        latest_model_resolved_reference = experiment_tracker.lookup_reference(LATEST_MODEL)
        if experiment_tracker.reference_exists(BEST_MODEL):
            best_model_resolved_reference = experiment_tracker.lookup_reference(BEST_MODEL)
            if best_model_resolved_reference != latest_model_resolved_reference:
                #Latest model is not the best model, let's remove it
                experiment_tracker.delete_model(latest_model_resolved_reference)
    experiment_tracker.make_reference(LATEST_MODEL, model_reference)

    if is_best:
        if experiment_tracker.reference_exists(BEST_MODEL):
            best_model_resolved_reference = experiment_tracker.lookup_reference(BEST_MODEL)
            if keep_snapshots == 'best':
                # We should only keep the all time best model, so we remove the previous best
                # It might have been removed before, so check whether the reference exists before
                if experiment_tracker.reference_exists(best_model_resolved_reference):
                    experiment_tracker.delete_model(best_model_resolved_reference)
        experiment_tracker.make_reference(BEST_MODEL, model_reference)
