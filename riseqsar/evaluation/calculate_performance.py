import csv

import pandas as pd
import os
import math

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve

import numpy as np

from riseqsar.evaluation.constants import *
from riseqsar.experiment.experiment_tracker import ExperimentTracker
from riseqsar.util import listify

def find_threshold(true_class, scores):
    """Find the threshold which optimizes the G-mean of specificity vs. sensitivity using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(true_class, scores)

    # get the best threshold
    J = tpr - fpr
    if J.max() == 0:
        print("No threshold makes TPR better than FPR, setting threshold to mean")
        threshold = np.mean(thresholds)
        return threshold.item()
    else:
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        if best_thresh > scores.max():
            raise ValueError("Best threshold is above any prediction score, this would likely lead to a faulty classifier. Check that the predicted scores are actually different.")
        return best_thresh.item()
    # n = 100
    # sort_index = np.argsort(scores)
    # sorted_logits = scores[sort_index]
    # n_points = min(len(sorted_logits), n)
    # indices_per_point = len(sorted_logits) / n_points
    # indices = (np.arange(n_points) * indices_per_point).astype(np.int)
    # quantiles = sorted_logits[indices]
    #
    # tprs = []
    # fprs = []
    # # predictions n x len(logits)
    # predictions = quantiles[:, np.newaxis] < scores[np.newaxis, :]
    # #correct_predictions = predictions == labels[np.newaxis, :]
    #
    # tps = np.sum((predictions == 1) & (true_class == 1), axis=1)
    # tns = np.sum((predictions == 0) & (true_class == 0), axis=1)
    # ps = np.sum(true_class == 1)
    # ns = np.sum(true_class == 0)
    # recall = tps/ps
    # specificity = tns/ns
    # balanced_accuracies = (recall+specificity)/2
    # best_quantile = quantiles[np.argmax(balanced_accuracies)]




def calculate_performance(*, true_class,
                          prediction_scores,
                          experiment_tracker: ExperimentTracker,
                          dataset_name,
                          tag=None,
                          pred_class=None,
                          threshold=0.5):
    my_output = dict()

    my_output[ROC_AUC] = roc_auc_score(true_class, prediction_scores)

    # calculate performance
    if pred_class is None:
        print(f"No class prediction provided, using threshold of {threshold}")
        pred_class = (prediction_scores > threshold).astype(int)

    my_output['prediction_threshold'] = threshold
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(true_class, pred_class).ravel()
    true_negative, false_positive, false_negative, true_positive = int(true_negative), int(false_positive), int(false_negative), int(true_positive)

    my_output[TRUE_NEGATIVE] = true_negative
    my_output[FALSE_POSITIVE] = false_positive
    my_output[FALSE_NEGATIVE] = false_negative
    my_output[TRUE_POSITIVE] = true_positive

    deno_1 = true_positive+false_negative
    sensitivity = true_positive/deno_1 if deno_1 != 0 else 0
    deno_2 = true_negative+false_positive
    specificity = true_negative/deno_2 if deno_2 != 0 else 0

    my_output[SENSITITIVTY] = sensitivity
    my_output[SPECIFICITY] = specificity

    # BA_train = (sensitivity + specificity)/2
    my_output[BALANCED_ACCURACY] = balanced_accuracy_score(true_class, pred_class)
    my_output[ACCURACY] = accuracy_score(true_class, pred_class)

    #precision = TP/(TP+FP)
    my_output[PRECISION] = precision_score(true_class, pred_class)

    my_output[DISCRETIZED_ROC_AUC] = roc_auc_score(true_class, pred_class)

    mcc_score = matthews_corrcoef(true_class, pred_class)
    my_output[MATTHEWS_CORRCOEF] = mcc_score

    #f1 = format(2*precision*sensitivity/(precision + sensitivity))
    f1 = f1_score(true_class, pred_class)
    my_output[F1] = f1

    kappa_val = cohen_kappa_score(true_class, pred_class)
    my_output[COHEN_KAPPA] = kappa_val

    # positive predictive value
    try:
        PPV = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        PPV = float('nan')

    # Negative predictive value
    try:
        NPV = true_negative / (true_negative + false_negative)
    except ZeroDivisionError:
        NPV = float('nan')

    my_output[POSITIVE_PREDICTIVE_VALUE] = PPV
    my_output[NEGATIVE_PREDICITVE_VALUE] = NPV

    # Fall out or false positive rate
    try:
        FPR = false_positive / (false_positive + true_negative)
    except ZeroDivisionError:
        FPR = float('nan')

    # False negative rate
    try:
        FNR = false_negative / (true_positive + false_negative)
    except ZeroDivisionError:
        FNR = float('nan')

    # False discovery rate
    try:
        FDR = false_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        FDR = float('nan')

    my_output[FALSE_POSITIVE_RATE] = FPR
    my_output[FALSE_NEGATIVE_RATE] = FNR
    my_output[FALSE_DISCOVERY_RATE] = FDR

    # Positive likelihood ratio
    try:
        PLR = sensitivity/(1-specificity)
    except ZeroDivisionError:
        PLR = float('nan')
    # Negative likelihood ratio
    try:
        NLR = (1-sensitivity)/specificity
    except ZeroDivisionError:
        NLR = float('nan')

    # Diagnostic odds ratio
    try:
        DOR = PLR/NLR
    except ZeroDivisionError:
        DOR = float('nan')

    my_output[POSITIVE_LIKELIHOOD_RATIO] = PLR
    my_output[NEGATIVE_LIKELIHOOD_RATIO] = NLR
    my_output[DIAGNOSTIC_ODDS_RATIO] = DOR

    predictions = dict(true_class=listify(true_class),
                       prediction_scores=listify(prediction_scores),
                       predicted_class=listify(pred_class))

    experiment_tracker.log_performance(dataset_name, my_output, tag=tag)
    experiment_tracker.log_predictions(dataset_name, predictions, tag=tag)

    return my_output


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test predictions')
    parser.add_argument('predictions',
                        help="Path to file containing the predictions. This should be a csv with two colums, "
                             "the first the true label, the second the predictive scores")
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions).values
    true_class = predictions[:,0]
    logits = predictions[:,1]
    threshold = find_threshold(true_class, logits)
    calculate_performance(predictions[:,0], predictions[:,1], '/tmp/test_results.csv', threshold=threshold)

if __name__ == '__main__':
    main()