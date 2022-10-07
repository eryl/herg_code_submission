import numpy as np

from riseqsar.evaluation.constants import *

class EvaluationMetric(object):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, EvaluationMetric):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            raise NotImplementedError(f"Equality with {other.__class__} has not been implemented")

    def __str__(self):
        return self.name

    def cmp(self, a, b):
        raise NotImplementedError()


class HigherIsBetterMetric(EvaluationMetric):
    def __init__(self, name):
        super().__init__(name)
        self.worst_value = -np.inf

    def cmp(self, a, b):
        return a > b

    def __repr__(self):
        return f'HigherIsBetterMetric({self.name})'


class LowerIsBetterMetric(EvaluationMetric):
    def __init__(self, name):
        super().__init__(name)
        self.worst_value = np.inf

    def cmp(self, a, b):
        return a < b

    def __repr__(self):
        return f'LowerIsBetterMetric({self.name})'



class Performance(object):
    def __init__(self, metric, value=None):
        self.metric = metric
        if value is None:
            value = metric.worst_value
        self.value = value

    def cmp(self, other):
        return self.metric.cmp(self.value, other.value)

    def __str__(self):
        return str(self.value)


class PerformanceCollection(object):
    def __init__(self, performances):
        self.metrics = [p.metric for p in performances]  # Keeps the order of the performance objects
        self.performances = {p.metric: p for p in performances}

    def cmp(self, other):
        for metric in self.metrics:
            performance = self.performances[metric]
            other_performance = other.get_performance(metric)
            if performance.cmp(other_performance):
                # This performance is better than the other
                return True
            elif other_performance.cmp(performance):
                # This performance is worse than the other
                return False
            else:
                # This performance is equal to the other, we need to look at the next metric
                continue
        return True  # If two performance collections are exactly the same, we return this one as the better

    def get_performance(self, metric):
        return self.performances[metric]

    def update(self, value_map):
        """
        Return a new PerformanceCollection where the metrics are updated according to the
        values in the value map
        :param value_map:
        :return:
        """

        performances = [Performance(metric, value_map[metric.name]) for metric in self.metrics]
        return PerformanceCollection(performances)

    def get_metrics(self):
        return self.metrics

    def __str__(self):
        return '_'.join('{}:{}'.format(metric.name, self.performances[metric].value) for metric in self.metrics)

    def items(self):
        yield from self.performances.items()

    def __getitem__(self, metric):
        return self.performances[metric]


def setup_performance(evaluation_metrics):
    base_performances = []
    for evaluation_metric in evaluation_metrics:
        if isinstance(evaluation_metric, str):
            if 'loss' in evaluation_metric:
                evaluation_metric = LowerIsBetterMetric(evaluation_metric)
            elif 'accuracy' in evaluation_metric:
                evaluation_metric = HigherIsBetterMetric(evaluation_metric)
            else:
                raise RuntimeError(
                    "Metric {} is not implemented, please supply an EvaluationMetric object instead.".format(evaluation_metric))
        if isinstance(evaluation_metric, EvaluationMetric):
            base_performance = Performance(evaluation_metric)
            base_performances.append(base_performance)
    initial_performance = PerformanceCollection(base_performances)
    return initial_performance

# DISCRETIZED_ROC_AUC = 'ROC_ACU_score_discretized'
# MATTHEWS_CORRCOEF = 'MCC'
# COHEN_KAPPA = 'cohen_kappa'
# POSITIVE_PREDICTIVE_VALUE = 'PPV_score'
# NEGATIVE_PREDICITVE_VALUE = 'NPV_score'
# FALSE_POSITIVE_RATE = 'FPR_score'
# FALSE_NEGATIVE_RATE = 'FNR_score'
# FALSE_DISCOVERY_RATE = 'FDR_score'
# POSITIVE_LIKELIHOOD_RATIO = 'PLR_score'
# NEGATIVE_LIKELIHOOD_RATIO = 'NLR_score'
# DIAGNOSTIC_ODDS_RATIO = 'DOR_score'
                                                    
metric_roc_auc = HigherIsBetterMetric(ROC_AUC)
metric_ba = HigherIsBetterMetric(BALANCED_ACCURACY)
metric_acc = HigherIsBetterMetric(ACCURACY)
metric_sensitivity = HigherIsBetterMetric(SENSITITIVTY)
metric_specificity = HigherIsBetterMetric(SPECIFICITY)
metric_precision = HigherIsBetterMetric(PRECISION)
metric_f1 = HigherIsBetterMetric(F1)
