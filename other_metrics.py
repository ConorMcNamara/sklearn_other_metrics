import numpy as np
import math
from sklearn.metrics import r2_score, explained_variance_score, precision_score
from sklearn.utils import check_array

# Regression


def adjusted_r2_score(y_true, y_pred, features_vector=None, num_features=None):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    if features_vector:
        if len(features_vector) >= len(y_true) - 1:
            raise Exception("Number of features is greater than number of rows and 1 degree of freedom")
        if len(features_vector) < 1:
            raise Exception("Cannot have less than one feature")
        p = len(features_vector)
        n = len(y_true)
    elif num_features:
        if num_features >= len(y_true) - 1:
            raise Exception("Number of features is greater than number of rows and 1 degree of freedom")
        if num_features < 1:
            raise Exception("Cannot have less than one feature")
        p = num_features
        n = len(y_true)
    else:
        raise Exception("No features available to calculate adjusted score")
    r2 = r2_score(y_true, y_pred) if r2_score(y_true, y_pred) > 0 else 0
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def adjusted_explained_variance_score(y_true, y_pred, features_vector=None, num_features=None):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    if features_vector:
        if len(features_vector) >= len(y_true) - 1:
            raise Exception("Number of features is greater than number of rows and 1 degree of freedom")
        if len(features_vector) < 1:
            raise Exception("Cannot have less than one feature")
        p = len(features_vector)
        n = len(y_true)
    elif num_features:
        if num_features >= len(y_true) - 1:
            raise Exception("Number of features is greater than number of rows and 1 degree of freedom")
        if num_features < 1:
            raise Exception("Cannot have less than one feature")
        p = num_features
        n = len(y_true)
    else:
        raise Exception("No features available to calculate adjusted score")
    evs = explained_variance_score(y_true, y_pred) if explained_variance_score(y_true, y_pred) > 0 else 0
    return 1 - (1 - evs) * (n - 1) / (n - p - 1)


def mape_error(y_true, y_pred):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape_score(y_true, y_pred):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    error = np.abs(y_true - y_pred)
    total = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.sum(error / total) / len(error)


def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    n = len(y_true)
    return math.sqrt(np.sum((y_true - y_pred)**2) / n)

# Binary Classification


def get_classification_labels(y_true, y_pred):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    if len(np.unique(y_true)) > 2:
        raise Exception("We have more than two classes for a Binary problem")
    if len(np.unique(y_pred)) > 2:
        raise Exception("We have more than two classes for a Binary problem")
    label_1 = sorted(np.unique(y_true))[1]
    label_0 = sorted(np.unique(y_true))[0]
    true_positive = len(np.where((y_true == label_1) & (y_pred == label_1))[0])
    false_positive = len(np.where((y_true == label_0) & (y_pred == label_1))[0])
    false_negative = len(np.where((y_true == label_1) * (y_pred == label_0))[0])
    true_negative = len(np.where((y_true == label_0) & (y_pred == label_0))[0])
    return true_positive, false_positive, false_negative, true_negative


def specificity_score(y_true, y_pred, type='Binary'):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
        return tn / (tn + fp)


def sensitivity_score(y_true, y_pred, type='Binary'):
    """This is exactly the same as recall"""
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
        return tp / (tp + fn)


def power_score(y_true, y_pred, type='Binary'):
    """This is just another way of saying sensitivity"""
    return sensitivity_score(y_true, y_pred, type=type)


def negative_predictive_score(y_true, y_pred, type='Binary'):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
        return tn / (tn + fn)


def false_negative_score(y_true, y_pred, type='Binary'):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
        return fn / (fn + tp)


def type_two_error_score(y_true, y_pred, type='Binary'):
    """This is exactly the same as false negative score """
    return false_negative_score(y_true, y_pred, type=type)


def false_positive_score(y_true, y_pred, type='Binary'):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
        return fp / (fp + tn)


def type_one_error_score(y_true, y_pred, type='Binary'):
    """This is exactly the same as false positive score"""
    return false_positive_score(y_true, y_pred, type=type)


def false_discovery_score(y_true, y_pred, type='Binary'):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
        return fp / (fp + tp)


def false_omission_rate(y_true, y_pred, type='Binary'):
    y_true, y_pred = check_array(y_true), check_array(y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
        return fn / (fn + tn)


def j_score(y_true, y_pred, type='Binary'):
    return sensitivity_score(y_true, y_pred, type=type) + specificity_score(y_true, y_pred, type=type) - 1


def markedness_score(y_true, y_pred, type='Binary'):
    return precision_score(y_true, y_pred) + negative_predictive_score(y_true, y_pred, type=type) - 1


def likelihood_ratio_positive(y_true, y_pred, type='Binary'):
    return sensitivity_score(y_true, y_pred, type=type) / (1 - specificity_score(y_true, y_pred, type=type))


def likelihood_ratio_negative(y_true, y_pred, type='Binary'):
    return specificity_score(y_true, y_pred, type=type) / (1 - sensitivity_score(y_true, y_pred, type=type))
