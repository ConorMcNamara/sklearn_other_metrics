import numpy as np
import math
from sklearn.metrics import r2_score, explained_variance_score

# Regression


def adjusted_r2_score(y_true, y_pred, features_vector=None, num_features=None):
    if len(y_true) != len(y_pred):
        raise Exception("Predictions and actual are not of same length")
    if features_vector:
        if len(features_vector) >= len(y_true) - 1:
            raise Exception("Number of features is greater than number of rows and 1 degree of freedom")
        p = len(features_vector)
        n = len(y_true)
    elif num_features:
        if num_features >= len(y_true) - 1:
            raise Exception("Number of features is greater than number of rows and 1 degree of freedom")
        p = num_features
        n = len(y_true)
    else:
        raise Exception("No features available to calculate adjusted score")
    r2 = r2_score(y_true, y_pred)
    if r2 < 0:
        r2 = 0
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def adjusted_explained_variance_score(y_true, y_pred, features_vector=None, num_features=None):
    if len(y_true) != len(y_pred):
        raise Exception("Predictions and actual are not of same length")
    if features_vector:
        if len(features_vector) >= len(y_true) - 1:
            raise Exception("Number of features is greater than number of rows and 1 degree of freedom")
        p = len(features_vector)
        n = len(y_true)
    elif num_features:
        if num_features >= len(y_true) - 1:
            raise Exception("Number of features is greater than number of rows and 1 degree of freedom")
        p = num_features
        n = len(y_true)
    else:
        raise Exception("No features available to calculate adjusted score")
    evs = explained_variance_score(y_true, y_pred)
    if evs < 0:
        evs = 0
    return 1 - (1 - evs) * (n - 1) / (n - p - 1)


def mape_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise Exception("Predictions and actual are not of same length")
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise Exception("Predictions and actual are not of same length")
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    error = np.abs(y_true - y_pred)
    total = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.sum(error / total) / len(error)


def root_mean_squared_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise Exception("Predictions and actual are not of the same length")
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    return math.sqrt(np.sum((y_true - y_pred)**2) / n)

# Classification


def get_classification_labels(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise Exception("Predictions and actual are not of the same length")
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    true_positive = len(np.where((y_true == 1) & (y_pred == 1))[0])
    false_positive = len(np.where((y_true == 0) & (y_pred == 1))[0])
    false_negative = len(np.where((y_true == 1) * (y_pred == 0))[0])
    true_negative = len(np.where((y_true == 0) & (y_pred == 0))[0])
    return true_positive, false_positive, false_negative, true_negative


def specificity_score(y_true, y_pred, type='Binary'):
    if len(y_true) != len(y_pred):
        raise Exception("Predictions and actual are not of the same length")
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if type.casefold() == 'binary':
        if not ((y_true == 0) | (y_true == 1)).all():
            raise Exception("Binary classes cannot have labels outside 0 or 1")
        elif not ((y_pred == 0) | (y_pred == 1)).all():
            raise Exception("Binary classes cannot have labels outside 0 or 1")
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
        return tn / (tn + fp)
