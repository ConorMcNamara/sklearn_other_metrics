import numpy as np
import math
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.metrics.classification import _check_targets
from sklearn.metrics.regression import _check_reg_targets

# Regression


def adjusted_r2_score(y_true, y_pred, features_vector=None, num_features=None):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput='raw_values')
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
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput='raw_values')
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
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput='raw_values')
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape_score(y_true, y_pred):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput='raw_values')
    error = np.abs(y_true - y_pred)
    total = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.sum(error / total) / len(error)


def root_mean_squared_error(y_true, y_pred):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput='raw_values')
    n = len(y_true)
    return math.sqrt(np.sum((y_true - y_pred)**2) / n)

# Classification


def get_classification_labels(y_true, y_pred):
    type_true, y_true, y_pred = _check_targets(y_true, y_pred)
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


def specificity_score(y_true, y_pred, type='Binary', positive_class=None):
    type_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    elif type.casefold() == 'multiclass':
        if positive_class:
            if isinstance(positive_class, str) or isinstance(positive_class, int):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise Exception("Cannot discern positive class for multiclass problem")
        else:
            raise Exception("Cannot calculate specificity score with None")
    return tn / (tn + fp)


def average_specificity_score(y_true, y_pred):
    if len(np.unique(y_true)) < 3:
        return specificity_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += specificity_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def sensitivity_score(y_true, y_pred, type='Binary', positive_class=None):
    """This is exactly the same as recall"""
    type_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    elif type.casefold() == 'multiclass':
        if positive_class:
            if isinstance(positive_class, str) or isinstance(positive_class, int):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise Exception("Cannot discern positive class for multiclass problem")
        else:
            raise Exception("Cannot calculate sensitivity score with None")
    return tp / (tp + fn)


def average_sensitivity_score(y_true, y_pred):
    if len(np.unique(y_true)) < 3:
        return sensitivity_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += sensitivity_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def power_score(y_true, y_pred, type='Binary', positive_class=None):
    """This is just another way of saying sensitivity"""
    return sensitivity_score(y_true, y_pred, type=type, positive_class=positive_class)


def average_power_score(y_true, y_pred):
    return average_sensitivity_score(y_true, y_pred)


def negative_predictive_score(y_true, y_pred, type='Binary', positive_class=None):
    type_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    elif type.casefold() == 'multiclass':
        if positive_class:
            if isinstance(positive_class, str) or isinstance(positive_class, int):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise Exception("Cannot discern positive class for multiclass problem")
        else:
            raise Exception("Cannot calculate negative predictive score with None")
    return tn / (tn + fn)


def average_negative_predictive_score(y_true, y_pred):
    if len(np.unique(y_true)) < 3:
        return negative_predictive_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += negative_predictive_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def false_negative_score(y_true, y_pred, type='Binary', positive_class=None):
    type_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    elif type.casefold() == 'multiclass':
        if positive_class:
            if isinstance(positive_class, str) or isinstance(positive_class, int):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise Exception("Cannot discern positive class for multiclass problem")
        else:
            raise Exception("Cannot calculate false negative score with None")
    return fn / (fn + tp)


def average_false_negative_score(y_true, y_pred):
    if len(np.unique(y_true)) < 3:
        return false_negative_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_negative_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def type_two_error_score(y_true, y_pred, type='Binary', positive_class=None):
    """This is exactly the same as false negative score """
    return false_negative_score(y_true, y_pred, type=type, positive_class=positive_class)


def average_type_two_error_score(y_true, y_pred):
    return average_false_negative_score(y_true, y_pred)


def false_positive_score(y_true, y_pred, type='Binary', positive_class=None):
    type_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    elif type.casefold() == 'multiclass':
        if positive_class:
            if isinstance(positive_class, str) or isinstance(positive_class, int):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise Exception("Cannot discern positive class for multiclass problem")
        else:
            raise Exception("Cannot calculate false positive score with None")
    return fp / (fp + tn)


def average_false_positive_score(y_true, y_pred):
    if len(np.unique(y_true)) < 3:
        return false_positive_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_positive_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def type_one_error_score(y_true, y_pred, type='Binary', positive_class=None):
    """This is exactly the same as false positive score"""
    return false_positive_score(y_true, y_pred, type=type, positive_class=positive_class)


def average_type_one_error_score(y_true, y_pred):
    return average_false_positive_score(y_true, y_pred)


def false_discovery_score(y_true, y_pred, type='Binary', positive_class=None):
    type_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    elif type.casefold() == 'multiclass':
        if positive_class:
            if isinstance(positive_class, str) or isinstance(positive_class, int):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise Exception("Cannot discern positive class for multiclass problem")
        else:
            raise Exception("Cannot calculate false discovery score with None")
    return fp / (fp + tp)


def average_false_discovery_score(y_true, y_pred):
    if len(np.unique(y_true)) < 3:
        return false_discovery_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_discovery_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def false_omission_rate(y_true, y_pred, type='Binary', positive_class=None):
    type_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if type.casefold() == 'binary':
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    elif type.casefold() == 'multiclass':
        if positive_class:
            if isinstance(positive_class, str) or isinstance(positive_class, int):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise Exception("Cannot discern positive class for multiclass problem")
        else:
            raise Exception("Cannot calculate false omission score with None")
    return fn / (fn + tn)


def average_false_omission_rate(y_true, y_pred):
    if len(np.unique(y_true)) < 3:
        return false_omission_rate(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_omission_rate(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def j_score(y_true, y_pred, type='Binary', positive_class=None):
    return sensitivity_score(y_true, y_pred, type=type, positive_class=positive_class) + \
           specificity_score(y_true, y_pred, type=type, positive_class=positive_class) - 1


def markedness_score(y_true, y_pred, type='Binary', positive_class=None):

    def precision_score(y_true, y_pred, type='Binary', positive_class=None):
        type_true, y_true, y_pred = _check_targets(y_true, y_pred)
        if type.casefold() == 'binary':
            tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
        elif type.casefold() == 'multiclass':
            if positive_class:
                if isinstance(positive_class, str) or isinstance(positive_class, int):
                    new_y_true = np.where(y_true == positive_class, 1, 0)
                    new_y_pred = np.where(y_pred == positive_class, 1, 0)
                    tp, fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
                else:
                    raise Exception("Cannot discern positive class for multiclass problem")
            else:
                raise Exception("Cannot calculate precision score with None")
        return tp / (tp + fp)

    return precision_score(y_true, y_pred, type=type, positive_class=positive_class) + negative_predictive_score(y_true, y_pred, type=type) - 1


def likelihood_ratio_positive(y_true, y_pred, type='Binary', positive_class=None):
    return sensitivity_score(y_true, y_pred, type=type, positive_class=positive_class) / \
           (1 - specificity_score(y_true, y_pred, type=type, positive_class=positive_class))


def likelihood_ratio_negative(y_true, y_pred, type='Binary', positive_class=None):
    return specificity_score(y_true, y_pred, type=type, positive_class=positive_class) / \
           (1 - sensitivity_score(y_true, y_pred, type=type, positive_class=positive_class))
