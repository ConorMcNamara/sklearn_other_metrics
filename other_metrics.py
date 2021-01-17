import numpy as np
import math
import pandas as pd
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.metrics._classification import _check_targets
from sklearn.metrics._regression import _check_reg_targets

# Regression


def adjusted_r2_score(y_true, y_pred, features_vector=None, num_features=None):
    """Calculates an adjusted R-squared that penalizes models that use too many superfluous features

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    features_vector: list or array like, default is None
        A list of all features used for our model.
    num_features: int, default is None
        The number of features used for our model. Used if features_vector is None

    Returns
    -------
    Our adjusted R-squared.
    """
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
    """Calculates an adjusted explained_variance_score that penalizes models that use too many superfluous features

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    features_vector: list or array like, default is None
        A list of all features used for our model.
    num_features: int, default is None
        The number of features used for our model. Used if features_vector is None

    Returns
    -------
    Our adjusted explained_variance_score.
    """
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


def mape_score(y_true, y_pred):
    """Calculates the Mean Absolute Percentage Error, a common metric used for Time Series Problems

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    Our MAPE score
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput='raw_values')
    if 0 in y_true:
        raise Exception('Cannot divide by zero')
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape_score(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error. Used when there are zeros in our y_true that would cause
    MAPE to be undefined.

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    Our SMAPE score
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput='raw_values')
    error = np.abs(y_true - y_pred)
    total = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.sum(error / total) / len(error)


def root_mean_squared_error(y_true, y_pred):
    """Calculates the Root Mean Squared Error for regression problems.

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    Our RMSE score
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput='raw_values')
    n = len(y_true)
    return math.sqrt(np.sum((y_true - y_pred)**2) / n)


def group_mean_log_mae(y_true, y_pred, groups, floor=1e-9):
    """Calculates the Group Mean Log Mean Absolute Error. Used in a Kaggle competition.

    Parameters
    ----------
    y_true: pandas DataFrame
        The true, or the expected, values of our problem; along with the group attached
    y_pred: pandas DataFrame
        The predicted values of our problem; along with the group attached
    groups: list or array like
        The
    floor: float, default is 1e-9
        The minimum value our Group Mean Log MAE can be (as 0 is undefined for log transformations).

    Returns
    -------
    Our Group Mean Log MAE score
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput='raw_values')
    y_true = pd.Series([i[0] for i in y_true])
    y_pred = pd.Series([i[0] for i in y_pred])
    maes = (y_true - y_pred).abs().groupby(groups).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()

# Classification


def get_classification_labels(y_true, y_pred):
    """Calculates the true positive, false positive, false negative and true negative values for a classification
    problem.

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The true positive, false positive, false negative and true negative values for our classification problem
    """
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
    """Calculates the specificit of a classification problem

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The specificity score
    """
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
    """Calculates the average specificty score. Used for when we have more than 2 classes and want our models' average
    performance for each class

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The average of each specificity score for each group/class
    """
    if len(np.unique(y_true)) < 3:
        return specificity_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += specificity_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def sensitivity_score(y_true, y_pred, type='Binary', positive_class=None):
    """This is exactly the same as recall

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The sensitivity score
    """
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
    """Calculates the average sensitivity score. Used for when we have more than 2 classes and want our models' average
    performance for each class

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The average of each sensitivity score for each group/class
    """
    if len(np.unique(y_true)) < 3:
        return sensitivity_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += sensitivity_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def power_score(y_true, y_pred, type='Binary', positive_class=None):
    """This is just another way of saying sensitivity

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The sensitivity score
    """
    return sensitivity_score(y_true, y_pred, type=type, positive_class=positive_class)


def average_power_score(y_true, y_pred):
    """This is another way of saying average_sensitivity_score

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The average of each sensitivity score for each group/class
    """
    return average_sensitivity_score(y_true, y_pred)


def negative_predictive_score(y_true, y_pred, type='Binary', positive_class=None):
    """Also known as type II error score. Calculates the percentage of true negatives we correctly identified compared to
    the number of true negative and false negatives.

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The negative predictive score
    """
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
    """Calculates the average negative predictive score. Used for when we have more than 2 classes and want our models'
    average performance for each class

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The average of each negative predictive score for each group/class
    """
    if len(np.unique(y_true)) < 3:
        return negative_predictive_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += negative_predictive_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def false_negative_score(y_true, y_pred, type='Binary', positive_class=None):
    """The inverse of our false positive score, calculates the number of false negatives compared to the number of
    false negatives and true positives.

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false positive score
    """
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
    """Calculates the average false negative score. Used for when we have more than 2 classes and want our models'
    average performance for each class

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The average of each false negative score for each group/class
    """
    if len(np.unique(y_true)) < 3:
        return false_negative_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_negative_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def type_two_error_score(y_true, y_pred, type='Binary', positive_class=None):
    """This is exactly the same as false negative score

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The type II error score
    """
    return false_negative_score(y_true, y_pred, type=type, positive_class=positive_class)


def average_type_two_error_score(y_true, y_pred):
    """This is exactly the same as average false negative score

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The average of each type II error score for each group/class
    """
    return average_false_negative_score(y_true, y_pred)


def false_positive_score(y_true, y_pred, type='Binary', positive_class=None):
    """Calculates the ratio of false positives to false positives and true negatives.

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false positive score
    """
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
    """Calculates the average false positive score. Used for when we have more than 2 classes and want our models'
    average performance for each class

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The average of each false positive score for each group/class
    """
    if len(np.unique(y_true)) < 3:
        return false_positive_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_positive_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def type_one_error_score(y_true, y_pred, type='Binary', positive_class=None):
    """This is exactly the same as false positive score

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The type I error score
    """
    return false_positive_score(y_true, y_pred, type=type, positive_class=positive_class)


def average_type_one_error_score(y_true, y_pred):
    """This is exactly the same as average false positive score

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The average type one error score
    """
    return average_false_positive_score(y_true, y_pred)


def false_discovery_score(y_true, y_pred, type='Binary', positive_class=None):
    """Calculates the ratio of false positives to false positives and true positives

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false discovery score
    """
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
    """Calculates the average false discovery score. Used for when we have more than 2 classes and want our models'
    average performance for each class

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The average false discovery score
    """
    if len(np.unique(y_true)) < 3:
        return false_discovery_score(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_discovery_score(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def false_omission_rate(y_true, y_pred, type='Binary', positive_class=None):
    """Calculates the ratio of false negatives to false negatives and true negatives

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false omission rate
    """
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
    """Calculates the average false omission rate. Used for when we have more than 2 classes and want our models'
    average performance for each class

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem

    Returns
    -------
    The average false omission rate
    """
    if len(np.unique(y_true)) < 3:
        return false_omission_rate(y_true, y_pred)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_omission_rate(y_true, y_pred, type='multiclass', positive_class=pos_class)
        return overall_score / len(unique_classes)


def j_score(y_true, y_pred, type='Binary', positive_class=None):
    """Calculate the j-score, or our sensitivity + specificity - 1

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The j score
    """
    return sensitivity_score(y_true, y_pred, type=type, positive_class=positive_class) + \
           specificity_score(y_true, y_pred, type=type, positive_class=positive_class) - 1


def markedness_score(y_true, y_pred, type='Binary', positive_class=None):
    """Calculates the markedness score, or the precision + negative predictive score - 1

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The markedness score
    """

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
    """Calculates the likehood ratio positive, or sensitivity / (1 - specificity)

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The positive likelihood ratio
    """
    return sensitivity_score(y_true, y_pred, type=type, positive_class=positive_class) / \
           (1 - specificity_score(y_true, y_pred, type=type, positive_class=positive_class))


def likelihood_ratio_negative(y_true, y_pred, type='Binary', positive_class=None):
    """Calculates the likelihood ratio negative, or specificity / (1 - sensitivity)

    Parameters
    ----------
    y_true: list or array like
        The true, or the expected, values of our problem
    y_pred: list or array like
        The predicted values of our problem
    type: str, ['binary', 'multiclass'], default is 'binary'
        Whether our problem is a binary classification or a multiclassification problem
    positive_class: int or str, default is None
        If type=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The negative likelihood ratio
    """
    return specificity_score(y_true, y_pred, type=type, positive_class=positive_class) / \
           (1 - sensitivity_score(y_true, y_pred, type=type, positive_class=positive_class))
