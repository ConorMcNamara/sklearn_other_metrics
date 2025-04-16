import math

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, explained_variance_score
from sklearn.metrics._classification import _check_targets
from sklearn.metrics._regression import _check_reg_targets


# Regression


def adjusted_r2_score(
    y_true: Union[Sequence[float], np.ndarray, pd.Series],
    y_pred: Union[Sequence[float], np.ndarray, pd.Series],
    features_vector: Optional[Union[Sequence[str], np.ndarray, pd.Series]] = None,
    num_features: Optional[int] = None,
) -> float:
    """Calculates an adjusted R-squared that penalizes models that use too many superfluous features.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem
    features_vector : list or array like, default=None
        A list of all features used for our model.
    num_features : int, default=None
        The number of features used for our model. Used if features_vector is None

    Returns
    -------
    Our adjusted R-squared.
    """
    y_problem, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput="raw_values")
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


def adjusted_explained_variance_score(
    y_true: Union[Sequence[float], np.ndarray, pd.Series],
    y_pred: Union[Sequence[float], np.ndarray, pd.Series],
    features_vector: Optional[Union[Sequence[str], np.ndarray, pd.Series]] = None,
    num_features: Optional[int] = None,
) -> float:
    """Calculates an adjusted explained_variance_score that penalizes models that use too many superfluous features.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem
    features_vector : list or array like, default=None
        A list of all features used for our model.
    num_features : int, default=None
        The number of features used for our model. Used if features_vector is None

    Returns
    -------
    Our adjusted explained_variance_score.
    """
    y_problem, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput="raw_values")
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


def mape_score(
    y_true: Union[Sequence[float], np.ndarray, pd.Series],
    y_pred: Union[Sequence[float], np.ndarray, pd.Series],
) -> float:
    """Calculates the Mean Absolute Percentage Error, a common metric used for Time Series Problems.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem

    Returns
    -------
    Our MAPE score
    """
    y_problem, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput="raw_values")
    if 0 in y_true:
        raise Exception("Cannot divide by zero")
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape_score(
    y_true: Union[Sequence[float], np.ndarray, pd.Series],
    y_pred: Union[Sequence[float], np.ndarray, pd.Series],
) -> float:
    """Calculates the Symmetric Mean Absolute Percentage Error. Used when there are zeros in our y_true that would cause
    MAPE to be undefined.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem

    Returns
    -------
    Our SMAPE score
    """
    y_problem, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput="raw_values")
    error = np.abs(y_true - y_pred)
    total = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.sum(error / total) / len(error)


def root_mean_squared_error(
    y_true: Union[Sequence[float], np.ndarray, pd.Series],
    y_pred: Union[Sequence[float], np.ndarray, pd.Series],
) -> float:
    """Calculates the Root Mean Squared Error for regression problems.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem

    Returns
    -------
    Our RMSE score
    """
    y_problem, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput="raw_values")
    n = len(y_true)
    return math.sqrt(np.sum((y_true - y_pred) ** 2) / n)


def group_mean_log_mae(
    y_true: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray],
    y_pred: Union[pd.DataFrame, pd.Series, Sequence, np.ndarray],
    groups: Union[Sequence, np.ndarray, pd.Series],
    floor: float = 1e-9,
) -> float:
    """Calculates the Group Mean Log Mean Absolute Error. Used in a Kaggle competition.

    Parameters
    ----------
    y_true : list or array-like
        The true, or the expected, values of our problem; along with the group attached
    y_pred : list or array-like
        The predicted values of our problem; along with the group attached
    groups : list or array like
        What our data is being grouped by.
    floor : float, default=1e-9
        The minimum value our Group Mean Log MAE can be (as 0 is undefined for log transformations).

    Returns
    -------
    Our Group Mean Log MAE score
    """
    y_problem, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput="raw_values")
    y_true = pd.Series([i[0] for i in y_true])
    y_pred = pd.Series([i[0] for i in y_pred])
    maes = (y_true - y_pred).abs().groupby(groups).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


# Classification


def get_classification_labels(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> Sequence[int]:
    """Calculates the true positive, false positive, false negative and true negative values for a classification
    problem.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem

    Returns
    -------
    The true positive, false positive, false negative and true negative values for our classification problem
    """
    problem_true, y_true, y_pred = _check_targets(y_true, y_pred)
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


def specificity_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """Calculates the specificity of a classification problem.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default=True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The specificity score
    """
    problem_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if is_binary:
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    else:
        if positive_class:
            if isinstance(positive_class, str) or isinstance(positive_class, int):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise TypeError("Cannot discern positive class for multiclass problem")
        else:
            raise ValueError("Cannot calculate specificity score with None")
    return tn / (tn + fp)


def average_specificity_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> float:
    """Calculates the average specificty score. Used for when we have more than 2 classes and want our models' average
    performance for each class.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
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
            overall_score += specificity_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def sensitivity_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: True,
    positive_class: Union[str, int] = None,
) -> float:
    """This is exactly the same as recall.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default=True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The sensitivity score
    """
    problem_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if is_binary:
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    else:
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


def average_sensitivity_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> float:
    """Calculates the average sensitivity score. Used for when we have more than 2 classes and want our models' average
    performance for each class.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem

    Returns
    -------
    The average of each sensitivity score for each group/class
    """
    if len(np.unique(y_true)) < 3:
        return sensitivity_score(y_true, y_pred, is_binary=True)
    else:
        overall_score = 0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += sensitivity_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def power_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """This is just another way of saying sensitivity.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The sensitivity score
    """
    return sensitivity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)


def average_power_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> float:
    """This is another way of saying average_sensitivity_score.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem

    Returns
    -------
    The average of each sensitivity score for each group/class
    """
    return average_sensitivity_score(y_true, y_pred)


def negative_predictive_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """Also known as problem II error score. Calculates the percentage of true negatives we correctly identified
    compared to the number of true negative and false negatives.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The negative predictive score
    """
    problem_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if is_binary:
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    else:
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


def average_negative_predictive_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> float:
    """Calculates the average negative predictive score. Used for when we have more than 2 classes and want our models'
    average performance for each class.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
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
            overall_score += negative_predictive_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def false_negative_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """The inverse of our false positive score, calculates the number of false negatives compared to the number of false
    negatives and true positives.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false positive score
    """
    problem_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if is_binary:
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    else:
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


def average_false_negative_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> float:
    """Calculates the average false negative score. Used for when we have more than 2 classes and want our models'
    average performance for each class.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
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
            overall_score += false_negative_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def problem_two_error_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """This is exactly the same as false negative score.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The problem II error score
    """
    return false_negative_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)


def average_problem_two_error_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> float:
    """This is exactly the same as average false negative score.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem

    Returns
    -------
    The average of each problem II error score for each group/class
    """
    return average_false_negative_score(y_true, y_pred)


def false_positive_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """Calculates the ratio of false positives to false positives and true negatives.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false positive score
    """
    problem_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if is_binary:
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    else:
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


def average_false_positive_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> float:
    """Calculates the average false positive score. Used for when we have more than 2 classes and want our models'
    average performance for each class.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
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
            overall_score += false_positive_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def problem_one_error_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """This is exactly the same as false positive score.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The problem I error score
    """
    return false_positive_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)


def average_problem_one_error_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> float:
    """This is exactly the same as average false positive score.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
        The predicted values of our problem

    Returns
    -------
    The average problem one error score
    """
    return average_false_positive_score(y_true, y_pred)


def false_discovery_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """Calculates the ratio of false positives to false positives and true positives.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false discovery score
    """
    problem_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if is_binary:
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    else:
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


def average_false_discovery_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> float:
    """Calculates the average false discovery score. Used for when we have more than 2 classes and want our models'
    average performance for each class.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
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
            overall_score += false_discovery_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def false_omission_rate(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """Calculates the ratio of false negatives to false negatives and true negatives.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false omission rate
    """
    problem_true, y_true, y_pred = _check_targets(y_true, y_pred)
    if is_binary:
        tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
    else:
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


def average_false_omission_rate(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> float:
    """Calculates the average false omission rate. Used for when we have more than 2 classes and want our models'
    average performance for each class.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our problem
    y_pred : list or array like
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
            overall_score += false_omission_rate(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def j_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """Calculate the j-score, or our sensitivity + specificity - 1

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The j score
    """
    return (
        sensitivity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
        + specificity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
        - 1
    )


def markedness_score(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary=True,
    positive_class: Union[str, int] = None,
) -> float:
    """Calculates the markedness score, or the precision + negative predictive score - 1

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The markedness score
    """

    def precision_score(
        y_true: Union[Sequence[int], np.ndarray, pd.Series],
        y_pred: Union[Sequence[int], np.ndarray, pd.Series],
        is_binary: bool = True,
        positive_class: Union[str, int] = None,
    ) -> float:
        problem_true, y_true, y_pred = _check_targets(y_true, y_pred)
        if is_binary:
            tp, fp, fn, tn = get_classification_labels(y_true, y_pred)
        else:
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

    return (
        precision_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
        + negative_predictive_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
        - 1
    )


def likelihood_ratio_positive(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """Calculates the likelihood ratio positive, or sensitivity / (1 - specificity)

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The positive likelihood ratio
    """
    return sensitivity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class) / (
        1 - specificity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
    )


def likelihood_ratio_negative(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    is_binary: bool = True,
    positive_class: Union[str, int] = None,
) -> float:
    """Calculates the likelihood ratio negative, or specificity / (1 - sensitivity)

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default=True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'succcess' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The negative likelihood ratio
    """
    return specificity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class) / (
        1 - sensitivity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
    )
