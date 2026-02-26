"""Classification metrics not included in scikit-learn."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics._classification import _check_targets


def get_classification_labels(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
) -> tuple[int, int, int, int]:
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

    Raises
    ------
    ValueError
        If more than two classes are present for a binary problem.
    """
    _problem_true, y_true, y_pred, _ = _check_targets(y_true, y_pred)
    if len(np.unique(y_true)) > 2:
        raise ValueError("More than two classes present in y_true for a binary classification problem")
    if len(np.unique(y_pred)) > 2:
        raise ValueError("More than two classes present in y_pred for a binary classification problem")
    label_1 = sorted(np.unique(y_true))[1]
    label_0 = sorted(np.unique(y_true))[0]
    true_positive = len(np.where((y_true == label_1) & (y_pred == label_1))[0])
    false_positive = len(np.where((y_true == label_0) & (y_pred == label_1))[0])
    false_negative = len(np.where((y_true == label_1) * (y_pred == label_0))[0])
    true_negative = len(np.where((y_true == label_0) & (y_pred == label_0))[0])
    return true_positive, false_positive, false_negative, true_negative


def specificity_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
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
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The specificity score

    Raises
    ------
    TypeError
        If positive_class is not str or int.
    ValueError
        If positive_class is None for multiclass problem.
    """
    _problem_true, y_true, y_pred, _ = _check_targets(y_true, y_pred)
    if is_binary:
        _tp, fp, _fn, tn = get_classification_labels(y_true, y_pred)
    else:
        if positive_class is not None:
            if isinstance(positive_class, str | int | np.integer):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                _tp, fp, _fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise TypeError("positive_class must be str or int for multiclass problem")
        else:
            raise ValueError("Cannot calculate specificity score with positive_class=None for multiclass problem")
    return tn / (tn + fp)


def average_specificity_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
) -> float:
    """Calculates the average specificity score. Used for when we have more than 2 classes and want our models' average
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
        overall_score = 0.0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += specificity_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def sensitivity_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
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
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The sensitivity score

    Raises
    ------
    TypeError
        If positive_class is not str or int.
    ValueError
        If positive_class is None for multiclass problem.
    """
    _problem_true, y_true, y_pred, _ = _check_targets(y_true, y_pred)
    if is_binary:
        tp, _fp, fn, _tn = get_classification_labels(y_true, y_pred)
    else:
        if positive_class is not None:
            if isinstance(positive_class, str | int | np.integer):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, _fp, fn, _tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise TypeError("positive_class must be str or int for multiclass problem")
        else:
            raise ValueError("Cannot calculate sensitivity score with positive_class=None for multiclass problem")
    return tp / (tp + fn)


def average_sensitivity_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
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
        overall_score = 0.0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += sensitivity_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def power_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
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
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The sensitivity score
    """
    return sensitivity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)


def average_power_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
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
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
) -> float:
    """Also known as Type II error score. Calculates the percentage of true negatives we correctly identified
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
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The negative predictive score

    Raises
    ------
    TypeError
        If positive_class is not str or int.
    ValueError
        If positive_class is None for multiclass problem.
    """
    _problem_true, y_true, y_pred, _ = _check_targets(y_true, y_pred)
    if is_binary:
        _tp, _fp, fn, tn = get_classification_labels(y_true, y_pred)
    else:
        if positive_class is not None:
            if isinstance(positive_class, str | int | np.integer):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                _tp, _fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise TypeError("positive_class must be str or int for multiclass problem")
        else:
            raise ValueError(
                "Cannot calculate negative predictive score with positive_class=None for multiclass problem"
            )
    return tn / (tn + fn)


def average_negative_predictive_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
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
        overall_score = 0.0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += negative_predictive_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def false_negative_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
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
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false negative score

    Raises
    ------
    TypeError
        If positive_class is not str or int.
    ValueError
        If positive_class is None for multiclass problem.
    """
    _problem_true, y_true, y_pred, _ = _check_targets(y_true, y_pred)
    if is_binary:
        tp, _fp, fn, _tn = get_classification_labels(y_true, y_pred)
    else:
        if positive_class is not None:
            if isinstance(positive_class, str | int | np.integer):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, _fp, fn, _tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise TypeError("positive_class must be str or int for multiclass problem")
        else:
            raise ValueError("Cannot calculate false negative score with positive_class=None for multiclass problem")
    return fn / (fn + tp)


def average_false_negative_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
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
        overall_score = 0.0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_negative_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def type_two_error_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
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
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The Type II error score
    """
    return false_negative_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)


def average_type_two_error_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
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
    The average of each Type II error score for each group/class
    """
    return average_false_negative_score(y_true, y_pred)


def false_positive_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
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
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false positive score

    Raises
    ------
    TypeError
        If positive_class is not str or int.
    ValueError
        If positive_class is None for multiclass problem.
    """
    _problem_true, y_true, y_pred, _ = _check_targets(y_true, y_pred)
    if is_binary:
        _tp, fp, _fn, tn = get_classification_labels(y_true, y_pred)
    else:
        if positive_class is not None:
            if isinstance(positive_class, str | int | np.integer):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                _tp, fp, _fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise TypeError("positive_class must be str or int for multiclass problem")
        else:
            raise ValueError("Cannot calculate false positive score with positive_class=None for multiclass problem")
    return fp / (fp + tn)


def average_false_positive_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
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
        overall_score = 0.0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_positive_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def type_one_error_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
) -> float:
    """This is exactly the same as false positive score.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default=True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The Type I error score
    """
    return false_positive_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)


def average_type_one_error_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
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
    The average Type I error score
    """
    return average_false_positive_score(y_true, y_pred)


def false_discovery_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
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
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false discovery score

    Raises
    ------
    TypeError
        If positive_class is not str or int.
    ValueError
        If positive_class is None for multiclass problem.
    """
    _problem_true, y_true, y_pred, _ = _check_targets(y_true, y_pred)
    if is_binary:
        tp, fp, _fn, _tn = get_classification_labels(y_true, y_pred)
    else:
        if positive_class is not None:
            if isinstance(positive_class, str | int | np.integer):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                tp, fp, _fn, _tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise TypeError("positive_class must be str or int for multiclass problem")
        else:
            raise ValueError("Cannot calculate false discovery score with positive_class=None for multiclass problem")
    return fp / (fp + tp)


def average_false_discovery_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
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
        overall_score = 0.0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_discovery_score(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def false_omission_rate(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
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
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The false omission rate

    Raises
    ------
    TypeError
        If positive_class is not str or int.
    ValueError
        If positive_class is None for multiclass problem.
    """
    _problem_true, y_true, y_pred, _ = _check_targets(y_true, y_pred)
    if is_binary:
        _tp, _fp, fn, tn = get_classification_labels(y_true, y_pred)
    else:
        if positive_class is not None:
            if isinstance(positive_class, str | int | np.integer):
                new_y_true = np.where(y_true == positive_class, 1, 0)
                new_y_pred = np.where(y_pred == positive_class, 1, 0)
                _tp, _fp, fn, tn = get_classification_labels(new_y_true, new_y_pred)
            else:
                raise TypeError("positive_class must be str or int for multiclass problem")
        else:
            raise ValueError("Cannot calculate false omission rate with positive_class=None for multiclass problem")
    return fn / (fn + tn)


def average_false_omission_rate(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
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
        overall_score = 0.0
        unique_classes = np.unique(y_true)
        for pos_class in unique_classes:
            overall_score += false_omission_rate(y_true, y_pred, is_binary=False, positive_class=pos_class)
        return overall_score / len(unique_classes)


def j_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
) -> float:
    """Calculate the J-score, or our sensitivity + specificity - 1 (Youden's J statistic).

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The J score
    """
    return (
        sensitivity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
        + specificity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
        - 1
    )


def markedness_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
) -> float:
    """Calculates the markedness score, or the precision + negative predictive score - 1.

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The markedness score
    """

    def precision_score(
        y_true: Sequence[int] | np.ndarray | pd.Series,
        y_pred: Sequence[int] | np.ndarray | pd.Series,
        is_binary: bool = True,
        positive_class: str | int | None = None,
    ) -> float:
        _problem_true, y_true, y_pred, _ = _check_targets(y_true, y_pred)
        if is_binary:
            tp, fp, _fn, _tn = get_classification_labels(y_true, y_pred)
        else:
            if positive_class is not None:
                if isinstance(positive_class, str | int):
                    new_y_true = np.where(y_true == positive_class, 1, 0)
                    new_y_pred = np.where(y_pred == positive_class, 1, 0)
                    tp, fp, _fn, _tn = get_classification_labels(new_y_true, new_y_pred)
                else:
                    raise TypeError("positive_class must be str or int for multiclass problem")
            else:
                raise ValueError("Cannot calculate precision score with positive_class=None for multiclass problem")
        return tp / (tp + fp)

    return (
        precision_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
        + negative_predictive_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
        - 1
    )


def likelihood_ratio_positive(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
) -> float:
    """Calculates the likelihood ratio positive, or sensitivity / (1 - specificity).

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default = True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The positive likelihood ratio
    """
    return sensitivity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class) / (
        1 - specificity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
    )


def likelihood_ratio_negative(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    is_binary: bool = True,
    positive_class: str | int | None = None,
) -> float:
    """Calculates the likelihood ratio negative, or specificity / (1 - sensitivity).

    Parameters
    ----------
    y_true : list or array like
        The true, or the expected, values of our model
    y_pred : list or array like
        The predicted values of our model
    is_binary : bool, default=True
        Whether our problem is a binary classification or a multiclassification problem
    positive_class : int or str, default=None
        If problem=='multiclass' then the class we are denoting as 'success' or 'positive' (i.e., the one marked as a 1).

    Returns
    -------
    The negative likelihood ratio
    """
    return specificity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class) / (
        1 - sensitivity_score(y_true, y_pred, is_binary=is_binary, positive_class=positive_class)
    )
