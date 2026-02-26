"""Regression metrics not included in scikit-learn."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.metrics._regression import _check_reg_targets


def adjusted_r2_score(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
    features_vector: Sequence[str] | np.ndarray | pd.Series | None = None,
    num_features: int | None = None,
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

    Raises
    ------
    ValueError
        If number of features is invalid or no features provided.
    """
    _, y_true, y_pred, _multioutput, _ = _check_reg_targets(
        y_true, y_pred, sample_weight=None, multioutput="raw_values"
    )
    if features_vector is not None:
        if len(features_vector) >= len(y_true) - 1:
            raise ValueError("Number of features is greater than or equal to number of rows minus 1 degree of freedom")
        if len(features_vector) < 1:
            raise ValueError("Cannot have less than one feature")
        p = len(features_vector)
        n = len(y_true)
    elif num_features is not None:
        if num_features >= len(y_true) - 1:
            raise ValueError("Number of features is greater than or equal to number of rows minus 1 degree of freedom")
        if num_features < 1:
            raise ValueError("Cannot have less than one feature")
        p = num_features
        n = len(y_true)
    else:
        raise ValueError("No features available to calculate adjusted score")
    r2 = r2_score(y_true, y_pred) if r2_score(y_true, y_pred) > 0 else 0
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def adjusted_explained_variance_score(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
    features_vector: Sequence[str] | np.ndarray | pd.Series | None = None,
    num_features: int | None = None,
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

    Raises
    ------
    ValueError
        If number of features is invalid or no features provided.
    """
    _, y_true, y_pred, _multioutput, _ = _check_reg_targets(
        y_true, y_pred, sample_weight=None, multioutput="raw_values"
    )
    if features_vector is not None:
        if len(features_vector) >= len(y_true) - 1:
            raise ValueError("Number of features is greater than or equal to number of rows minus 1 degree of freedom")
        if len(features_vector) < 1:
            raise ValueError("Cannot have less than one feature")
        p = len(features_vector)
        n = len(y_true)
    elif num_features is not None:
        if num_features >= len(y_true) - 1:
            raise ValueError("Number of features is greater than or equal to number of rows minus 1 degree of freedom")
        if num_features < 1:
            raise ValueError("Cannot have less than one feature")
        p = num_features
        n = len(y_true)
    else:
        raise ValueError("No features available to calculate adjusted score")
    evs = explained_variance_score(y_true, y_pred) if explained_variance_score(y_true, y_pred) > 0 else 0
    return 1 - (1 - evs) * (n - 1) / (n - p - 1)


def mape_score(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
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

    Raises
    ------
    ZeroDivisionError
        If y_true contains zero values.
    """
    _, y_true, y_pred, _multioutput, _ = _check_reg_targets(
        y_true, y_pred, sample_weight=None, multioutput="raw_values"
    )
    if 0 in y_true:
        raise ZeroDivisionError("Cannot calculate MAPE when y_true contains zero values")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def smape_score(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
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
    _, y_true, y_pred, _multioutput, _ = _check_reg_targets(
        y_true, y_pred, sample_weight=None, multioutput="raw_values"
    )
    error = np.abs(y_true - y_pred)
    total = np.abs(y_true) + np.abs(y_pred)
    return float(100 * np.sum(error / total) / len(error))


def root_mean_squared_error(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
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
    _, y_true, y_pred, _multioutput, _ = _check_reg_targets(
        y_true, y_pred, sample_weight=None, multioutput="raw_values"
    )
    n = len(y_true)
    return math.sqrt(np.sum((y_true - y_pred) ** 2) / n)


def group_mean_log_mae(
    y_true: pd.DataFrame | pd.Series | Sequence[Any] | np.ndarray,
    y_pred: pd.DataFrame | pd.Series | Sequence[Any] | np.ndarray,
    groups: Sequence[Any] | np.ndarray | pd.Series,
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
    _, y_true, y_pred, _multioutput, _ = _check_reg_targets(
        y_true, y_pred, sample_weight=None, multioutput="raw_values"
    )
    y_true = pd.Series([i[0] for i in y_true])
    y_pred = pd.Series([i[0] for i in y_pred])
    maes = (y_true - y_pred).abs().groupby(groups).mean()
    return float(np.log(maes.map(lambda x: max(x, floor))).mean())
