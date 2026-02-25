"""Additional regression and classification metrics for scikit-learn.

This package provides metrics that are commonly used in machine learning but are not
included in scikit-learn's standard metrics module, such as:

- Regression: MAPE, SMAPE, adjusted R², RMSE, Group Mean Log MAE
- Classification: Specificity, Sensitivity, NPV, FDR, FOR, Likelihood Ratios, and more

Examples
--------
>>> from sklearn_other_metrics import mape_score, specificity_score
>>> y_true = [1, 2, 3, 4]
>>> y_pred = [1.1, 2.2, 2.9, 4.1]
>>> mape_score(y_true, y_pred)
5.208333333333333
"""

from sklearn_other_metrics._classification import (
    average_false_discovery_score,
    average_false_negative_score,
    average_false_omission_rate,
    average_false_positive_score,
    average_negative_predictive_score,
    average_power_score,
    average_sensitivity_score,
    average_specificity_score,
    average_type_one_error_score,
    average_type_two_error_score,
    false_discovery_score,
    false_negative_score,
    false_omission_rate,
    false_positive_score,
    get_classification_labels,
    j_score,
    likelihood_ratio_negative,
    likelihood_ratio_positive,
    markedness_score,
    negative_predictive_score,
    power_score,
    sensitivity_score,
    specificity_score,
    type_one_error_score,
    type_two_error_score,
)
from sklearn_other_metrics._regression import (
    adjusted_explained_variance_score,
    adjusted_r2_score,
    group_mean_log_mae,
    mape_score,
    root_mean_squared_error,
    smape_score,
)

__version__ = "0.1.0"

__all__ = [
    # Regression metrics
    "adjusted_r2_score",
    "adjusted_explained_variance_score",
    "mape_score",
    "smape_score",
    "root_mean_squared_error",
    "group_mean_log_mae",
    # Classification helper
    "get_classification_labels",
    # Binary classification metrics
    "specificity_score",
    "sensitivity_score",
    "power_score",
    "negative_predictive_score",
    "false_negative_score",
    "false_positive_score",
    "false_discovery_score",
    "false_omission_rate",
    "type_one_error_score",
    "type_two_error_score",
    "j_score",
    "markedness_score",
    "likelihood_ratio_positive",
    "likelihood_ratio_negative",
    # Multiclass classification metrics (averages)
    "average_specificity_score",
    "average_sensitivity_score",
    "average_power_score",
    "average_negative_predictive_score",
    "average_false_negative_score",
    "average_false_positive_score",
    "average_false_discovery_score",
    "average_false_omission_rate",
    "average_type_one_error_score",
    "average_type_two_error_score",
]
