# sklearn-other-metrics

[![CI](https://github.com/ConorMcNamara/sklearn_other_metrics/workflows/CI/badge.svg)](https://github.com/ConorMcNamara/sklearn_other_metrics/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typed with mypy](https://img.shields.io/badge/typed-mypy-blue.svg)](https://github.com/python/mypy)

Additional regression and classification metrics for scikit-learn that are commonly used in machine learning but not included in the standard library.

## Features

- 🎯 **Regression Metrics**: MAPE, SMAPE, Adjusted R², RMSE, Group Mean Log MAE
- 📊 **Classification Metrics**: Specificity, Sensitivity, NPV, FDR, FOR, Likelihood Ratios, and more
- 🔒 **Type Safe**: Full type hints with PEP 561 compliance
- 📦 **Modern**: Uses Python 3.10+ features and follows best practices
- 🚀 **Fast**: Optimized implementations using NumPy

## Installation

```bash
pip install sklearn-other-metrics
```

For development:

```bash
git clone https://github.com/ConorMcNamara/sklearn_other_metrics.git
cd sklearn_other_metrics
pip install -e ".[dev]"
```

## Quick Start

### Regression Metrics

```python
from sklearn_other_metrics import mape_score, adjusted_r2_score

# Mean Absolute Percentage Error
y_true = [100, 200, 300]
y_pred = [110, 190, 310]
mape = mape_score(y_true, y_pred)
print(f"MAPE: {mape:.2f}%")  # MAPE: 5.56%

# Adjusted R² Score
y_true = [1, 2, 3, 4, 5]
y_pred = [1.1, 2.0, 2.9, 4.2, 4.8]
adj_r2 = adjusted_r2_score(y_true, y_pred, num_features=2)
print(f"Adjusted R²: {adj_r2:.4f}")  # Adjusted R²: 0.9825
```

### Classification Metrics

```python
from sklearn_other_metrics import (
    specificity_score,
    sensitivity_score,
    j_score,
)

y_true = [0, 0, 1, 1]
y_pred = [0, 1, 1, 1]

# Specificity (True Negative Rate)
spec = specificity_score(y_true, y_pred)
print(f"Specificity: {spec:.2f}")  # Specificity: 0.50

# Sensitivity (Recall / True Positive Rate)
sens = sensitivity_score(y_true, y_pred)
print(f"Sensitivity: {sens:.2f}")  # Sensitivity: 1.00

# Youden's J Statistic
j = j_score(y_true, y_pred)
print(f"J-Score: {j:.2f}")  # J-Score: 0.50
```

## Available Metrics

### Regression Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Adjusted R² | `adjusted_r2_score` | R² penalized for number of features |
| Adjusted Explained Variance | `adjusted_explained_variance_score` | Explained variance penalized for features |
| MAPE | `mape_score` | Mean Absolute Percentage Error |
| SMAPE | `smape_score` | Symmetric Mean Absolute Percentage Error |
| RMSE | `root_mean_squared_error` | Root Mean Squared Error |
| Group Mean Log MAE | `group_mean_log_mae` | Grouped logarithmic MAE |

### Classification Metrics

#### Binary Classification

| Metric | Function | Description |
|--------|----------|-------------|
| Specificity | `specificity_score` | True Negative Rate |
| Sensitivity | `sensitivity_score` | True Positive Rate (Recall) |
| Power | `power_score` | Alias for sensitivity |
| Negative Predictive Value | `negative_predictive_score` | NPV = TN / (TN + FN) |
| False Negative Rate | `false_negative_score` | FN / (FN + TP) |
| False Positive Rate | `false_positive_score` | FP / (FP + TN) |
| False Discovery Rate | `false_discovery_score` | FP / (FP + TP) |
| False Omission Rate | `false_omission_rate` | FN / (FN + TN) |
| Type I Error | `type_one_error_score` | Alias for false positive rate |
| Type II Error | `type_two_error_score` | Alias for false negative rate |
| J-Score | `j_score` | Youden's J statistic (Sensitivity + Specificity - 1) |
| Markedness | `markedness_score` | Precision + NPV - 1 |
| Likelihood Ratio Positive | `likelihood_ratio_positive` | Sensitivity / (1 - Specificity) |
| Likelihood Ratio Negative | `likelihood_ratio_negative` | Specificity / (1 - Sensitivity) |

#### Multiclass Classification

All binary metrics have `average_*` versions for multiclass problems:
- `average_specificity_score`
- `average_sensitivity_score`
- `average_false_positive_score`
- ... and more

```python
from sklearn_other_metrics import average_specificity_score

# Multiclass example
y_true = [0, 0, 1, 1, 2, 2]
y_pred = [0, 1, 1, 1, 2, 0]

avg_spec = average_specificity_score(y_true, y_pred)
print(f"Average Specificity: {avg_spec:.2f}")
```

## Advanced Usage

### Multiclass with Specific Positive Class

```python
from sklearn_other_metrics import specificity_score

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 1, 0, 2, 2]

# Calculate specificity for class 1
spec = specificity_score(
    y_true,
    y_pred,
    is_binary=False,
    positive_class=1
)
print(f"Specificity for class 1: {spec:.2f}")
```

### Adjusted Metrics with Feature Vector

```python
from sklearn_other_metrics import adjusted_r2_score

y_true = [1, 2, 3, 4, 5]
y_pred = [1.1, 2.0, 2.9, 4.2, 4.8]

# Pass feature names
features = ["feature_1", "feature_2", "feature_3"]
adj_r2 = adjusted_r2_score(y_true, y_pred, features_vector=features)

# Or just the count
adj_r2 = adjusted_r2_score(y_true, y_pred, num_features=3)
```

## Error Handling

The package uses specific exception types for better error handling:

```python
from sklearn_other_metrics import mape_score

y_true = [0, 1, 2, 3]  # Contains zero!
y_pred = [1, 2, 3, 4]

try:
    mape = mape_score(y_true, y_pred)
except ZeroDivisionError as e:
    print(f"Error: {e}")
    # Error: Cannot calculate MAPE when y_true contains zero values
```

Exception types used:
- `ValueError`: Invalid input values (e.g., too many features, invalid class labels)
- `TypeError`: Wrong type for parameters (e.g., positive_class must be str or int)
- `ZeroDivisionError`: Division by zero in calculations

## Type Safety

This package is fully typed and passes strict mypy checking:

```python
from sklearn_other_metrics import mape_score
import numpy as np

# All these work fine
y_true_list = [1, 2, 3]
y_true_array = np.array([1, 2, 3])
y_true_series = pd.Series([1, 2, 3])

mape_score(y_true_list, [1.1, 2.1, 3.1])
mape_score(y_true_array, np.array([1.1, 2.1, 3.1]))
mape_score(y_true_series, pd.Series([1.1, 2.1, 3.1]))
```

## Requirements

- Python ≥ 3.10
- NumPy ≥ 2.0.0
- pandas ≥ 2.0.0
- scikit-learn ≥ 1.4.0

## Development

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/ConorMcNamara/sklearn_other_metrics.git
cd sklearn_other_metrics

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate XML coverage report (for CI/CD)
pytest --cov=src --cov-report=xml

# Run specific test file with coverage
pytest tests/test_regression.py --cov=src

# Set minimum coverage threshold (fail if below 90%)
pytest --cov=src --cov-fail-under=90
```

View the HTML coverage report by opening `htmlcov/index.html` in your browser after running with `--cov-report=html`.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{sklearn_other_metrics,
  author = {McNamara, Conor},
  title = {sklearn-other-metrics: Additional metrics for scikit-learn},
  year = {2024},
  url = {https://github.com/ConorMcNamara/sklearn_other_metrics}
}
```

## Acknowledgments

- Built on top of [scikit-learn](https://scikit-learn.org/)
- Inspired by commonly used metrics in machine learning research and competitions

## Support

- 📧 Email: conor.s.mcnamara@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/ConorMcNamara/sklearn_other_metrics/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/ConorMcNamara/sklearn_other_metrics/discussions)

---

Made with ❤️ by [Conor McNamara](https://github.com/ConorMcNamara)
