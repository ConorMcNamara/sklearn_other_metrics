# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modern package structure with src/ layout
- PEP 561 compliance with py.typed marker
- Comprehensive type hints using Python 3.10+ syntax
- Pre-commit hooks configuration
- Modern GitHub Actions CI/CD pipeline
- Comprehensive test suite with pytest fixtures
- Code coverage reporting
- Mypy strict type checking
- Ruff linting and formatting

### Changed
- **BREAKING**: Package structure moved to src/sklearn_other_metrics
- **BREAKING**: Updated minimum Python version to 3.10
- **BREAKING**: Type hints now use modern `|` syntax instead of `Union`
- Replaced generic `Exception` with specific exception types:
  - `ValueError` for invalid values
  - `TypeError` for type errors
  - `ZeroDivisionError` for division by zero
- Updated dependencies to modern versions
- Split monolithic module into `_regression.py` and `_classification.py`
- Improved error messages with more descriptive text
- Modernized test suite with better organization and fixtures

### Fixed
- Type annotation inconsistencies in function signatures
- Improved error handling and exception specificity

### Removed
- Dropped support for Python < 3.10
- Removed poetry.lock (package now uses hatchling)
- Removed old requirements.txt (use pyproject.toml instead)

## [0.1.0] - Initial Release

### Added
- Initial implementation of regression metrics:
  - `adjusted_r2_score`
  - `adjusted_explained_variance_score`
  - `mape_score`
  - `smape_score`
  - `root_mean_squared_error`
  - `group_mean_log_mae`

- Initial implementation of classification metrics:
  - `specificity_score`
  - `sensitivity_score`
  - `power_score`
  - `negative_predictive_score`
  - `false_negative_score`
  - `false_positive_score`
  - `false_discovery_score`
  - `false_omission_rate`
  - `type_one_error_score`
  - `type_two_error_score`
  - `j_score`
  - `markedness_score`
  - `likelihood_ratio_positive`
  - `likelihood_ratio_negative`
  - Average versions of all multiclass metrics

[Unreleased]: https://github.com/ConorMcNamara/sklearn_other_metrics/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ConorMcNamara/sklearn_other_metrics/releases/tag/v0.1.0
