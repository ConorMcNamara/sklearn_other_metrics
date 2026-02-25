# Contributing to sklearn-other-metrics

Thank you for your interest in contributing to sklearn-other-metrics! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Setting Up Your Development Environment

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/sklearn_other_metrics.git
cd sklearn_other_metrics
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the package in development mode**

```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks**

```bash
pre-commit install
```

## Development Workflow

### Code Style

This project uses:
- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Pre-commit** hooks to ensure code quality

Before committing, ensure your code passes all checks:

```bash
# Format code
ruff format .

# Lint code
ruff check . --fix

# Type check
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/test_regression.py

# Run specific test
pytest tests/test_regression.py::TestMapeScore::test_perfect_correlation
```

### Type Hints

All code must have complete type hints. We use modern Python 3.10+ syntax:

```python
# Good
def func(x: int | float, y: str | None = None) -> list[int]:
    ...

# Avoid (old syntax)
from typing import Union, Optional, List
def func(x: Union[int, float], y: Optional[str] = None) -> List[int]:
    ...
```

### Adding New Metrics

When adding a new metric:

1. **Add the implementation** to the appropriate module:
   - `src/sklearn_other_metrics/_regression.py` for regression metrics
   - `src/sklearn_other_metrics/_classification.py` for classification metrics

2. **Include comprehensive docstrings** following NumPy style:

```python
def new_metric(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Brief description of the metric.

    Longer description if needed.

    Parameters
    ----------
    y_true : list or array-like
        The true values
    y_pred : list or array-like
        The predicted values

    Returns
    -------
    float
        The computed metric value

    Raises
    ------
    ValueError
        If input validation fails

    Examples
    --------
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1.1, 2.1, 2.9]
    >>> new_metric(y_true, y_pred)
    0.05
    """
```

3. **Add to public API** in `src/sklearn_other_metrics/__init__.py`:
   - Import the function
   - Add to `__all__` list

4. **Write tests** in the appropriate test file:
   - Use pytest fixtures where appropriate
   - Test normal cases, edge cases, and error conditions
   - Aim for >90% code coverage

5. **Update documentation**:
   - Add entry to README.md
   - Update CHANGELOG.md

### Exception Handling

Use specific exception types:

```python
# Good
if num_features < 1:
    raise ValueError("num_features must be at least 1")

if not isinstance(positive_class, (str, int)):
    raise TypeError("positive_class must be str or int")

if 0 in y_true:
    raise ZeroDivisionError("Cannot calculate MAPE when y_true contains zero")

# Avoid
if something_wrong:
    raise Exception("Something went wrong")
```

## Pull Request Process

1. **Create a feature branch**

```bash
git checkout -b feature/my-new-feature
```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Ensure all checks pass**

```bash
pytest
ruff check .
ruff format --check .
mypy src/
```

4. **Commit your changes**

```bash
git add .
git commit -m "Add new feature: description"
```

Pre-commit hooks will run automatically. Fix any issues they find.

5. **Push to your fork**

```bash
git push origin feature/my-new-feature
```

6. **Create a Pull Request**
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template with:
     - Description of changes
     - Related issues (if any)
     - Testing done
     - Breaking changes (if any)

### PR Checklist

- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Type hints added/updated
- [ ] Code formatted with ruff
- [ ] No linting errors
- [ ] Commit messages are descriptive

## Reporting Issues

When reporting bugs or requesting features, please:

1. **Search existing issues** to avoid duplicates
2. **Use issue templates** if available
3. **Include**:
   - Python version
   - Package version
   - Minimal reproducible example
   - Expected vs actual behavior
   - Error messages and tracebacks

## Code Review Process

All submissions require review. Maintainers will:
- Check code quality and style
- Verify tests pass
- Review documentation
- Suggest improvements

Be patient and responsive to feedback!

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion on GitHub
- Reach out to maintainers

Thank you for contributing! 🎉
