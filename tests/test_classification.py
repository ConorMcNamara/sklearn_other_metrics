"""Tests for classification metrics."""

from collections.abc import Sequence

import pytest

from sklearn_other_metrics import (
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


class TestGetClassificationLabels:
    """Test suite for get_classification_labels function."""

    @pytest.mark.parametrize(
        "y_true, y_pred",
        [([0, 1, 2, 3, 4], [0, 0, 1, 1, 0]), ([0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])],
    )
    def test_more_than_two_classes_raises_exception(
        self, y_true: Sequence[int], y_pred: Sequence[int]
    ) -> None:
        """Test that ValueError is raised when more than 2 classes are present."""
        with pytest.raises(ValueError):
            get_classification_labels(y_true, y_pred)

    @staticmethod
    def test_correct_label_counts() -> None:
        """Test that classification labels are correctly calculated."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        true_positive, false_positive, false_negative, true_negative = get_classification_labels(
            y_true, y_pred
        )
        assert true_positive == 2
        assert false_positive == 0
        assert true_negative == 2
        assert false_negative == 0


class TestSpecificityScore:
    """Test suite for specificity_score function."""

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], False, None),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], False, [1, 2]),
        ],
    )
    def test_invalid_positive_class_raises_exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: Sequence | None,
    ) -> None:
        """Test that appropriate exceptions are raised for invalid positive_class."""
        with pytest.raises((TypeError, ValueError)):
            specificity_score(y_true, y_pred, is_binary, positive_class)

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 3, 0.5),
        ],
    )
    def test_specificity_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test specificity score calculation."""
        result = specificity_score(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)


class TestAverageSpecificityScore:
    """Test suite for average_specificity_score function."""

    @staticmethod
    def test_multiclass_average() -> None:
        """Test average specificity score for multiclass problem."""
        y_true = [1, 1, 2, 2, 3, 3]
        y_pred = [1, 2, 3, 3, 2, 1]
        result = average_specificity_score(y_true, y_pred)
        assert isinstance(result, float)


class TestSensitivityScore:
    """Test suite for sensitivity_score function."""

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], False, None),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], False, [1, 2]),
        ],
    )
    def test_invalid_positive_class_raises_exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: Sequence | None,
    ) -> None:
        """Test that appropriate exceptions are raised for invalid positive_class."""
        with pytest.raises((TypeError, ValueError)):
            sensitivity_score(y_true, y_pred, is_binary, positive_class)

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 3, 0.0),
        ],
    )
    def test_sensitivity_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test sensitivity score calculation."""
        result = sensitivity_score(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)


class TestPowerScore:
    """Test suite for power_score function (alias for sensitivity)."""

    @staticmethod
    def test_power_equals_sensitivity() -> None:
        """Test that power_score equals sensitivity_score."""
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        power = power_score(y_true, y_pred)
        sensitivity = sensitivity_score(y_true, y_pred)
        assert power == pytest.approx(sensitivity)


class TestAveragePowerScore:
    """Test suite for average_power_score function."""

    @staticmethod
    def test_average_power_equals_average_sensitivity() -> None:
        """Test that average_power_score equals average_sensitivity_score."""
        y_true = [1, 1, 2, 2, 3, 3]
        y_pred = [1, 2, 3, 3, 2, 1]
        avg_power = average_power_score(y_true, y_pred)
        avg_sensitivity = average_sensitivity_score(y_true, y_pred)
        assert avg_power == pytest.approx(avg_sensitivity)


class TestNegativePredictiveScore:
    """Test suite for negative_predictive_score function."""

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], False, None),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], False, [1, 2]),
        ],
    )
    def test_invalid_positive_class_raises_exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: Sequence | None,
    ) -> None:
        """Test that appropriate exceptions are raised for invalid positive_class."""
        with pytest.raises((TypeError, ValueError)):
            negative_predictive_score(y_true, y_pred, is_binary, positive_class)

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 3, 0.5),
        ],
    )
    def test_negative_predictive_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test negative predictive score calculation."""
        result = negative_predictive_score(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)


class TestFalseNegativeScore:
    """Test suite for false_negative_score function."""

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 2, 1.0),
        ],
    )
    def test_false_negative_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test false negative score calculation."""
        result = false_negative_score(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)


class TestTypeErrorScores:
    """Test suite for Type I and Type II error scores."""

    @staticmethod
    def test_type_two_equals_false_negative() -> None:
        """Test that type_two_error_score equals false_negative_score."""
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        type_two = type_two_error_score(y_true, y_pred)
        false_neg = false_negative_score(y_true, y_pred)
        assert type_two == pytest.approx(false_neg)

    @staticmethod
    def test_type_one_equals_false_positive() -> None:
        """Test that type_one_error_score equals false_positive_score."""
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        type_one = type_one_error_score(y_true, y_pred)
        false_pos = false_positive_score(y_true, y_pred)
        assert type_one == pytest.approx(false_pos)


class TestFalsePositiveScore:
    """Test suite for false_positive_score function."""

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 1, 0.25),
        ],
    )
    def test_false_positive_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test false positive score calculation."""
        result = false_positive_score(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)


class TestFalseDiscoveryScore:
    """Test suite for false_discovery_score function."""

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 3, 1.0),
        ],
    )
    def test_false_discovery_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test false discovery score calculation."""
        result = false_discovery_score(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)


class TestFalseOmissionRate:
    """Test suite for false_omission_rate function."""

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 2, 0.5),
        ],
    )
    def test_false_omission_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test false omission rate calculation."""
        result = false_omission_rate(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)


class TestJScore:
    """Test suite for j_score function (Youden's J statistic)."""

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 0.0),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 2, -0.5),
        ],
    )
    def test_j_score_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test J-score calculation."""
        result = j_score(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)


class TestMarkednessScore:
    """Test suite for markedness_score function."""

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 0.0),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 2, -0.5),
        ],
    )
    def test_markedness_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test markedness score calculation."""
        result = markedness_score(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)


class TestLikelihoodRatios:
    """Test suite for likelihood ratio functions."""

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 1.0),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 2, 0.0),
        ],
    )
    def test_likelihood_ratio_positive_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test positive likelihood ratio calculation."""
        result = likelihood_ratio_positive(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "y_true, y_pred, is_binary, positive_class, expected",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], True, None, 1.0),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], False, 3, 0.5),
        ],
    )
    def test_likelihood_ratio_negative_calculation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        is_binary: bool,
        positive_class: int | None,
        expected: float,
    ) -> None:
        """Test negative likelihood ratio calculation."""
        result = likelihood_ratio_negative(y_true, y_pred, is_binary, positive_class)
        assert result == pytest.approx(expected)


class TestAverageMetrics:
    """Test suite for average metric functions."""

    @pytest.fixture
    def multiclass_data(self) -> tuple[list[int], list[int]]:
        """Fixture for multiclass classification data."""
        return [1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1]

    def test_all_average_functions_return_float(
        self, multiclass_data: tuple[list[int], list[int]]
    ) -> None:
        """Test that all average functions return float values."""
        y_true, y_pred = multiclass_data

        assert isinstance(average_specificity_score(y_true, y_pred), float)
        assert isinstance(average_sensitivity_score(y_true, y_pred), float)
        assert isinstance(average_power_score(y_true, y_pred), float)
        assert isinstance(average_negative_predictive_score(y_true, y_pred), float)
        assert isinstance(average_false_negative_score(y_true, y_pred), float)
        assert isinstance(average_false_positive_score(y_true, y_pred), float)
        assert isinstance(average_false_discovery_score(y_true, y_pred), float)
        assert isinstance(average_false_omission_rate(y_true, y_pred), float)
        assert isinstance(average_type_one_error_score(y_true, y_pred), float)
        assert isinstance(average_type_two_error_score(y_true, y_pred), float)


if __name__ == "__main__":
    pytest.main([__file__])
