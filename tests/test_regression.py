"""Tests for regression metrics."""

from collections.abc import Sequence

import pytest

from sklearn_other_metrics import (
    adjusted_explained_variance_score,
    adjusted_r2_score,
    group_mean_log_mae,
    mape_score,
    root_mean_squared_error,
    smape_score,
)


class TestAdjustedR2Score:
    """Test suite for adjusted_r2_score function."""

    @pytest.fixture
    def perfect_predictions(self) -> tuple[list[int], list[int]]:
        """Fixture for perfect predictions."""
        return [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]

    @pytest.fixture
    def imperfect_predictions(self) -> tuple[list[int], list[int]]:
        """Fixture for imperfect predictions."""
        return [0, 1, 2, 3, 4], [4, 3, 2, 1, 0]

    @pytest.mark.parametrize(
        "y_true, y_pred, features_vector",
        [
            # None features_vector should raise ValueError
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None),
            # Too many features (>= n-1) should raise ValueError
            (
                [0, 1, 2, 3, 4],
                [4, 3, 2, 1, 0],
                ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"],
            ),
            # Empty features list should raise ValueError
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], []),
        ],
    )
    def test_features_vector_exception(
        self, y_true: Sequence, y_pred: Sequence, features_vector: Sequence | None
    ) -> None:
        """Test that ValueError is raised for invalid features_vector."""
        with pytest.raises(ValueError):
            adjusted_r2_score(y_true, y_pred, features_vector)

    @pytest.mark.parametrize(
        "y_true, y_pred, features_vector, num_features",
        [
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None, None),
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None, -1),
        ],
    )
    def test_num_features_exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        features_vector: None,
        num_features: int | None,
    ) -> None:
        """Test that ValueError is raised for invalid num_features."""
        with pytest.raises(ValueError):
            adjusted_r2_score(y_true, y_pred, features_vector, num_features)

    @pytest.mark.parametrize(
        "features_vector,num_features,expected",
        [
            (["feature_1", "feature_2"], None, 1.0),
            (None, 2, 1.0),
        ],
    )
    def test_perfect_correlation(
        self,
        perfect_predictions: tuple[list[int], list[int]],
        features_vector: Sequence[str] | None,
        num_features: int | None,
        expected: float,
    ) -> None:
        """Test adjusted R² score with perfect correlation."""
        y_true, y_pred = perfect_predictions
        result = adjusted_r2_score(y_true, y_pred, features_vector, num_features)
        assert result == pytest.approx(expected)


class TestAdjustedExplainedVarianceScore:
    """Test suite for adjusted_explained_variance_score function."""

    @pytest.fixture
    def perfect_predictions(self) -> tuple[list[int], list[int]]:
        """Fixture for perfect predictions."""
        return [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]

    @pytest.mark.parametrize(
        "y_true, y_pred, features_vector",
        [
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None),
            (
                [0, 1, 2, 3, 4],
                [4, 3, 2, 1, 0],
                ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"],
            ),
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], []),
        ],
    )
    def test_features_vector_exception(
        self, y_true: Sequence, y_pred: Sequence, features_vector: Sequence | None
    ) -> None:
        """Test that ValueError is raised for invalid features_vector."""
        with pytest.raises(ValueError):
            adjusted_explained_variance_score(y_true, y_pred, features_vector)

    @pytest.mark.parametrize(
        "y_true, y_pred, features_vector, num_features",
        [
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None, None),
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None, -1),
        ],
    )
    def test_num_features_exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        features_vector: None,
        num_features: int | None,
    ) -> None:
        """Test that ValueError is raised for invalid num_features."""
        with pytest.raises(ValueError):
            adjusted_explained_variance_score(y_true, y_pred, features_vector, num_features)

    @pytest.mark.parametrize(
        "features_vector,num_features,expected",
        [
            (["feature_1", "feature_2"], None, 1.0),
            (None, 2, 1.0),
        ],
    )
    def test_perfect_correlation(
        self,
        perfect_predictions: tuple[list[int], list[int]],
        features_vector: Sequence[str] | None,
        num_features: int | None,
        expected: float,
    ) -> None:
        """Test adjusted explained variance score with perfect correlation."""
        y_true, y_pred = perfect_predictions
        result = adjusted_explained_variance_score(y_true, y_pred, features_vector, num_features)
        assert result == pytest.approx(expected)


class TestMapeScore:
    """Test suite for mape_score function."""

    @staticmethod
    def test_zero_in_y_true_raises_exception() -> None:
        """Test that ZeroDivisionError is raised when y_true contains zero."""
        y_true, y_pred = [0, 2, 3, 4], [1, 1, 2, 3]
        with pytest.raises(ZeroDivisionError):
            mape_score(y_true, y_pred)

    @staticmethod
    def test_perfect_correlation() -> None:
        """Test MAPE score with perfect predictions."""
        y_true = [1, 2, 3, 4]
        y_pred = [1, 2, 3, 4]
        assert mape_score(y_true, y_pred) == pytest.approx(0.0)


class TestSmapeScore:
    """Test suite for smape_score function."""

    @staticmethod
    def test_perfect_correlation() -> None:
        """Test SMAPE score with perfect predictions."""
        y_true = [1, 2, 3, 4]
        y_pred = [1, 2, 3, 4]
        assert smape_score(y_true, y_pred) == pytest.approx(0.0)


class TestRootMeanSquaredError:
    """Test suite for root_mean_squared_error function."""

    @staticmethod
    def test_perfect_correlation() -> None:
        """Test RMSE with perfect predictions."""
        y_true = [1, 2, 3, 4]
        y_pred = [1, 2, 3, 4]
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(0.0)


class TestGroupMeanLogMAE:
    """Test suite for group_mean_log_mae function."""

    @staticmethod
    def test_perfect_correlation() -> None:
        """Test group mean log MAE with perfect predictions."""
        y_true, y_pred, groups = [0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 2, 2]
        result = group_mean_log_mae(y_true, y_pred, groups)
        assert result == pytest.approx(-20.72326583694641)


if __name__ == "__main__":
    pytest.main([__file__])
