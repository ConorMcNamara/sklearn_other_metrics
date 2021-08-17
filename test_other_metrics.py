import other_metrics

from typing import Optional, Sequence

import pytest


class TestAdjustedR2Score:
    @pytest.mark.parametrize(
        "y_true, y_pred, features_vector",
        [
            ({1: 1, 2: 2}, {1: 1, 2: 2}, ["feature_1"]),
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None),
            (
                [0, 1, 2, 3, 4],
                [4, 3, 2, 1, 0],
                ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"],
            ),
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], []),
        ],
    )
    def test_adjustedR2Score_featuresVectorException(
        self, y_true: Sequence, y_pred: Sequence, features_vector: Optional[Sequence]
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.adjusted_r2_score(y_true, y_pred, features_vector)

    @pytest.mark.parametrize(
        "y_true, y_pred, features_vector, num_features",
        [
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None, None),
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None, -1),
        ],
    )
    def test_adjustedR2Score_numFeaturesException(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        features_vector: Optional,
        num_features: Optional[int],
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.adjusted_r2_score(
                y_true, y_pred, features_vector, num_features
            )

    @pytest.mark.parametrize(
        "y_true, y_pred, features_vector, num_features",
        [
            ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], ["feature_1", "feature_2"], None),
            ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], None, 2),
        ],
    )
    def test_adjustedR2Score_PerfectCorrelation(
        self,
        y_true: Sequence,
        y_pred: Sequence,
        features_vector: Optional[Sequence[str]],
        num_features: Optional[int],
    ) -> None:
        assert (
            other_metrics.adjusted_r2_score(
                y_true, y_pred, features_vector, num_features
            )
            == 1
        )


class TestAdjustedExplainedVarianceScore:
    @pytest.mark.parametrize(
        "y_true, y_pred, features_vector",
        [
            ({1: 1, 2: 2}, {1: 1, 2: 2}, ["feature_1"]),
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None),
            (
                [0, 1, 2, 3, 4],
                [4, 3, 2, 1, 0],
                ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"],
            ),
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], []),
        ],
    )
    def test_adjustedExplainedVarianceScore_featuresVectorException(
        self, y_true: Sequence, y_pred: Sequence, features_vector: Optional[Sequence]
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.adjusted_explained_variance_score(
                y_true, y_pred, features_vector
            )

    @pytest.mark.parametrize(
        "y_true, y_pred, features_vector, num_features",
        [
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None, None),
            ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], None, -1),
        ],
    )
    def test_adjustedExplainedVarianceScore_numFeaturesException(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        features_vector: Optional,
        num_features: Optional[int],
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.adjusted_explained_variance_score(
                y_true, y_pred, features_vector, num_features
            )

    @pytest.mark.parametrize(
        "y_true, y_pred, features_vector, num_features",
        [
            ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], ["feature_1", "feature_2"], None),
            ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], None, 2),
        ],
    )
    def test_adjustedR2Score_PerfectCorrelation(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        features_vector: Optional[Sequence[str]],
        num_features: Optional[int],
    ) -> None:
        assert (
            other_metrics.adjusted_explained_variance_score(
                y_true, y_pred, features_vector, num_features
            )
            == 1
        )


class TestMapeScore:
    @staticmethod
    def test_mapeScore_Exception() -> None:
        y_true, y_pred = [0, 2, 3, 4], [1, 1, 2, 3]
        with pytest.raises(Exception):
            other_metrics.mape_score(y_true, y_pred)

    @staticmethod
    def test_mapeScore_PerfectCorrelation() -> None:
        y_true = [1, 2, 3, 4]
        y_pred = [1, 2, 3, 4]
        assert other_metrics.mape_score(y_true, y_pred) == pytest.approx(0.0)


class TestSmapeScore:
    @staticmethod
    def test_smapeError_PerfectCorrelation() -> None:
        y_true = [1, 2, 3, 4]
        y_pred = [1, 2, 3, 4]
        assert other_metrics.smape_score(y_true, y_pred) == pytest.approx(0.0)


class TestGroupMeanLogMAE:
    @staticmethod
    def test_groupMeanLogMAE_PerfectCorrelation() -> None:
        y_true, y_pred, groups = [0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 2, 2]
        assert other_metrics.group_mean_log_mae(
            y_true, y_pred, groups
        ) == pytest.approx(-20.72326583694641)


class TestGetClassificationLabels:
    @pytest.mark.parametrize(
        "y_true, y_pred",
        [([0, 1, 2, 3, 4], [0, 0, 1, 1, 0]), ([0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])],
    )
    def test_getClassificationLabels_MoreThanTwoClasses_Exception(
        self, y_true: Sequence[int], y_pred: Sequence[int]
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.get_classification_labels(y_true, y_pred)

    def test_getClassificationLabels_Results(self) -> None:
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        (
            true_positive,
            false_positive,
            false_negative,
            true_negative,
        ) = other_metrics.get_classification_labels(y_true, y_pred)
        assert true_positive == 2
        assert false_positive == 0
        assert true_negative == 2
        assert false_negative == 0


class TestSpecificityScore:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", None),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", [1, 2]),
        ],
    )
    def test_specificityScore_Exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[Sequence],
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.specificity_score(y_true, y_pred, problem, positive_class)

    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 3),
        ],
    )
    def test_specificityScore_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
    ) -> None:
        assert (
            other_metrics.specificity_score(y_true, y_pred, problem, positive_class)
            == 0.5
        )


class TestSensitivityScore:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", None),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", [1, 2]),
        ],
    )
    def test_sensitivityScore_Exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[Sequence],
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.sensitivity_score(y_true, y_pred, problem, positive_class)

    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class, result",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 3, 0.0),
        ],
    )
    def test_sensitivityScore_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
        result: float,
    ) -> None:
        assert (
            other_metrics.sensitivity_score(y_true, y_pred, problem, positive_class)
            == result
        )


class TestNegativePredictiveScore:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", None),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", [1, 2]),
        ],
    )
    def test_negativePredictiveScore_Exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[Sequence],
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.negative_predictive_score(
                y_true, y_pred, problem, positive_class
            )

    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 3),
        ],
    )
    def test_negativePredictiveScore_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
    ) -> None:
        assert (
            other_metrics.negative_predictive_score(
                y_true, y_pred, problem, positive_class
            )
            == 0.5
        )


class TestFalseNegativeScore:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", None),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", [1, 2]),
        ],
    )
    def test_falseNegativeScore_Exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[Sequence],
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.false_negative_score(y_true, y_pred, problem, positive_class)

    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class, result",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 2, 1),
        ],
    )
    def test_falseNegativeScore_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
        result: float,
    ) -> None:
        assert (
            other_metrics.false_negative_score(y_true, y_pred, problem, positive_class)
            == result
        )


class TestFalsePositiveScore:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", None),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", [1, 2]),
        ],
    )
    def test_falsePositiveScore_Exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[Sequence],
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.false_positive_score(y_true, y_pred, problem, positive_class)

    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class, result",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 1, 0.25),
        ],
    )
    def test_falsePositiveScore_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
        result: float,
    ) -> None:
        assert (
            other_metrics.false_positive_score(y_true, y_pred, problem, positive_class)
            == result
        )


class TestFalseDiscoveryScore:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", None),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", [1, 2]),
        ],
    )
    def test_falseDiscoveryScore_Exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[Sequence],
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.false_discovery_score(y_true, y_pred, problem, positive_class)

    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class, result",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None, 0.5),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 3, 1.0),
        ],
    )
    def test_falseDiscoveryScore_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
        result: float,
    ) -> None:
        assert (
            other_metrics.false_discovery_score(y_true, y_pred, problem, positive_class)
            == result
        )


class TestFalseOmissionRate:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class",
        [
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", None),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 0, 1, 2], "multiclass", [1, 2]),
        ],
    )
    def test_falseOmissionRate_Exception(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[Sequence],
    ) -> None:
        with pytest.raises(Exception):
            other_metrics.false_omission_rate(y_true, y_pred, problem, positive_class)

    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 2),
        ],
    )
    def test_falseOmissionRate_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
    ) -> None:
        assert (
            other_metrics.false_omission_rate(y_true, y_pred, problem, positive_class)
            == 0.5
        )


class TestJScore:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class, result",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None, 0.0),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 2, -0.5),
        ],
    )
    def test_jScore_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
        result: float,
    ) -> None:
        assert other_metrics.j_score(y_true, y_pred, problem, positive_class) == result


class TestMarkednessScore:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class, result",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None, 0.0),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 2, -0.5),
        ],
    )
    def test_markednessScore_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
        result: float,
    ) -> None:
        assert (
            other_metrics.markedness_score(y_true, y_pred, problem, positive_class)
            == result
        )


class TestLikelihoodRatioPositive:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class, result",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None, 1.0),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 2, 0.0),
        ],
    )
    def test_likelihoodRatioPositive_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
        result: float,
    ) -> None:
        assert (
            other_metrics.likelihood_ratio_positive(
                y_true, y_pred, problem, positive_class
            )
            == result
        )


class TestLikelihoodRatioNegative:
    @pytest.mark.parametrize(
        "y_true, y_pred, problem, positive_class, result",
        [
            ([1, 0, 1, 0], [1, 0, 0, 1], "binary", None, 1.0),
            ([1, 1, 2, 2, 3, 3], [1, 2, 3, 3, 2, 1], "multiclass", 3, 0.5),
        ],
    )
    def test_likelihoodRatioNegative_result(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        problem: str,
        positive_class: Optional[int],
        result: float,
    ) -> None:
        assert (
            other_metrics.likelihood_ratio_negative(
                y_true, y_pred, problem, positive_class
            )
            == result
        )
