import unittest
import other_metrics

class TestOtherMetrics(unittest.TestCase):

    def test_adjustedR2Score_Dictionary_Exception(self):
        y_true = {1: 1, 2: 2}
        y_pred = {1: 1, 2: 2}
        self.assertRaises(Exception, other_metrics.adjusted_r2_score, y_true, y_pred, ['feature_1'])

    def test_adjustedR2Score_NoFeatures_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.adjusted_r2_score, y_true, y_pred)

    def test_adjustedR2Score_TooManyFeaturesList_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.adjusted_r2_score, y_true, y_pred,
                          ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5'])

    def test_adjustedR2Score_TooLittleFeaturesList_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.adjusted_r2_score, y_true, y_pred, [])

    def test_adjustedR2Score_TooManyFeaturesInt_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.adjusted_r2_score, y_true, y_pred, None, 5)

    def test_adjustedR2Score_TooLittleFeaturesInt_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.adjusted_r2_score, y_true, y_pred, None, -1)

    def test_adjustedR2Score_PerfectCorrelationList_One(self):
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        self.assertEqual(other_metrics.adjusted_r2_score(y_true, y_pred, features_vector=['feature_1', 'feature_2']), 1.0)

    def test_adjustedR2Score_PerfectCorrelationInt_One(self):
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        self.assertEqual(other_metrics.adjusted_r2_score(y_true, y_pred, num_features=2), 1.0)

    def test_adjustedExplainedVarianceScore_NoFeatures_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.adjusted_r2_score, y_true, y_pred)

    def test_adjustedExplainedVarianceScore_TooManyFeaturesList_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.adjusted_explained_variance_score, y_true, y_pred,
                          ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5'])

    def test_adjustedExplainedVarianceScore_TooLittleFeaturesList_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.adjusted_explained_variance_score, y_true, y_pred, [])

    def test_adjustedExplainedVarianceScore_TooManyFeaturesInt_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.adjusted_explained_variance_score, y_true, y_pred, None, 5)

    def test_adjustedExplainedVarianceScore_TooLittleFeaturesInt_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.adjusted_explained_variance_score, y_true, y_pred, None, -1)

    def test_adjustedExplainedVarianceScore_PerfectCorrelationList_One(self):
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        self.assertEqual(other_metrics.adjusted_explained_variance_score(y_true, y_pred, features_vector=['feature_1', 'feature_2']),
                         1.0)

    def test_adjustedExplainedVarianceScore_PerfectCorrelationInt_One(self):
        y_true = [0, 1, 2, 3]
        y_pred = [0, 1, 2, 3]
        self.assertEqual(other_metrics.adjusted_explained_variance_score(y_true, y_pred, num_features=2), 1.0)

    def test_getClassificationLabels_MoreThanTwoClasses_Exception(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 2, 1, 0]
        self.assertRaises(Exception, other_metrics.get_classification_labels, y_true, y_pred)

    def test_specificityScore_NonePositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.specificity_score, y_true, y_pred, 'multiclass')

    def test_specificityScore_NotValidPositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.specificity_score, y_true, y_pred, "multiclass", [1, 2])

    def test_sensitivityScore_NonePositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.sensitivity_score, y_true, y_pred, 'multiclass')

    def test_sensitivityScore_NotValidPositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.sensitivity_score, y_true, y_pred, "multiclass", [1, 2])

    def test_negativePredictiveScore_NonePositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.negative_predictive_score, y_true, y_pred, 'multiclass')

    def test_negativePredictiveScore_NotValidPositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.negative_predictive_score, y_true, y_pred, "multiclass", [1, 2])

    def test_falseNegativeScore_NonePositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.false_negative_score, y_true, y_pred, 'multiclass')

    def test_falseNegativeScoreNotValidPositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.false_negative_score, y_true, y_pred, "multiclass", [1, 2])

    def test_falsePositiveScore_NonePositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.false_positive_score, y_true, y_pred, 'multiclass')

    def test_falsePositiveScore_NotValidPositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.false_positive_score, y_true, y_pred, "multiclass", [1, 2])

    def test_falseDiscoveryScore_NonePositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.false_discovery_score, y_true, y_pred, 'multiclass')

    def test_falseDiscoveryScore_NotValidPositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.false_discovery_score, y_true, y_pred, "multiclass", [1, 2])

    def test_falseOmissionRate_NonePositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.false_omission_rate, y_true, y_pred, 'multiclass')

    def test_specificityScore_NotValidPositiveClass_Exception(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        self.assertRaises(Exception, other_metrics.false_omission_rate, y_true, y_pred, "multiclass", [1, 2])

    def test_specificityScore_binaryResult_pointFive(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.specificity_score(y_true, y_pred), 0.5)

    def test_specificityScore_multiclassResult_pointFive(self):
        y_true = [1, 1, 2, 2, 3, 3]
        y_pred = [1, 2, 3, 3, 2, 1]
        self.assertEqual(other_metrics.specificity_score(y_true, y_pred, 'multiclass', 3), 0.5)

    def test_sensitivityScore_binaryResult_pointFive(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.sensitivity_score(y_true, y_pred), 0.5)

    def test_sensitivityScore_multiclassResult_Zero(self):
        y_true = [1, 1, 2, 2, 3, 3]
        y_pred = [1, 2, 3, 3, 2, 1]
        self.assertEqual(other_metrics.sensitivity_score(y_true, y_pred, 'multiclass', 3), 0.0)

    def test_negativePredictiveScore_binaryResult_pointFive(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.negative_predictive_score(y_true, y_pred), 0.5)

    def test_negativePredictiveScore_multiclassResult_pointFive(self):
        y_true = [1, 1, 2, 2, 3, 3]
        y_pred = [1, 2, 3, 3, 2, 1]
        self.assertEqual(other_metrics.negative_predictive_score(y_true, y_pred, 'multiclass', 3), 0.5)

    def test_falseNegativeScore_binaryResult_pointFive(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.false_negative_score(y_true, y_pred), 0.5)

    def test_falseNegativeScore_multiclassResult_One(self):
        y_true = [1, 1, 2, 2, 3, 3]
        y_pred = [1, 2, 3, 3, 2, 1]
        self.assertEqual(other_metrics.false_negative_score(y_true, y_pred, 'multiclass', 2), 1.0)

    def test_falsePositiveScore_binaryResult_pointFive(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.false_positive_score(y_true, y_pred), 0.5)

    def test_falsePositiveScore_multiclassResult_pointTwoFive(self):
        y_true = [1, 1, 2, 2, 3, 3]
        y_pred = [1, 2, 3, 3, 2, 1]
        self.assertEqual(other_metrics.false_positive_score(y_true, y_pred, 'multiclass', 1), 0.25)

    def test_falseDiscoveryScore_binaryResult_pointFive(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.false_discovery_score(y_true, y_pred), 0.5)

    def test_falseDiscoveryScore_multiclassResult_One(self):
        y_true = [1, 1, 2, 2, 3, 3]
        y_pred = [1, 2, 3, 3, 2, 1]
        self.assertEqual(other_metrics.false_discovery_score(y_true, y_pred, 'multiclass', 3), 1.0)

    def test_falseOmissionRate_binaryResult_pointFive(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.false_omission_rate(y_true, y_pred), 0.5)

    def test_falseOmissionRate_multiclassResult_pointFive(self):
        y_true = [1, 1, 2, 2, 3, 3]
        y_pred = [1, 2, 3, 3, 2, 1]
        self.assertEqual(other_metrics.false_omission_rate(y_true, y_pred, 'multiclass', 2), 0.5)

    def test_jScore_binaryResult_Zero(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.j_score(y_true, y_pred), 0.0)

    def test_markednessScore_binaryResult_Zero(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.markedness_score(y_true, y_pred), 0.0)

    def test_likelihoodRatioPositive_binaryResult_One(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.likelihood_ratio_positive(y_true, y_pred), 1.0)

    def test_likelihoodRatioNegative_binaryResult_One(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertEqual(other_metrics.likelihood_ratio_negative(y_true, y_pred), 1.0)
