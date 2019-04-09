import unittest
import other_metrics

class TestOtherMetrics(unittest.TestCase):

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

    def test_checkLabels_String_Exception(self):
        y_pred = "1, 2, 3, 4 ,5"
        self.assertRaises(Exception, other_metrics.check_array, y_pred)

    def test_checkLabels_Dict_Exception(self):
        y_pred = {"1": 1, "2": 2}
        self.assertRaises(Exception, other_metrics.check_array, y_pred)

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
