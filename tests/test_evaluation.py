import numpy as np

from src.evaluation import evaluate_predictions


def test_multiclass_metrics_match_known_fixture():
    truth = np.asarray([0, 1, 0, 1])
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    result = evaluate_predictions(truth, probabilities, ["negative", "positive"], "multiclass")
    assert result["accuracy"] == 1.0
    assert result["precision_macro"] == 1.0
    assert result["f1_macro"] == 1.0
    assert result["roc_auc_macro_ovr"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_roc_auc_is_none_when_test_has_one_class():
    result = evaluate_predictions(
        np.asarray([[1, 0], [1, 0]]),
        np.asarray([[0.8, 0.2], [0.7, 0.3]]),
        ["present", "absent"],
        "multilabel",
    )
    assert result["per_class"]["present"]["roc_auc"] is None
    assert result["per_class"]["absent"]["roc_auc"] is None
    assert result["roc_auc_macro"] is None

