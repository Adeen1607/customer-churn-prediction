import pandas as pd
import pytest

from src.train import evaluate, select_threshold


def test_select_threshold_maximizes_precision_at_required_recall() -> None:
    labels = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.10, 0.40, 0.60, 0.90])

    threshold, precision, recall = select_threshold(
        labels,
        probabilities,
        minimum_recall=1.0,
    )

    assert threshold == pytest.approx(0.60)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_evaluate_reports_confusion_matrix_and_classification_metrics() -> None:
    labels = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.10, 0.70, 0.80, 0.90])
    predictions = pd.Series([0, 1, 1, 1])

    result = evaluate(labels, probabilities, predictions)

    assert result["true_negatives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0
    assert result["true_positives"] == 2
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.8)
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)
