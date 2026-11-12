"""Guard against invalid threshold-selection inputs and misleading recall results."""

import math

import pandas as pd
import pytest

from src.train import select_threshold


@pytest.mark.parametrize("minimum_recall", [0.0, -0.1, 1.01, float("nan")])
def test_rejects_invalid_recall_requirement(minimum_recall):
    with pytest.raises(ValueError, match="minimum_recall"):
        select_threshold(pd.Series([0, 1]), pd.Series([0.2, 0.8]), minimum_recall)


@pytest.mark.parametrize(
    "labels, scores, error",
    [
        ([0, 1], [0.2], "equal lengths"),
        ([], [], "nonempty"),
        ([0, 0], [0.2, 0.8], "positive label"),
        ([0, 2], [0.2, 0.8], "binary"),
        ([0, None], [0.2, 0.8], "binary"),
        ([0, 1], [0.2, float("nan")], "Probabilities"),
        ([0, 1], [0.2, float("inf")], "Probabilities"),
        ([0, 1], [-0.1, 0.8], "Probabilities"),
        ([0, 1], [0.2, 1.1], "Probabilities"),
    ],
)
def test_rejects_invalid_validation_samples(labels, scores, error):
    with pytest.raises(ValueError, match=error):
        select_threshold(pd.Series(labels), pd.Series(scores), 0.75)


def test_valid_threshold_selection_is_unchanged():
    threshold, precision, recall = select_threshold(
        pd.Series([0, 0, 1, 1]), pd.Series([0.1, 0.4, 0.6, 0.9]), 1.0
    )
    assert (threshold, precision, recall) == pytest.approx((0.6, 1.0, 1.0))
