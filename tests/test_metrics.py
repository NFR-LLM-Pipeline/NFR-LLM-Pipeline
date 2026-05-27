"""Tests for src.metrics."""

from src.metrics import confusion_matrix, calc_metrics

def test_perfect_predictions():
    gold_labels = ["A", "B", "A", "B"]
    predicted_labels = ["A", "B", "A", "B"]
    result = calc_metrics(gold_labels, predicted_labels, labels=["A", "B"])

    assert result["A"]["precision"] == 1.0
    assert result["A"]["recall"] == 1.0
    assert result["A"]["f1"] == 1.0

def test_all_wrong():
    gold_labels = ["A", "A", "A"]
    predicted_labels = ["B", "B", "B"]
    result = calc_metrics(gold_labels, predicted_labels, labels=["A", "B"])

    assert result["A"]["recall"] == 0.0
    assert result["B"]["precision"] == 0.0

def test_mixed_predictions():
    gold_labels = ["NFR", "NFR", "FR", "FR", "NFR"]
    predicted_labels = ["NFR", "FR", "FR", "NFR", "NFR"]
    result = calc_metrics(gold_labels, predicted_labels, labels=["NFR", "FR"])
    
    assert abs(result["NFR"]["precision"] - 2 / 3) < 1e-9
    assert abs(result["NFR"]["recall"] - 2 / 3) < 1e-9

def test_confusion_matrix_multiclass():
    gold_labels = ["performance", "security", "other", "performance"]
    predicted_labels = ["performance", "performance", "other", "security"]
    labels = ["performance", "security", "other"]
    cm = confusion_matrix(gold_labels, predicted_labels, labels=labels)

    assert cm["performance"]["performance"] == 1
    assert cm["performance"]["security"] == 1
    assert cm["security"]["performance"] == 1
    assert cm["other"]["other"] == 1
