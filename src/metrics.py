"""Compute evaluation metrics: precision, recall, F1-score, confusion matrix."""

def calc_metrics(
    gold_labels: list[str],
    predicted_labels: list[str],
    labels: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute precision/recall/F1 for each label"""
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("gold_labels and predicted_labels must have the same length")

    if labels is None:
        labels = sorted(set(gold_labels) | set(predicted_labels))

    results: dict[str, dict[str, float]] = {}

    for label in labels:
        true_positive = sum(1 for true_label, predicted_label in zip(gold_labels, predicted_labels) if true_label == label and predicted_label == label)
        false_positive = sum(1 for true_label, predicted_label in zip(gold_labels, predicted_labels) if true_label != label and predicted_label == label)
        false_negative = sum(1 for true_label, predicted_label in zip(gold_labels, predicted_labels) if true_label == label and predicted_label != label)

        precision = true_positive/ (true_positive+ false_positive) if (true_positive+ false_positive) > 0 else 0.0
        recall = true_positive/ (true_positive+ false_negative) if (true_positive+ false_negative) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        results[label] = {"precision": precision, "recall": recall, "f1": f1}

    return results

def confusion_matrix(
    gold_labels: list[str],
    predicted_labels: list[str],
    labels: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Compute a confusion matrix as a nested dict"""
    if labels is None:
        labels = sorted(set(gold_labels) | set(predicted_labels))

    matrix: dict[str, dict[str, int]] = {
        true_label: {predicted_label: 0 for predicted_label in labels} for true_label in labels
    }

    for true_label, predicted_label in zip(gold_labels, predicted_labels):
        if true_label in matrix and predicted_label in matrix[true_label]:
            matrix[true_label][predicted_label] += 1

    return matrix