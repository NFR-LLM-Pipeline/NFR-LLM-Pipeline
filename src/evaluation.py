"""Evaluate pipeline predictions against gold labels."""

from src import (
    ClassificationResult,
    IdentificationResult,
    PipelineResult,
    Requirement,
)
from src.metrics import confusion_matrix, calc_metrics

# Labels
IDENTIFICATION_LABELS = ["NFR", "FR"]
# target NFR categories
TARGET_CATEGORIES = ["performance", "security", "maintainability"]
# classification labels
ALL_CLASSIFICATION_LABELS = ["performance", "security", "maintainability", "other"]

def analyze_other(gold_labels: list[str], predicted_labels: list[str]) -> dict:
    """Analyze the residual "other" class per proposal requirements"""
    pred_other_count = sum(1 for p in predicted_labels if p == "other")
    true_other_count = sum(1 for t in gold_labels if t == "other")

    # Target category statements predicted as "other"
    target_to_other: dict[str, int] = {}
    for t, p in zip(gold_labels, predicted_labels):
        if t in TARGET_CATEGORIES and p == "other":
            target_to_other[t] = target_to_other.get(t, 0) + 1

    # True "other" statements predicted as target categories
    other_to_target: dict[str, int] = {}
    for t, p in zip(gold_labels, predicted_labels):
        if t == "other" and p in TARGET_CATEGORIES:
            other_to_target[p] = other_to_target.get(p, 0) + 1

    return {
        "predicted_other_count": pred_other_count,
        "true_other_count": true_other_count,
        "target_confused_as_other": target_to_other,
        "other_confused_as_target": other_to_target,
    }

def evaluate_identification(
    gold: list[Requirement],
    predictions: list[IdentificationResult],
) -> dict:
    """RQa: Evaluate NFR identification (NFR vs FR)"""
    pred_map = {p.req_id: p for p in predictions}

    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    missing_ids: list[str] = []

    for req in gold:
        gold_labels.append("NFR" if req.is_nfr else "FR")
        if req.req_id in pred_map:
            predicted_labels.append("NFR" if pred_map[req.req_id].predicted_is_nfr else "FR")
        else:
            # Model gave no response for this requirement
            predicted_labels.append("FR")
            missing_ids.append(req.req_id)

    n_nfr = sum(1 for t in gold_labels if t == "NFR")
    n_fr = sum(1 for t in gold_labels if t == "FR")

    return {
        "metrics": calc_metrics(gold_labels, predicted_labels, IDENTIFICATION_LABELS),
        "confusion_matrix": confusion_matrix(gold_labels, predicted_labels, IDENTIFICATION_LABELS),
        "n_samples": len(gold_labels),
        "n_nfr": n_nfr,
        "n_fr": n_fr,
        "no_fr_in_project": n_fr == 0,
        "missing_responses": len(missing_ids),
        "missing_ids": missing_ids,
    }

def evaluate_classification(
    gold: list[Requirement],
    predictions: list[ClassificationResult],
) -> dict:
    """RQb: Evaluate NFR classification on gold NFR requirements only"""
    pred_map = {p.req_id: p.predicted_category for p in predictions}
    gold_nfrs = [r for r in gold if r.is_nfr]

    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    missing_ids: list[str] = []

    for req in gold_nfrs:
        gold_labels.append(req.category)
        if req.req_id in pred_map:
            predicted_labels.append(pred_map[req.req_id])
        else:
            # Model gave no classification for this gold NFR
            predicted_labels.append("_unclassified")
            missing_ids.append(req.req_id)

    # P/R/F1 only for the three target categories
    target_metrics = calc_metrics(gold_labels, predicted_labels, TARGET_CATEGORIES)
    
    confusion_matrix_labels = ALL_CLASSIFICATION_LABELS + (["_unclassified"] if missing_ids else [])
    full_confusion_matrix = confusion_matrix(gold_labels, predicted_labels, confusion_matrix_labels)

    # Separate "other" class analysis
    other_analysis = analyze_other(gold_labels, predicted_labels)

    return {
        "metrics": target_metrics,
        "confusion_matrix": full_confusion_matrix,
        "other_analysis": other_analysis,
        "n_samples": len(gold_labels),
        "missing_responses": len(missing_ids),
        "missing_ids": missing_ids,
    }

def evaluate_end_to_end(
    gold: list[Requirement],
    pipeline_result: PipelineResult,
) -> dict:
    """RQc: evaluate the whole pipeline, not just one step"""
    identification_eval = evaluate_identification(gold, pipeline_result.identification)

    identified_nfr_ids = {
        p.req_id for p in pipeline_result.identification if p.predicted_is_nfr
    }
    classification_map = {
        p.req_id: p.predicted_category for p in pipeline_result.classification
    }
    gold_nfrs = [r for r in gold if r.is_nfr]
    gold_nfr_ids = {r.req_id for r in gold_nfrs}
    gold_map = {r.req_id: r for r in gold}

    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    missed_nfr_ids: list[str] = []
    unclassified_nfr_ids: list[str] = []

    # go through the real NFRs first
    for req in gold_nfrs:
        gold_labels.append(req.category)
        if req.req_id not in identified_nfr_ids:
            # missed already in step 1
            predicted_labels.append("_missed")
            missed_nfr_ids.append(req.req_id)
        elif req.req_id not in classification_map:
            # identified as NFR but model gave no classification output
            predicted_labels.append("_unclassified")
            unclassified_nfr_ids.append(req.req_id)
        else:
            # Correctly passed through both steps
            predicted_labels.append(classification_map[req.req_id])

    # False positives: FRs incorrectly identified as NFRs
    false_positives: list[dict[str, str]] = []
    for req_id in sorted(identified_nfr_ids):
        if req_id in gold_nfr_ids:
            continue
        true_class = gold_map[req_id].category if req_id in gold_map else "unknown"
        pred_category = classification_map.get(req_id, "_unclassified")

        gold_labels.append(true_class)
        predicted_labels.append(pred_category)

        false_positives.append({
            "req_id": req_id,
            "true_class": true_class,
            "predicted_category": pred_category,
        })

    # only report metrics for the 3 target classes
    target_metrics = calc_metrics(gold_labels, predicted_labels, TARGET_CATEGORIES) if gold_labels else {}

    # Confusion matrix labels
    confusion_matrix_labels = list(ALL_CLASSIFICATION_LABELS)
    for label in gold_labels + predicted_labels:
        if label not in confusion_matrix_labels:
            confusion_matrix_labels.append(label)
    full_confusion_matrix = confusion_matrix(gold_labels, predicted_labels, confusion_matrix_labels) if gold_labels else {}
    other_analysis = analyze_other(gold_labels, predicted_labels) if gold_labels else {}

    classification_eval = {
        "metrics": target_metrics,
        "confusion_matrix": full_confusion_matrix,
        "other_analysis": other_analysis,
        "n_samples": len(gold_labels),
    }

    error_breakdown = {
        "total_gold_nfrs": len(gold_nfrs),
        "correctly_identified": len(gold_nfrs) - len(missed_nfr_ids),
        "missed_in_identification": len(missed_nfr_ids),
        "missed_nfr_ids": missed_nfr_ids,
        "unclassified_after_identification": len(unclassified_nfr_ids),
        "unclassified_nfr_ids": unclassified_nfr_ids,
        "false_positives": false_positives,
        "false_positive_count": len(false_positives),
    }

    return {
        "identification": identification_eval,
        "classification": classification_eval,
        "error_breakdown": error_breakdown,
    }