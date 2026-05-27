from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src import PipelineResult
from src.data_loader import load_promise_dataset
from src.evaluation import (
    TARGET_CATEGORIES,
    evaluate_classification,
    evaluate_end_to_end,
    evaluate_identification,
)
from src.rq_results import _latest_file, _load_predictions


PILOT_PROJECT_IDS: list[str] = ["6", "9"]
PILOT_VARIANT_DIRS: dict[str, str] = {
    "V1": "Pilot-V1",
    "V2": "Pilot-V2",
    "V3": "Pilot-V3",
}
DEFAULT_DATA_PATH = Path("data/nfr.csv")
DEFAULT_OUTPUTS_DIR = Path("outputs")
DEFAULT_PILOT_PROMPT_SENSITIVITY_OUTPUT_PATH = (
    DEFAULT_OUTPUTS_DIR / "results" / "pilot_prompt_sensitivity_results.json"
)


def target_macro_f1(evaluation_section: dict[str, Any]) -> float:
    """Macro-F1 over performance, security, and maintainability only."""
    metrics = evaluation_section.get("metrics", {})
    return sum(
        float(metrics.get(cat, {}).get("f1", 0.0)) for cat in TARGET_CATEGORIES
    ) / len(TARGET_CATEGORIES)


def _category_support(requirements: list) -> dict[str, int]:
    """Gold support per target category, excluding the residual 'other' class."""
    support = {cat: 0 for cat in TARGET_CATEGORIES}
    for req in requirements:
        cat = getattr(req, "category", None)
        if cat in support:
            support[cat] += 1
    return support


def _load_pooled_variant(
    variant_dir: Path,
    gold_requirements: list,
) -> dict[str, Any]:
    """Pool Projects 6 and 9 for one prompt variant and recompute metrics."""
    pooled_identification = []
    pooled_classification = []
    pooled_pipeline_identification = []
    pooled_pipeline_classification = []
    missing_files: list[dict[str, str]] = []

    for project_id in PILOT_PROJECT_IDS:
        identify_path = _latest_file(variant_dir, "identify", project_id)
        classify_path = _latest_file(variant_dir, "classify", project_id)
        pipeline_path = _latest_file(variant_dir, "pipeline", project_id)

        for step, path in (
            ("identify", identify_path),
            ("classify", classify_path),
            ("pipeline", pipeline_path),
        ):
            if path is None:
                missing_files.append(
                    {
                        "variant": variant_dir.name,
                        "project_id": project_id,
                        "step": step,
                    }
                )

        if identify_path is not None:
            pooled_identification.extend(_load_predictions(identify_path, "identify"))
        if classify_path is not None:
            pooled_classification.extend(_load_predictions(classify_path, "classify"))
        if pipeline_path is not None:
            pipeline_result = _load_predictions(pipeline_path, "pipeline")
            pooled_pipeline_identification.extend(pipeline_result.identification)
            pooled_pipeline_classification.extend(pipeline_result.classification)

    pooled_pipeline = PipelineResult(
        project_id="pilot-pooled-6-9",
        identification=pooled_pipeline_identification,
        classification=pooled_pipeline_classification,
    )

    identification_eval = (
        evaluate_identification(gold_requirements, pooled_identification)
        if pooled_identification
        else {}
    )
    classification_eval = (
        evaluate_classification(gold_requirements, pooled_classification)
        if pooled_classification
        else {}
    )
    pipeline_eval = (
        evaluate_end_to_end(gold_requirements, pooled_pipeline)
        if pooled_pipeline_identification
        else {}
    )

    return {
        "nfr_f1": identification_eval.get("metrics", {}).get("NFR", {}).get("f1", 0.0),
        "gold_macro_f1": target_macro_f1(classification_eval)
        if classification_eval
        else 0.0,
        "end_to_end_macro_f1": target_macro_f1(pipeline_eval.get("classification", {}))
        if pipeline_eval
        else 0.0,
        "identification": identification_eval,
        "classification_gold": classification_eval,
        "pipeline": pipeline_eval,
        "missing_files": missing_files,
    }


def _load_project_level_gold_classification(
    variant_dir: Path,
    project_id: str,
    project_requirements: list,
) -> dict[str, Any]:
    """Compute project-level gold-NFR classification F1 for one variant."""
    classify_path = _latest_file(variant_dir, "classify", project_id)
    if classify_path is None:
        return {
            "missing_file": {
                "variant": variant_dir.name,
                "project_id": project_id,
                "step": "classify",
            },
            "f1": {cat: 0.0 for cat in TARGET_CATEGORIES},
        }

    classification_predictions = _load_predictions(classify_path, "classify")
    classification_eval = evaluate_classification(project_requirements, classification_predictions)
    return {
        "f1": {
            cat: classification_eval["metrics"].get(cat, {}).get("f1", 0.0)
            for cat in TARGET_CATEGORIES
        },
        "support": _category_support(project_requirements),
        "classification": classification_eval,
    }


def aggregate_pilot_prompt_sensitivity(
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
    data_path: Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    """Aggregate the descriptive pilot prompt-sensitivity results."""
    outputs_dir = Path(outputs_dir)
    data_path = Path(data_path)

    all_requirements = load_promise_dataset(data_path)
    pilot_set = set(PILOT_PROJECT_IDS)
    pilot_requirements = [r for r in all_requirements if r.project_id in pilot_set]

    results: dict[str, Any] = {
        "config": {
            "data_path": str(data_path),
            "pilot_projects": list(PILOT_PROJECT_IDS),
            "prompt_variants": list(PILOT_VARIANT_DIRS.keys()),
            "macro_f1_labels": list(TARGET_CATEGORIES),
        },
        "pooled_projects_6_9": {},
        "project_level_gold_classification_f1": {},
        "missing_files": [],
    }

    for variant, folder in PILOT_VARIANT_DIRS.items():
        variant_dir = outputs_dir / folder
        pooled = _load_pooled_variant(variant_dir, pilot_requirements)
        results["pooled_projects_6_9"][variant] = pooled
        results["missing_files"].extend(pooled.get("missing_files", []))

        for project_id in PILOT_PROJECT_IDS:
            project_requirements = [
                r for r in all_requirements if r.project_id == project_id
            ]
            project_result = _load_project_level_gold_classification(
                variant_dir, project_id, project_requirements
            )
            results["project_level_gold_classification_f1"].setdefault(
                project_id, {}
            )[variant] = project_result
            if "missing_file" in project_result:
                results["missing_files"].append(project_result["missing_file"])

    return results


def save_pilot_prompt_sensitivity(
    output: dict[str, Any],
    path: Path = DEFAULT_PILOT_PROMPT_SENSITIVITY_OUTPUT_PATH,
) -> Path:
    """Write pilot prompt-sensitivity results to JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    return path
