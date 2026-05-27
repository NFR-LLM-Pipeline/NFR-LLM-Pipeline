from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from src import ClassificationResult, IdentificationResult, PipelineResult
from src.data_loader import get_project_ids, load_promise_dataset
from src.evaluation import (
    TARGET_CATEGORIES,
    evaluate_classification,
    evaluate_end_to_end,
    evaluate_identification,
)


PILOT_PROJECT_IDS: list[str] = ["6", "9"]
DEFAULT_RUNS_DIR = Path("outputs")
DEFAULT_DATA_PATH = Path("data/nfr.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/results/final_rq_results.json")
AGGREGATION_METHOD_NOTE = (
    "dataset-level per run, then mean ± sample standard deviation across runs"
)

_FILE_PATTERN = re.compile(
    r"^(?P<step>identify|classify|pipeline)_(?P<pid>[^_]+)_\d{8}_\d{6}\.json$"
)


# ---------- math ----------

def _stats(values: list[float]) -> dict[str, Any]:
    """Return ``{mean, std, per_run}`` with sample SD (Bessel)."""
    n = len(values)
    mean = sum(values) / n if n else 0.0
    if n < 2:
        std = 0.0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return {"mean": mean, "std": std, "per_run": list(values)}


# ---------- discovery -----------

def discover_runs(runs_dir: Path) -> list[Path]:
    """Return ``run-*`` subfolders sorted by numeric suffix."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    runs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run-")]

    def key(p: Path) -> tuple:
        m = re.fullmatch(r"run-(\d+)", p.name)
        return (0, int(m.group(1))) if m else (1, p.name)

    return sorted(runs, key=key)


def _latest_file(run_dir: Path, step: str, project_id: str) -> Path | None:
    """Most recent JSON file for ``(step, project_id)`` in ``run_dir``."""
    matches = [
        p
        for p in run_dir.glob(f"{step}_{project_id}_*.json")
        if (m := _FILE_PATTERN.match(p.name))
        and m.group("step") == step
        and m.group("pid") == project_id
    ]
    return max(matches, key=lambda p: p.name) if matches else None


# ---------- prediction loaders ----------

def _load_predictions(path: Path, step: str) -> Any:
    """Load predictions for ``step`` (``identify`` / ``classify`` / ``pipeline``)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    preds = data.get("predictions", {})

    def id_list(items: list[dict]) -> list[IdentificationResult]:
        return [
            IdentificationResult(
                req_id=str(p["req_id"]),
                predicted_is_nfr=bool(p["predicted_is_nfr"]),
            )
            for p in items
        ]

    def cls_list(items: list[dict]) -> list[ClassificationResult]:
        return [
            ClassificationResult(
                req_id=str(p["req_id"]),
                predicted_category=str(p["predicted_category"]),
            )
            for p in items
        ]

    if step == "identify":
        return id_list(preds.get("identification", []))
    if step == "classify":
        return cls_list(preds.get("classification", []))
    if step == "pipeline":
        return PipelineResult(
            project_id=str(preds.get("project_id", data.get("project_id", ""))),
            identification=id_list(preds.get("identification", [])),
            classification=cls_list(preds.get("classification", [])),
        )
    raise ValueError(f"Unknown step: {step!r}")


# ---------- aggregation helpers ----------

def _confusion_matrices_aggregated(matrices: list[dict]) -> tuple[dict, dict]:
    """Return ``(sum_cm, mean_cm)`` across runs as nested dicts of floats."""
    if not matrices:
        return {}, {}
    rows: set[str] = set()
    cols: set[str] = set()
    for m in matrices:
        for row, row_cols in m.items():
            rows.add(row)
            cols.update(row_cols.keys())
    n = len(matrices)
    summed = {
        row: {col: sum(m.get(row, {}).get(col, 0) for m in matrices) for col in cols}
        for row in rows
    }
    mean = {row: {col: v / n for col, v in row_cols.items()} for row, row_cols in summed.items()}
    return summed, mean


def _prf_per_run(
    per_run_data: list[dict], section: str, label: str, *, pipeline_inner: str | None = None
) -> dict[str, dict]:
    p, r, f = [], [], []
    for run in per_run_data:
        section_eval = run.get(section) or {}
        if pipeline_inner:
            section_eval = section_eval.get(pipeline_inner, {})
        m = section_eval.get("metrics", {}).get(label, {})
        if not m:
            continue
        p.append(float(m.get("precision", 0.0)))
        r.append(float(m.get("recall", 0.0)))
        f.append(float(m.get("f1", 0.0)))
    return {"precision": _stats(p), "recall": _stats(r), "f1": _stats(f)}


def _classification_errors_after_correct_identification(pipeline_eval: dict) -> int:
    """Gold NFRs that passed step 1 but were misclassified (not unclassified)."""
    if not pipeline_eval:
        return 0
    cm = pipeline_eval.get("classification", {}).get("confusion_matrix", {})
    eb = pipeline_eval.get("error_breakdown", {})
    correct = sum(cm.get(c, {}).get(c, 0) for c in list(TARGET_CATEGORIES) + ["other"])
    return max(
        0,
        eb.get("correctly_identified", 0)
        - correct
        - eb.get("unclassified_after_identification", 0),
    )


# ---------- per-run pooled evaluation ----------

def _evaluate_run(
    run_dir: Path,
    selected_project_ids: list[str],
    gold_requirements: list,
    missing_entries: list[dict[str, str]],
) -> dict[str, Any]:
    """Pool predictions across selected projects, then recompute dataset-level metrics."""
    pooled_id: list[IdentificationResult] = []
    pooled_cls: list[ClassificationResult] = []
    pooled_pl_id: list[IdentificationResult] = []
    pooled_pl_cls: list[ClassificationResult] = []

    for pid in selected_project_ids:
        for step, sink in (("identify", pooled_id), ("classify", pooled_cls)):
            path = _latest_file(run_dir, step, pid)
            if path is None:
                missing_entries.append({"run": run_dir.name, "project_id": pid, "step": step})
            else:
                sink.extend(_load_predictions(path, step))

        pl_path = _latest_file(run_dir, "pipeline", pid)
        if pl_path is None:
            missing_entries.append({"run": run_dir.name, "project_id": pid, "step": "pipeline"})
        else:
            pl = _load_predictions(pl_path, "pipeline")
            pooled_pl_id.extend(pl.identification)
            pooled_pl_cls.extend(pl.classification)

    pooled_pipeline = PipelineResult(
        project_id="ALL", identification=pooled_pl_id, classification=pooled_pl_cls
    )

    return {
        "run": run_dir.name,
        "identification": (
            evaluate_identification(gold_requirements, pooled_id) if pooled_id else None
        ),
        "classification": (
            evaluate_classification(gold_requirements, pooled_cls) if pooled_cls else None
        ),
        "pipeline": (
            evaluate_end_to_end(gold_requirements, pooled_pipeline) if pooled_pl_id else None
        ),
    }


# ---------- main entry point ----------

def aggregate_rq_results(
    runs_dir: Path,
    data_path: Path,
    selected_project_ids: list[str] | None = None,
) -> dict[str, Any]:

    runs_dir = Path(runs_dir)
    data_path = Path(data_path)

    all_requirements = load_promise_dataset(data_path)
    all_project_ids = get_project_ids(all_requirements)

    if selected_project_ids is None:
        selected_project_ids = [
            pid for pid in all_project_ids if pid not in PILOT_PROJECT_IDS
        ]
    else:
        selected_project_ids = [str(pid) for pid in selected_project_ids]

    selected_set = set(selected_project_ids)
    excluded_project_ids = [pid for pid in all_project_ids if pid not in selected_set]
    gold_requirements = [r for r in all_requirements if r.project_id in selected_set]

    missing_entries: list[dict[str, str]] = []
    per_run_data: list[dict[str, Any]] = [
        _evaluate_run(run_dir, selected_project_ids, gold_requirements, missing_entries)
        for run_dir in discover_runs(runs_dir)
    ]

    # Per-RQ aggregations
    nfr_metrics = _prf_per_run(per_run_data, "identification", "NFR")
    fr_metrics = _prf_per_run(per_run_data, "identification", "FR")
    id_cm_sum, id_cm_mean = _confusion_matrices_aggregated(
        [r["identification"].get("confusion_matrix", {}) for r in per_run_data if r.get("identification")]
    )

    rqb_target_metrics = {
        cat: _prf_per_run(per_run_data, "classification", cat) for cat in TARGET_CATEGORIES
    }
    cls_cm_sum, cls_cm_mean = _confusion_matrices_aggregated(
        [r["classification"].get("confusion_matrix", {}) for r in per_run_data if r.get("classification")]
    )

    rqc_target_metrics = {
        cat: _prf_per_run(per_run_data, "pipeline", cat, pipeline_inner="classification")
        for cat in TARGET_CATEGORIES
    }
    pl_cm_sum, pl_cm_mean = _confusion_matrices_aggregated(
        [r["pipeline"].get("classification", {}).get("confusion_matrix", {}) for r in per_run_data if r.get("pipeline")]
    )

    rqc_id_nfr = _prf_per_run(
        per_run_data, "pipeline", "NFR", pipeline_inner="identification"
    )
    rqc_id_fr = _prf_per_run(
        per_run_data, "pipeline", "FR", pipeline_inner="identification"
    )
    rqc_id_cm_sum, rqc_id_cm_mean = _confusion_matrices_aggregated(
        [
            r["pipeline"].get("identification", {}).get("confusion_matrix", {})
            for r in per_run_data
            if r.get("pipeline")
        ]
    )

    pl_runs = [r["pipeline"] for r in per_run_data if r.get("pipeline")]

    def _eb_stats(key: str) -> dict:
        return _stats([float(pl.get("error_breakdown", {}).get(key, 0)) for pl in pl_runs])

    error_propagation = {
        "missed_in_identification": _eb_stats("missed_in_identification"),
        "false_positive_count": _eb_stats("false_positive_count"),
        "unclassified_after_identification": _eb_stats("unclassified_after_identification"),
        "classification_errors_after_correct_identification": _stats(
            [float(_classification_errors_after_correct_identification(pl)) for pl in pl_runs]
        ),
    }

    return {
        "config": {
            "runs_dir": str(runs_dir),
            "data_path": str(data_path),
            "selected_projects": list(selected_project_ids),
            "excluded_projects": list(excluded_project_ids),
            "pilot_projects": list(PILOT_PROJECT_IDS),
            "aggregation_method": AGGREGATION_METHOD_NOTE,
            "n_runs": len(per_run_data),
            "run_names": [r["run"] for r in per_run_data],
            "missing_files": missing_entries,
        },
        "rqa_identification": {
            "title": "RQa: NFR identification",
            "nfr_metrics": nfr_metrics,
            "fr_metrics_supplementary": fr_metrics,
            "sum_confusion_matrix": id_cm_sum,
            "mean_confusion_matrix": id_cm_mean,
        },
        "rqb_classification_gold": {
            "title": "RQb: Classification given gold NFRs",
            "target_category_metrics": rqb_target_metrics,
            "sum_confusion_matrix": cls_cm_sum,
            "mean_confusion_matrix": cls_cm_mean,
        },
        "rqc_end_to_end_pipeline": {
            "title": "RQc: Full two-step pipeline",
            "identification_metrics": {
                "nfr": rqc_id_nfr,
                "fr_supplementary": rqc_id_fr,
                "sum_confusion_matrix": rqc_id_cm_sum,
                "mean_confusion_matrix": rqc_id_cm_mean,
            },
            "target_category_metrics": rqc_target_metrics,
            "error_propagation": error_propagation,
            "sum_confusion_matrix": pl_cm_sum,
            "mean_confusion_matrix": pl_cm_mean,
        },
    }


def save_rq_results(output: dict[str, Any], path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Write aggregated results to JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    return path
