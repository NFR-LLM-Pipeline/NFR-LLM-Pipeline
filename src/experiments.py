"""CLI entry point for running experiments."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.data_loader import group_by_project, load_promise_dataset
from src.evaluation import (
    evaluate_classification,
    evaluate_end_to_end,
    evaluate_identification,
)
from dataclasses import asdict
from src.llm_runner import LLMConfig
from src.pipeline import classification, identification, run_pipeline

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

def save_results(results: dict, step: str, project_id: str) -> Path:
    """Save result file under outputs/."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{step}_{project_id}_{timestamp}.json"
    path = OUTPUTS_DIR / filename
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path

def run_experiment(args: argparse.Namespace) -> None:
    """Execute an experiment based on CLI arguments"""
    data_path = Path(args.data)
    requirements = load_promise_dataset(data_path)
    grouped = group_by_project(requirements)

    all_projects = args.projects if args.projects else sorted(grouped.keys())
    config = LLMConfig(model=args.model)

    pilot_ids = set(args.pilot_ids or [])

    # If prompt variant specified, override prompt template names
    prompt_suffix = f"_{args.prompt_variant}" if args.prompt_variant else ""
    # keep track of which prompt version was used
    variant_info = {
        "prompt_variant": args.prompt_variant or "default",
        "pilot_ids": sorted(pilot_ids),
    }

    # Run on all requested projects
    for project_id in all_projects:
        if project_id not in grouped:
            print(f"Warning: project '{project_id}' not found, skipping.")
            continue

        project_reqs = grouped[project_id]
        is_pilot = project_id in pilot_ids
        print(f"\n{'='*67}")
        pilot_tag = " [PILOT]" if is_pilot else ""
        print(f"Project: {project_id} ({len(project_reqs)} requirements){pilot_tag}")
        print(f"{'='*67}")

        if args.step == "identify":
            id_results = identification(project_reqs, config, f"identify{prompt_suffix}", is_pilot=is_pilot)
            eval_results = evaluate_identification(project_reqs, id_results)
            results = {
                "step": "identify",
                "project_id": project_id,
                "evaluation": eval_results,
                "predictions": {"identification": [asdict(r) for r in id_results]},
            }

        elif args.step == "classify":
            gold_nfrs = [r for r in project_reqs if r.is_nfr]
            cls_results = classification(gold_nfrs, config, f"classify{prompt_suffix}", is_pilot=is_pilot)
            eval_results = evaluate_classification(project_reqs, cls_results)
            results = {
                "step": "classify",
                "project_id": project_id,
                "evaluation": eval_results,
                "predictions": {"classification": [asdict(r) for r in cls_results]},
            }

        elif args.step == "pipeline":
            pipeline_result = run_pipeline(project_reqs, config, prompt_suffix, is_pilot=is_pilot)
            eval_results = evaluate_end_to_end(project_reqs, pipeline_result)
            # asdict will convert the PipelineResult dataclass (and nested dataclasses)
            results = {
                "step": "pipeline",
                "project_id": project_id,
                "evaluation": eval_results,
                "predictions": asdict(pipeline_result),
            }

        else:
            print(f"Unknown step: {args.step}")
            sys.exit(1)

        results["is_pilot"] = is_pilot
        results["variant_info"] = variant_info

        path = save_results(results, args.step, project_id)
        print(f"Results saved to: {path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NFR Classification Pipeline — Experiment Runner",
    )
    parser.add_argument(
        "--step",
        choices=["identify", "classify", "pipeline"],
        required=True,
        help="Which pipeline step to run.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/nfr.csv",
        help="Path to the PROMISE NFR dataset CSV.",
    )
    parser.add_argument(
        "--projects",
        nargs="*",
        default=None,
        help="Specific project IDs to evaluate. Defaults to all.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="LLM model name.",
    )

    parser.add_argument(
        "--prompt-variant",
        type=str,
        default=None,
        help="Prompt variant",
    )

    parser.add_argument(
        "--pilot-ids",
        nargs="*",
        default=None,
        help="Project IDs to mark as pilot projects in result metadata (used for prompt development).",
    )

    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main()