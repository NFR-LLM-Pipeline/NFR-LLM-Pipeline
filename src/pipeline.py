"""The two-step NFR identification and classification pipeline.

This pipeline intentionally decomposes prediction into two sequential steps:
1. identification: detect which requirements are NFRs within a project.
2. classification: assign identified NFRs to performance, security, maintainability, or other.

This separation supports independent evaluation of identification performance and
end-to-end pipeline performance.
"""

from src import (
    ClassificationResult,
    IdentificationResult,
    PipelineResult,
    Requirement,
)
from src.llm_runner import LLMConfig, send_prompt
from src.parser import (
    format_requirements_for_prompt,
    parse_classification_response,
    parse_identification_response,
)
from src.prompts import load_and_render

def identification(
    requirements: list[Requirement],
    config: LLMConfig | None = None,
    prompt_name: str = "identify",
    is_pilot: bool = False,
) -> list[IdentificationResult]:
    """Step 1: Identify which requirements are NFRs"""
    project_id = requirements[0].project_id if requirements else ""
    req_dicts = [{"req_id": r.req_id, "text": r.text} for r in requirements]
    formatted = format_requirements_for_prompt(req_dicts)

    prompt = load_and_render(prompt_name, requirements=formatted)
    response = send_prompt(prompt, config, step="identify", project_id=project_id, is_pilot=is_pilot)

    return parse_identification_response(response)

def classification(
    requirements: list[Requirement],
    config: LLMConfig | None = None,
    prompt_name: str = "classify",
    is_pilot: bool = False,
) -> list[ClassificationResult]:
    """Step 2: Classify identified NFRs into categories"""
    project_id = requirements[0].project_id if requirements else ""
    req_dicts = [{"req_id": r.req_id, "text": r.text} for r in requirements]
    formatted = format_requirements_for_prompt(req_dicts)

    prompt = load_and_render(prompt_name, requirements=formatted)
    response = send_prompt(prompt, config, step="classify", project_id=project_id, is_pilot=is_pilot)

    return parse_classification_response(response)

def run_pipeline(
    requirements: list[Requirement],
    config: LLMConfig | None = None,
    prompt_suffix: str = "",
    is_pilot: bool = False,
) -> PipelineResult:
    """Run both steps for project"""
    if not requirements:
        raise ValueError("No requirements provided")

    project_id = requirements[0].project_id

    # Step 1: identification
    id_results = identification(requirements, config, f"identify{prompt_suffix}", is_pilot=is_pilot)

    # only send predicted NFRs to step 2
    predicted_nfr_ids = {r.req_id for r in id_results if r.predicted_is_nfr}
    nfr_requirements = [r for r in requirements if r.req_id in predicted_nfr_ids]

    # step 2: classification
    cls_results = []
    if nfr_requirements:
        cls_results = classification(nfr_requirements, config, f"classify{prompt_suffix}", is_pilot=is_pilot)

    return PipelineResult(
        project_id=project_id,
        identification=id_results,
        classification=cls_results,
    )