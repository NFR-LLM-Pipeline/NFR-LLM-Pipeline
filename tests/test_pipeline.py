"""Tests for pipeline evaluation logic and two-step behavior."""

from src import ClassificationResult, IdentificationResult, PipelineResult, Requirement
from src.evaluation import evaluate_classification, evaluate_end_to_end
from src import pipeline as pipeline_module


def make_req(req_id: str, text: str, is_nfr: bool, category: str, project_id: str = "P1") -> Requirement:
    return Requirement(req_id=req_id, text=text, project_id=project_id, is_nfr=is_nfr, category=category)


def test_evaluate_classification_on_gold_nfrs():
    gold = [
        make_req("R1", "Encrypt all passwords.", True, "security"),
        make_req("R2", "Generate monthly reports.", False, "F"),
        make_req("R3", "Respond within 2 seconds.", True, "performance"),
    ]

    predictions = [
        ClassificationResult(req_id="R1", predicted_category="security"),
        ClassificationResult(req_id="R3", predicted_category="performance"),
    ]

    result = evaluate_classification(gold, predictions)

    assert result["metrics"]["security"]["precision"] == 1.0
    assert result["metrics"]["performance"]["recall"] == 1.0
    assert result["missing_responses"] == 0
    assert result["n_samples"] == 2


def test_evaluate_end_to_end_counts_missed_and_false_positives():
    gold = [
        make_req("R1", "Encrypt all passwords.", True, "security"),
        make_req("R2", "Generate monthly reports.", False, "F"),
        make_req("R3", "Respond within 2 seconds.", True, "performance"),
    ]

    pipeline_result = PipelineResult(
        project_id="P1",
        identification=[
            IdentificationResult(req_id="R1", predicted_is_nfr=True),
            IdentificationResult(req_id="R2", predicted_is_nfr=True),
            IdentificationResult(req_id="R3", predicted_is_nfr=False),
        ],
        classification=[
            ClassificationResult(req_id="R1", predicted_category="security"),
            ClassificationResult(req_id="R2", predicted_category="other"),
        ],
    )

    result = evaluate_end_to_end(gold, pipeline_result)

    assert result["identification"]["missing_responses"] == 0
    assert result["error_breakdown"]["missed_in_identification"] == 1
    assert result["error_breakdown"]["false_positive_count"] == 1
    assert any(fp["req_id"] == "R2" for fp in result["error_breakdown"]["false_positives"])
    assert result["classification"]["n_samples"] == 3


def test_end_to_end_false_positives_lower_target_precision():
    gold = [
        make_req("R1", "Fast response", True, "performance"),
        make_req("R2", "Login works", False, "F"),
    ]

    pipeline = PipelineResult(
        project_id="P1",
        identification=[
            IdentificationResult("R1", True),
            IdentificationResult("R2", True),  # false positive in step 1
        ],
        classification=[
            ClassificationResult("R1", "performance"),
            ClassificationResult("R2", "performance"),  # false positive for performance
        ],
    )

    result = evaluate_end_to_end(gold, pipeline)
    perf = result["classification"]["metrics"]["performance"]

    assert perf["precision"] == 0.5   # 1 true performance, 1 false performance
    assert perf["recall"] == 1.0      # the only real performance requirement was found


def test_run_pipeline_flow_and_pilot_propagation(monkeypatch):
    reqs = [
        make_req("R1", "Encrypt passwords.", True, "security"),
        make_req("R2", "Monthly report.", False, "F"),
    ]

    captured_calls: list[dict] = []

    def fake_send_prompt(prompt, config=None, step="", project_id="", is_pilot=False):
        captured_calls.append(
            {
                "step": step,
                "project_id": project_id,
                "is_pilot": is_pilot,
            }
        )
        if step.startswith("identify"):
            return "R1: NFR\nR2: FR"
        return "R1: security"

    monkeypatch.setattr(pipeline_module, "send_prompt", fake_send_prompt)

    result = pipeline_module.run_pipeline(reqs, is_pilot=True)

    assert len(captured_calls) == 2
    assert {call["step"] for call in captured_calls} == {"identify", "classify"}
    assert all(call["is_pilot"] is True for call in captured_calls)

    assert result.project_id == "P1"

    assert len(result.identification) == 2
    assert result.identification[0].req_id == "R1"
    assert result.identification[0].predicted_is_nfr is True
    assert result.identification[1].req_id == "R2"
    assert result.identification[1].predicted_is_nfr is False

    assert len(result.classification) == 1
    assert result.classification[0].req_id == "R1"
    assert result.classification[0].predicted_category == "security"