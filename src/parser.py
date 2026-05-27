"""Parse LLM text responses into structured prediction objects."""

import re
from src import ClassificationResult, IdentificationResult


class ParseError(ValueError):
    """Raised when an LLM response cannot be parsed."""


LINE_PATTERN = re.compile(r"^\s*(?P<req_id>[A-Za-z0-9_\-]+)\s*:\s*(?P<label>[A-Za-z]+)\s*$")

IDENTIFICATION_ALLOWED = {"nfr", "fr"}
CLASSIFICATION_ALLOWED = {
    "performance",
    "security",
    "maintainability",
    "other",
}


def parse_identification_response(response: str) -> list[IdentificationResult]:
    parsed_items = _parse_response(response)

    results: list[IdentificationResult] = []

    for req_id, label in parsed_items:
        label = label.lower().strip()

        if label not in IDENTIFICATION_ALLOWED:
            raise ParseError(f"Invalid label in identification: {label}")

        results.append(
            IdentificationResult(
                req_id=req_id,
                predicted_is_nfr=(label == "nfr"),
            )
        )

    return results


def parse_classification_response(response: str) -> list[ClassificationResult]:
    parsed_items = _parse_response(response)

    results: list[ClassificationResult] = []

    for req_id, label in parsed_items:
        label = label.lower().strip()

        if label not in CLASSIFICATION_ALLOWED:
            raise ParseError(f"Invalid category: {label}")

        results.append(
            ClassificationResult(
                req_id=req_id,
                predicted_category=label,
            )
        )

    return results


def format_requirements_for_prompt(requirements: list[dict]) -> str:
    """
    Canonical format used in ALL experiments.
    """

    return "\n".join(
        f"{i}. [{req['req_id']}] {req['text']}"
        for i, req in enumerate(requirements, start=1)
    )


def _parse_response(response: str) -> list[tuple[str, str]]:
    if not response or not response.strip():
        raise ParseError("Empty model response")

    parsed_lines: list[tuple[str, str]] = []
    for line in response.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        match = LINE_PATTERN.match(stripped)
        if not match:
            raise ParseError(f"Invalid line format: {stripped}")

        req_id = match.group("req_id")
        label = match.group("label").strip()
        parsed_lines.append((req_id, label))

    return parsed_lines
