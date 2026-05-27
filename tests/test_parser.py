"""Tests for src.parser."""

import pytest

from src.parser import (
    ParseError,
    format_requirements_for_prompt,
    parse_classification_response,
    parse_identification_response,
)


def test_format_requirements_for_prompt():
    requirements = [
        {"req_id": "R1", "text": "The system shall be fast."},
        {"req_id": "R2", "text": "The application must authenticate users."},
    ]

    expected = "1. [R1] The system shall be fast.\n2. [R2] The application must authenticate users."
    assert format_requirements_for_prompt(requirements) == expected


def test_parse_identification_response_lines():
    response = "R1: NFR\nR2: FR"
    results = parse_identification_response(response)

    labels = {r.req_id: r.predicted_is_nfr for r in results}
    assert labels["R1"] is True
    assert labels["R2"] is False


def test_parse_identification_response_rejects_invalid_formats():
    responses = [
        "R1 NFR",
        "R2 - FR",
        "[R3] other",
        "R3 => non-functional",
        "R1: maybe",
        "[{\"req_id\": \"R1\", \"predicted_is_nfr\": true}]",
    ]

    for response in responses:
        with pytest.raises(ParseError):
            parse_identification_response(response)


def test_parse_classification_response_lines():
    response = "R1: security\nR2: performance\nR3: other"
    results = parse_classification_response(response)

    labels = {r.req_id: r.predicted_category for r in results}
    assert labels["R1"] == "security"
    assert labels["R2"] == "performance"
    assert labels["R3"] == "other"


def test_parse_classification_response_rejects_invalid_formats():
    responses = [
        "R1 security",
        "R2 - performance",
        "[R3] other",
        "R3 => other",
        "R1: unknown-category",
        "[{\"req_id\": \"R1\", \"category\": \"security\"}]",
    ]

    for response in responses:
        with pytest.raises(ParseError):
            parse_classification_response(response)
