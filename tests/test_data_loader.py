"""Tests for src.data_loader."""

from src import Requirement
from src.data_loader import get_project_ids, group_by_project, load_promise_dataset

from pathlib import Path


def _make_reqs() -> list[Requirement]:
    return [
        Requirement("R1", "Fast response", "P1", True, "performance"),
        Requirement("R2", "Login feature", "P1", False, "F"),
        Requirement("R3", "Encrypted data", "P2", True, "security"),
        Requirement("R4", "Search works", "P2", False, "F"),
        Requirement("R5", "Modular code", "P2", True, "maintainability"),
    ]


def test_group_by_project():
    grouped = group_by_project(_make_reqs())
    assert set(grouped.keys()) == {"P1", "P2"}
    assert len(grouped["P1"]) == 2
    assert len(grouped["P2"]) == 3


def test_get_project_ids():
    ids = get_project_ids(_make_reqs())
    assert ids == ["P1", "P2"]


def test_group_preserves_order():
    reqs = _make_reqs()
    grouped = group_by_project(reqs)
    assert grouped["P1"][0].req_id == "R1"
    assert grouped["P1"][1].req_id == "R2"


def test_load_promise_dataset_maps_labels(tmp_path: Path):
    csv_path = tmp_path / "nfr_test.csv"
    csv_path.write_text(
        """ProjectID,RequirementText,class
            P1,Login feature,F
            P1,Fast response,PE
            P2,Encrypted data,SE
            P2,Modular code,MN
            P3,Nice interface,US
        """,
        encoding="utf-8",
    )

    requirements = load_promise_dataset(csv_path)

    assert len(requirements) == 5

    assert requirements[0].req_id == "R1"
    assert requirements[0].is_nfr is False
    assert requirements[0].category == "F"

    assert requirements[1].is_nfr is True
    assert requirements[1].category == "performance"

    assert requirements[2].category == "security"
    assert requirements[3].category == "maintainability"

    assert requirements[4].is_nfr is True
    assert requirements[4].category == "other"


def test_get_project_ids_sorts_numeric_project_ids():
    reqs = [
        Requirement("R1", "Requirement 1", "1", True, "performance"),
        Requirement("R2", "Requirement 2", "10", True, "security"),
        Requirement("R3", "Requirement 3", "2", False, "F"),
    ]

    ids = get_project_ids(reqs)

    assert ids == ["1", "2", "10"]