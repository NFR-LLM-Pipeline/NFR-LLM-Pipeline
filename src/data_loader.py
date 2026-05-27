"""Load and preprocess the PROMISE NFR dataset."""

import re
import pandas as pd
from pathlib import Path
from src import Requirement

CATEGORY_MAP: dict[str, str] = {
    "F": "F", # functional
    "PE": "performance",
    "SE": "security",
    "MN": "maintainability",
    # other NFR
    "US": "other",
    "SC": "other",
    "A": "other",
    "FT": "other",
    "L": "other",
    "LF": "other",
    "O": "other",
    "PO": "other",
}

REQUIRED_COLUMNS = {"ProjectID", "RequirementText", "class"}


def load_promise_dataset(filepath: Path) -> list[Requirement]:
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    df = pd.read_csv(filepath, keep_default_na=False)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    requirements: list[Requirement] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        raw_class = str(row["class"]).strip().upper()
        text = str(row["RequirementText"]).strip()
        project_id = str(row["ProjectID"]).strip()

        if raw_class not in CATEGORY_MAP:
            raise ValueError(
                f"Unknown PROMISE class label '{raw_class}' at row {idx + 1}"
            )
        if not text:
            raise ValueError(f"Empty requirement text at row {idx + 1}")
        if not project_id:
            raise ValueError(f"Empty ProjectID at row {idx + 1}")

        requirements.append(
            Requirement(
                req_id=f"R{idx + 1}",
                text=text,
                project_id=project_id,
                is_nfr=raw_class != "F",
                category=CATEGORY_MAP[raw_class],
            )
        )

    return requirements


def group_by_project(requirements: list[Requirement]) -> dict[str, list[Requirement]]:
    grouped: dict[str, list[Requirement]] = {}
    for req in requirements:
        grouped.setdefault(req.project_id, []).append(req)
    return grouped

def sort_key(project_id: str) -> tuple[str, int, str]:
    """Sort key that orders"""
    project_id = project_id.strip()

    match = re.fullmatch(r"([A-Za-z]*)(\d+)", project_id)
    if match:
        prefix, number = match.groups()
        return prefix, int(number), ""

    return project_id, -1, project_id


def get_project_ids(requirements: list[Requirement]) -> list[str]:
    return sorted({req.project_id for req in requirements}, key=sort_key)