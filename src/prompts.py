"""Load and render prompt templates from external files."""

from pathlib import Path
from string import Template


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_template(filename: str) -> str:
    path = PROMPT_DIR / filename
    # Allow callers to pass filenames with or without a .txt extension
    if not path.exists():
        path = PROMPT_DIR / f"{filename}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    return path.read_text(encoding="utf-8")


def render_template(template: str, **kwargs: str) -> str:
    try:
        return Template(template).substitute(**kwargs)
    except KeyError as e:
        raise KeyError(f"Missing placeholder in template: {e}")


def load_and_render(filename: str, **kwargs: str) -> str:
    template = load_template(filename)
    return render_template(template, **kwargs)