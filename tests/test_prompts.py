"""Tests for src.prompts."""

from src.prompts import load_and_render, load_template, render_template


def test_load_template_identify():
    template = load_template("identify")
    assert "Requirements:" in template
    # Accept either $-style or {}-style placeholders depending on implementation
    assert ("$requirements" in template) or ("{requirements}" in template)


def test_load_template_classify():
    template = load_template("classify")
    # wording may vary slightly; ensure intent is present
    assert "classify" in template.lower()
    assert ("$requirements" in template) or ("{requirements}" in template)


def test_render_template_substitution():
    template = "Hello, $name! Requirements:\n$requirements"
    rendered = render_template(template, name="Tester", requirements="1. [R1] Example")

    assert "Hello, Tester!" in rendered
    assert "1. [R1] Example" in rendered


def test_load_and_render():
    # Load raw template and render with appropriate placeholder style
    template = load_template("identify")
    sample = "1. [R1] A sample requirement."
    if "{requirements}" in template:
        rendered = template.format(requirements=sample)
    else:
        rendered = load_and_render("identify", requirements=sample)

    assert "Requirements:" in rendered
    assert sample in rendered
    

def test_identify_prompt_substitutes_requirements():
    rendered = load_and_render(
        "identify",
        requirements="TEST_123",
    )

    assert "TEST_123" in rendered
    assert "$requirements" not in rendered


def test_classify_prompt_substitutes_requirements():
    rendered = load_and_render(
        "classify",
        requirements="TEST_123",
    )

    assert "TEST_123" in rendered
    assert "$requirements" not in rendered