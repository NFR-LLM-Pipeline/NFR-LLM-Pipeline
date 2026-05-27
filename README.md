# NFR-LLM Pipeline

## Pipeline Overview

The code implements a two-step LLM pipeline for PROMISE dataset requirements.

1. **Identification step**
   - All requirements for a given project are sent to the model in one prompt.
   - The model predicts whether each requirement is a functional requirement (`FR`) or a non-functional requirement (`NFR`).
   - This step is evaluated independently with precision/recall/F1 for `NFR` vs `FR`.

2. **Classification step**
   - Only the requirements identified as NFRs are sent back to the model.
   - The model assigns one of: `performance`, `security`, `maintainability`, or `other`.
   - Classification is evaluated in two ways:
     - Directly on gold NFRs to measure classification performance independently of identification errors.
     - End-to-end through the full pipeline, where only identified NFRs are classified.

## Prompting strategy

- The pipeline uses zero-shot prompting in both steps.
- In the identification step, the model receives all requirements for a single project and is instructed to label each statement as `FR` or `NFR`.
- In the classification step, only the requirements identified as NFRs are sent back to the model and the model is instructed to label each one as `performance`, `security`, `maintainability`, or `other`.
- No labeled requirement examples are provided in the prompts; only output formatting guidance is included.
- Prompt templates are defined in `prompts/identify.txt` and `prompts/classify.txt`.
- Prompt variants can be controlled via `--prompt-variant` for sensitivity analysis.

## Prompt development and pilot phase

Prompt templates are intended to be developed during a pilot phase using 2–3 projects from the dataset. Pilot project IDs and prompt changes should be documented separately in the thesis appendix. Results should be reported both with and without pilot-based prompt refinement to assess any bias introduced by prompt optimization.

Use the `--pilot-ids` flag to tag specific projects as pilot in the result metadata. Every result JSON will contain an `is_pilot: true/false` field and the full pilot set in `variant_info.pilot_ids`.

```bash
# Run identification on projects 1 and 2, tagged as pilot projects
python -m src.experiments --step identify --data data/nfr.csv --projects 1 2 --pilot-ids 1 2

# Full pipeline run across all projects, with projects 1 and 2 tagged as pilot
python -m src.experiments --step pipeline --data data/nfr.csv --pilot-ids 1 2
```

## Reproducibility

The repository is structured to support reproducibility:

- Prompt texts are stored in external files under `prompts/`.
- LLM inputs and outputs are logged to `outputs/raw_responses/` by `src/llm_runner.py`.
- The model configuration is deterministic by default (`temperature=0.0`).
- Evaluation is separated by step: identification, classification on gold NFRs, and full end-to-end pipeline.
- The parser in `src/parser.py` only accepts explicit structured outputs or strict line formats, and raises a parse error for malformed model responses.
- Partial parsing is not allowed; any invalid line or unrecognized label causes the response to be rejected.
- Experiment metadata records `prompt_variant` and `project_id` for reproducible reruns.

## Running experiments

Example commands:

```bash
python -m src.experiments --step identify --data data/nfr.csv
python -m src.experiments --step classify --data data/nfr.csv
python -m src.experiments --step pipeline --data data/nfr.csv
```

To use a variant prompt:

```bash
python -m src.experiments --step pipeline --data data/nfr.csv --prompt-variant alt1
```

## Setup

- Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run tests:

```bash
python -m pytest
```

## CLI Usage

- The main experiment runner is `src.experiments` and supports three steps:

```bash
python -m src.experiments --step identify --data data/nfr.csv
python -m src.experiments --step classify --data data/nfr.csv
python -m src.experiments --step pipeline --data data/nfr.csv
```

- Optional flags:
   - `--prompt-variant <name>` - select an alternate prompt template variant stored in `prompts/`.
   - `--pilot-ids <id>` - tag the listed project IDs as pilot projects in result metadata.

- Quick example: run the full pipeline with an alternate prompt

```bash
python -m src.experiments --step pipeline --data data/nfr.csv --prompt-variant alt1
```


## Evaluation

The evaluation code is in `src/evaluation.py` and supports:
- `evaluate_identification(...)`
- `evaluate_classification(...)`
- `evaluate_end_to_end(...)`

Metrics include precision, recall, F1, confusion matrices, and analysis of the residual `other` class.

## Test coverage

Unit tests currently cover:
- parser behavior in `tests/test_parser.py`
- prompt loading and rendering in `tests/test_prompts.py`
- pipeline error counting in `tests/test_pipeline.py`
- metric calculations in `tests/test_metrics.py`

## Dataset

This repository includes a CSV-formatted version of the PROMISE NFR dataset originally published by:
Jane Cleland-Huang, Sepideh Mazrouee, Huang Liguoand Dan Port, “nfr”. Zenodo, Mar. 17, 2007. doi: 10.5281/zenodo.268542.
[Online]. Available: https://zenodo.org/records/268542
Original dataset license: CC BY 4.0.
The dataset has only been reformatted for use in this study.

## Streamlit Dashboard

```bash
streamlit run dashboard.py
```

## Project Structure

| File | Role |
|------|------|
| `src/__init__.py` | Core data models used across the project |
| `src/data_loader.py` | Load PROMISE dataset, group by project |
| `src/prompts.py` | Load and render prompt templates from `prompts/` |
| `src/llm_runner.py` | Send prompts to LLM API and log raw responses|
| `src/parser.py` | Parse LLM text responses into structured predictions |
| `src/metrics.py` | Precision, recall, F1, confusion matrix |
| `src/evaluation.py` | Evaluation logic for identification, classification, and end-to-end results |
| `src/pipeline.py` | Execute identification, classification, and the full pipeline |
| `src/experiments.py` | CLI entry point for running experiments |
| `src/rq_results.py` | Aggregate final RQ level results across repeated runs |
| `src/pilot_prompt_sensitivity.py` | Aggregate prompt-sensitivity results for the pilot phase |
| `prompts/identify.txt` | Prompt template for NFR identification |
| `prompts/classify.txt` | Prompt template for NFR classification |
| `tests/` | Unit tests |
| `dashboard.py` | Streamlit dashboard |
