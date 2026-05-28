# NFR-LLM Pipeline

## What is this repository for?

This repository contains a two-step GPT-4.1 pipeline for non-functional requirement (NFR) identification and classification using the PROMISE NFR dataset.

The pipeline performs the following tasks:

1. Identifies requirements as either:

   * Functional Requirements (FR)
   * Non-Functional Requirements (NFR)

2. Classifies NFRs into the following categories:

   * performance
   * security
   * maintainability
   * other

The pipeline is evaluated in three configurations:

* **Identification evaluation**
  FR vs NFR classification

* **Classification evaluation (gold NFRs)**
  Classification performance on ground-truth NFR statements only

* **End-to-end evaluation**
  Full pipeline evaluation where only predicted NFRs are classified

The repository also includes evaluation scripts, prompts, experiment outputs, and a Streamlit dashboard.

---

## How do I get set up?

### Prerequisites

* Python 3.10+
* OpenAI API key

### Setup

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root directory:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Running experiments

### Identification only

```bash
python -m src.experiments --step identify --data data/nfr.csv
```

### Classification on gold NFRs

```bash
python -m src.experiments --step classify --data data/nfr.csv
```

### Full end-to-end pipeline

```bash
python -m src.experiments --step pipeline --data data/nfr.csv
```

---

## Prompt variants

Alternative prompts can be selected using:

```bash
python -m src.experiments \
  --step pipeline \
  --data data/nfr.csv \
  --prompt-variant <name>`
```

Prompt templates are located in prompts folder.

## Running the dashboard

Launch the Streamlit dashboard with:

```bash
streamlit run dashboard.py
```


## Dataset

This repository includes a CSV-formatted version of the PROMISE NFR dataset originally published by:
Jane Cleland-Huang, Sepideh Mazrouee, Huang Liguoand Dan Port, “nfr”. Zenodo, Mar. 17, 2007. doi: 10.5281/zenodo.268542.
[Online]. Available: https://zenodo.org/records/268542
Original dataset license: CC BY 4.0.
The dataset has only been reformatted for use in this study.


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
