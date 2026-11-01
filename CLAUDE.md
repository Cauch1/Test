# Post-test interpretation layer

Adds a structured summary (Stage A) and AI commentary (Stage B) to the
post-test climatic data pipeline. The brief is the source of the hard rules:
the model never calculates numbers, no raw CSV data or client names go to the
API, and every number in the commentary must trace to the summary JSON.

## Rules

- Run the full test suite (`python3 -m pytest -q`) before every commit. Never
  commit while any test fails. If a test fails, fix the code or, when the test
  itself is wrong, fix the test and say why in the commit message.
- Thresholds and spec limits belong in YAML (`config/`, `specs/`), never in code.
- Do not rewrite or rename the existing pipeline modules. List every existing
  file before changing it.

## Layout

- `interpret/`: the package. Stage A: schema, phases, metrics, criteria, anomalies, summary, adapter. Stage B: commentary, client, interpret, validate, deliverables, cli.
- `prompts/`: the Stage B system prompt.
- `config/`: phase detection, anomaly thresholds, channel location metadata.
- `specs/`: one YAML per test specification (metrics and criteria).
- `tests/`: pytest suite; `tests/synthetic.py` holds the closed-form fixture.
