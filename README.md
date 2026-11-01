# Post-test interpretation layer

This package adds two stages to the post-test pipeline, between the compute step and the deliverable build:

- **Stage A, structured summary.** Python computes the phases, metrics, spec criteria, and anomaly checks, and writes them to `<test_id>_summary.json`. This file is the only source of numbers.
- **Stage B, AI interpretation.** The Claude API reads the summary JSON, never the raw CSV, and returns draft commentary as templates. The model writes no numbers: it writes placeholder keys such as `{metric.time_to_target}`, and Python fills in the value and unit from the summary.

All commentary is a draft for engineer review. It appears in the PowerPoint and Word outputs under the heading "Draft commentary, engineer review required".

## Install

```bash
pip install -e ".[test]"
python -m pytest -q
```

## Run the pipeline

The standalone command takes a CSV with a `time_s` column and one column per standard channel name. This is the data after the existing YAML channel matching.

1. To write the summary only:

   ```bash
   python -m interpret.cli run.csv --test-id T0001 --facility CWT2 \
       --run-date 2026-10-30 --spec specs/bvs_2_7.yaml --out results/
   ```

2. To test Stage B with no network or API key, add `--interpret --offline`. A mock client writes commentary from the summary's own values, so every check and output runs.

3. To run Stage B against the API, set your key and add `--interpret`:

   ```bash
   export ANTHROPIC_API_KEY=...        # never put the key in code or config
   python -m interpret.cli run.csv ... --interpret
   ```

With `--interpret`, the output folder holds:

| File | Contents |
|---|---|
| `T0001_summary.json` | Stage A summary: phases, channels, metrics, criteria, anomalies |
| `T0001_interpretation_raw.json` | Raw API response, always written |
| `T0001_validation_log.json` | Every check result, always written |
| `T0001_interpretation.json` | Validated commentary, written only when the checks pass |
| `T0001_draft_commentary.pptx`, `.docx` | Draft slides and section |

If a check fails, the command exits with code 2 and lists each failed check with the sentence that caused it. The validated commentary file and the deliverables are not written.

### Connect to the existing pipeline

You need 3 small edits to the existing entry point and builders:

1. Where the argument parser is built, call `interpret.cli.add_cli_flags(parser)`.
2. After the compute step, call `interpret.cli.run_after_compute(args, df, test_id=..., facility=..., run_date=..., out_dir=..., pptx=prs, docx=doc)`. Pass the `Presentation` and `Document` objects your builders already hold.
3. In `config/channel_locations.yaml`, add `display_name`, `height`, `row`, and `side` for each standard channel name. The display name is what the report shows, for example "Row 1 left foot".

Without `--interpret`, `run_after_compute` returns immediately and the pipeline output is unchanged.

## Read the results

- **Criterion status** is `PASS`, `FAIL`, or `REVIEW_REQUIRED`. A channel with a flatline or drift that overlaps a metric's window is *suspect*. Each affected criterion is checked with all channels and with suspect channels excluded. If the two verdicts differ, the status is `REVIEW_REQUIRED` and you decide. To accept the excluded verdict automatically, set `auto_exclude: true` in `config/anomaly_thresholds.yaml`.
- **Anomaly IDs** (A01, A02, ...) are Python detections. The commentary explains each one once. Anything the model adds is listed separately and labelled "model-suggested, unverified".
- **Summary `notes`** list checks that were skipped, for example the drift check when the run has no stabilised phase.

## How the commentary is checked

The model returns templates. Before anything is filled in, the validator checks them in this order:

1. **Placeholder keys.** Every key must exist in the catalogue built from the summary. The catalogue is sent to the model with the summary, so it sees every available key.
2. **Digits.** Any digit left in the model's own text fails the draft, even a correct value. The only exceptions are IDs and references: the test ID, facility, spec reference, anomaly IDs, channel IDs, and any terms you add to `digit_allowlist` in `config/interpret.yaml`.
3. **Structure.** Each anomaly is explained once, each criterion ID is commented once, and the results run to 2 to 5 paragraphs.
4. **Fill.** Python replaces each key with its value and unit, using the decimals in `display_decimals` (for example 1 dp for degC and min).
5. **Final assertion.** The unit-aware number checker runs on the filled text. With correct fill code it can't fail; it guards against a fill or formatting bug.

These are logged as warnings and don't fail the draft:

- a channel's or anomaly's value quoted in a sentence that doesn't name the channel or anomaly
- spelled-out numbers ("two sensors")
- em dashes and missing value-unit spaces

Placeholder families:

| Family | Keys |
|---|---|
| Test | `{test.id}`, `{test.facility}`, `{test.spec_ref}`, `{test.run_date}` |
| Counts | `{counts.n_channels}`, `{counts.duration}`, `{counts.sample_interval}` |
| Phases | `{phase.<id>.start}`, `{phase.<id>.end}` |
| Metrics | `{metric.<id>}`, `.excluding_suspect`, `.suspect_channels`, `.window_start`, `.window_end` |
| Criteria | `{criteria.<id>.name}`, `.limit`, `.comparator`, `.measured`, `.margin`, `.status`, `.measured_excluding`, `.margin_excluding`, `.verdict_all`, `.verdict_excluding` |
| Channels | `{channel.<id>.name}`, `.location`, `.min`, `.max`, `.mean.<phase id>` |
| Anomalies | `{anomaly.<id>.channel}`, `.type`, `.start`, `.end`, `.evidence.<key>` |

A criterion's ID is its `id` in the spec YAML, or its metric name if it has no `id`.

## Add a new spec

You add a spec as a YAML file. You don't need code changes.

1. Copy `specs/bvs_2_7.yaml` to `specs/<spec>.yaml` and set `spec_ref`.
2. Set `cabin_reference` to the channels that drive phase detection.
3. Under `metrics`, define each metric with one of 3 types:

   | Type | Parameters | Result |
   |---|---|---|
   | `time_to_threshold` | `channels`, `threshold`, `direction` (`below` or `above`), `from_phase`, optional `smoothing_s`, `stratify_by` | Minutes from the phase start until the group mean crosses the threshold |
   | `window_mean` | `channels`, `window`, optional `stratify_by` | Mean over a window |
   | `group_difference` | `a`, `b`, `window`, optional `pair_by` | Group `a` minus group `b` |

   - Selectors such as `{height: breath, row: first}` match channel metadata. `row` accepts a number, `first`, or `last`.
   - A window is either `{phase: stabilised, last_min: 30}` (or `first_min`) or absolute `{start_min: 60, end_min: 70}`.
   - Use `stratify_by: [row]` for means and `pair_by: [row, side]` for gradients. If a suspect channel is excluded, these keep the remaining channels balanced, so a front-to-rear gradient doesn't bias the result.

4. Under `criteria`, link each criterion to a metric with `comparator` (`<=`, `>=`, `<`, `>`), `limit`, and optional `abs: true`.
5. Run `python -m pytest -q`, then run the command above with `--spec specs/<spec>.yaml` on a known run and check the summary JSON by hand.

`specs/bvs_2_7.yaml` holds placeholder limits. Replace them from the issued spec before you use it on a programme.

## Change the prompt

The system prompt is `prompts/system_interpretation.txt`. Edit the wording there; no code changes are needed. To use a different file, set `prompt_file` in `config/interpret.yaml`.

Keep these rules in the prompt, because the validator enforces them and a draft that breaks them is rejected:

- Write every value as a placeholder key from the list, and never as digits.
- Use the criterion ID, not its name, in `criteria_commentary`.
- Name the channel (`{channel.<id>.name}`) or anomaly in any sentence that quotes one channel's or one anomaly's value. Breaking this only produces a warning.
- Explain every anomaly ID once, and put new findings under `model_suggested_anomalies`.
- Write one sentence per criterion, using the criterion name exactly.

After you change the prompt, run one known test with `--interpret` and read the validation log.

## Configuration

| File | Controls |
|---|---|
| `config/interpret.yaml` | Model (default `claude-sonnet-5-5`), effort, refusal fallback, number tolerance, style checks |
| `config/anomaly_thresholds.yaml` | Flatline, drift, spike, and missing-data thresholds; drift reference window; suspect types; `auto_exclude` |
| `config/phases.yaml` | Phase inference from the cabin reference rate of change |
| `config/forbidden_terms.local.yaml` | Client and programme names that block a send. This file is not tracked; copy it from the `.example` file. |

## Known limits

- **A correctly filled value can still be the wrong one.** The validator proves every number came from the summary, but not that the model chose the right key. For example, the model could write `{metric.front_rear_gradient}` in a sentence about the foot-to-head gradient. Engineer review has to catch this; read each filled number against its sentence.
- **Drift is only checked from the start of its reference window.** It isn't checked during the pull-down.
- **A flatline on a channel without neighbours goes undetected.** Neighbour groups need at least 3 channels.

## Measure placeholder error rates

The trial script runs the synthetic faulty pull-down through the real API several times and reports error counts per run. It sends synthetic data only.

```bash
export ANTHROPIC_API_KEY=...
python scripts/placeholder_trial.py --runs 5 --out trial/
```
