# Post-test interpretation layer

This package adds two stages to the post-test pipeline, between the compute step and the deliverable build:

- **Stage A, structured summary.** Python computes the phases, metrics, spec criteria, and anomaly checks, and writes them to `<test_id>_summary.json`. This file is the only source of numbers.
- **Stage B, AI interpretation.** The Claude API reads the summary JSON, never the raw CSV, and returns draft commentary. A validator checks every number in the draft against the summary before the draft reaches a deliverable.

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
3. In `config/channel_locations.yaml`, add `height`, `row`, and `side` for each standard channel name.

Without `--interpret`, `run_after_compute` returns immediately and the pipeline output is unchanged.

## Read the results

- **Criterion status** is `PASS`, `FAIL`, or `REVIEW_REQUIRED`. A channel with a flatline or drift that overlaps a metric's window is *suspect*. Each affected criterion is checked with all channels and with suspect channels excluded. If the two verdicts differ, the status is `REVIEW_REQUIRED` and you decide. To accept the excluded verdict automatically, set `auto_exclude: true` in `config/anomaly_thresholds.yaml`.
- **Anomaly IDs** (A01, A02, ...) are Python detections. The commentary explains each one once. Anything the model adds is listed separately and labelled "model-suggested, unverified".
- **Summary `notes`** list checks that were skipped, for example the drift check when the run has no stabilised phase.

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

- Quote numbers exactly as they appear in the summary, with no calculation or unit conversion.
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

- **The number tolerance is loose for small values.** The floor of 0.1 units is set in the brief. On the synthetic run, 43 % of random values between 0 and 5 would match some summary value, so the validator rarely catches an invented small number. It is strong for values above 5.
- **Drift is only checked from the start of its reference window.** It isn't checked during the pull-down.
- **A flatline on a channel without neighbours goes undetected.** Neighbour groups need at least 3 channels.
