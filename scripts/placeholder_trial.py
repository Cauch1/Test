"""Run the synthetic faulty pull-down through the real API N times and report
placeholder error rates. Synthetic data only; no client data is sent.

    export ANTHROPIC_API_KEY=...
    python scripts/placeholder_trial.py --runs 5 --out trial/

Each run writes its raw response and validation log under --out, and the
script prints a per-run table plus totals.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from interpret.channels import load_channel_info  # noqa: E402
from interpret.client import AnthropicClient, InterpretationError  # noqa: E402
from interpret.interpret import ROOT, build_payload, load_cfg  # noqa: E402
from interpret.summary import build_summary, write_summary  # noqa: E402
from interpret.validate import CommentaryValidationError, validate  # noqa: E402
from tests.synthetic import make_pulldown  # noqa: E402

KINDS = ["placeholder", "digit", "structure", "final_assertion", "naming", "number_word", "style"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="trial")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg()
    load = lambda p: yaml.safe_load((ROOT / p).read_text())
    summary = build_summary(
        make_pulldown(faults=True, noise_sd=0.05, seed=1),
        load_channel_info(ROOT / "config" / "channel_locations.yaml"),
        test_id="T0001", facility="CWT2", run_date=date(2026, 10, 30),
        spec=load("specs/bvs_2_7.yaml"), phase_cfg=load("config/phases.yaml")["phase_detection"],
        anomaly_cfg=load("config/anomaly_thresholds.yaml"))
    write_summary(summary, out)

    system = (ROOT / cfg["prompt_file"]).read_text()
    user = build_payload(summary, cfg)
    client = AnthropicClient(cfg)

    rows = []
    for i in range(1, args.runs + 1):
        try:
            result = client.complete(system, user)
        except InterpretationError as e:
            print(f"run {i}: API error: {e}")
            rows.append({"run": i, "api_error": str(e)})
            continue
        (out / f"run{i}_raw.json").write_text(result.raw)
        usage = json.loads(result.raw).get("usage", {})
        try:
            log, _ = validate(result.commentary, summary, rel=cfg["tolerance"]["relative"],
                              abs_tol=cfg["tolerance"]["absolute"], decimals=cfg["display_decimals"],
                              extra_allowlist=cfg.get("digit_allowlist", []))
        except CommentaryValidationError as e:
            log = e.log
        (out / f"run{i}_log.json").write_text(json.dumps(log, indent=2))
        rows.append({"run": i, "passed": log["passed"], "placeholders": log["placeholders_used"],
                     **log["counts"], "input_tokens": usage.get("input_tokens"),
                     "output_tokens": usage.get("output_tokens")})

    ok = [r for r in rows if "api_error" not in r]
    head = ["run", "passed", "placeholders", *KINDS, "input_tokens", "output_tokens"]
    print("\t".join(head))
    for r in ok:
        print("\t".join(str(r.get(h)) for h in head))
    if ok:
        total_ph = sum(r["placeholders"] for r in ok)
        print(f"\nRuns passed: {sum(r['passed'] for r in ok)}/{len(ok)}")
        print(f"Unknown keys: {sum(r['placeholder'] for r in ok)} of {total_ph} placeholders "
              f"({100 * sum(r['placeholder'] for r in ok) / max(total_ph, 1):.1f} %)")
        for k in ("digit", "structure", "final_assertion", "naming", "number_word", "style"):
            print(f"{k}: {sum(r[k] for r in ok)} across {len(ok)} runs")
    (out / "trial_summary.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
