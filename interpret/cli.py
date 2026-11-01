"""Command-line entry point, and the hook for the existing pipeline.

Standalone use (CSV already mapped to standard channel names, plus time_s):
    python -m interpret.cli data.csv --test-id T0001 --facility CWT2 \
        --run-date 2026-10-30 --spec specs/bvs_2_7.yaml --out results/
    ... add --interpret for Stage B, and --offline to use the mock client.

Existing pipeline: call add_cli_flags(parser) where its parser is built, and
run_after_compute(args, df, ...) after its compute step. Without --interpret,
Stage B does not run and the pipeline output is unchanged.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from .adapter import run_stage_a
from .client import InterpretationError
from .interpret import PrivacyError, interpret
from .validate import CommentaryValidationError


def add_cli_flags(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("AI interpretation")
    g.add_argument("--interpret", action="store_true",
                   help="Write the summary JSON and draft AI commentary for engineer review")
    g.add_argument("--offline", action="store_true",
                   help="With --interpret: use the mock client (no network, no API key)")
    g.add_argument("--spec", default="specs/bvs_2_7.yaml", help="Spec YAML for metrics and criteria")


def run_after_compute(args, df: pd.DataFrame, *, test_id: str, facility: str, run_date: date,
                      out_dir: str | Path, pptx=None, docx=None):
    """Stage A then Stage B. Returns the Commentary, or None without --interpret.

    pptx / docx: the existing builders' Presentation and Document objects; the
    draft section is appended to whichever are given.
    """
    if not args.interpret:
        return None
    _, summary_path = run_stage_a(df, test_id=test_id, facility=facility, run_date=run_date,
                                  spec_path=args.spec, out_dir=out_dir)
    commentary = interpret(summary_path, offline=args.offline)

    from .deliverables import add_draft_section_docx, add_draft_slides_pptx
    if pptx is not None:
        add_draft_slides_pptx(pptx, commentary)
    if docx is not None:
        add_draft_section_docx(docx, commentary)
    return commentary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m interpret.cli", description=__doc__.splitlines()[0])
    p.add_argument("csv", help="CSV with a time_s column and one column per standard channel name")
    p.add_argument("--test-id", required=True)
    p.add_argument("--facility", required=True, choices=["CWT1", "CWT2", "VTEOS", "VCC"])
    p.add_argument("--run-date", required=True, type=date.fromisoformat)
    p.add_argument("--out", required=True)
    add_cli_flags(p)
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    out = Path(args.out)
    if not args.interpret:
        # Standalone without --interpret: Stage A only.
        _, path = run_stage_a(df, test_id=args.test_id, facility=args.facility,
                              run_date=args.run_date, spec_path=args.spec, out_dir=out)
        print(f"Summary written: {path}")
        return 0

    from docx import Document
    from pptx import Presentation
    prs, doc = Presentation(), Document()
    try:
        run_after_compute(args, df, test_id=args.test_id, facility=args.facility,
                          run_date=args.run_date, out_dir=out, pptx=prs, docx=doc)
    except (CommentaryValidationError, PrivacyError, InterpretationError) as e:
        print(f"Interpretation stopped: {type(e).__name__}\n{e}", file=sys.stderr)
        return 2
    prs.save(out / f"{args.test_id}_draft_commentary.pptx")
    doc.save(out / f"{args.test_id}_draft_commentary.docx")
    print(f"Summary, commentary and draft deliverables written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
