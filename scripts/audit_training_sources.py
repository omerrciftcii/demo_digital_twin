"""Audit text CCOs against a raw fact snapshot without printing clinical data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_ai_selective_context.training_sources import (
    SourceAuditError,
    audit_text_cco_and_facts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cco-dir", required=True, type=Path)
    parser.add_argument("--facts-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional aggregate JSON report. No clinical text or subject ID is written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_text_cco_and_facts(args.cco_dir.resolve(), args.facts_dir.resolve())
    except SourceAuditError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
