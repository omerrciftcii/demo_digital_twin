"""Evaluate source selection offline. Reports/SQLite stay in ignored local data."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_ai_selective_context.evaluation import evaluate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="Local JSONL annotation file")
    parser.add_argument("--semantic", action="store_true", help="Use cached local embeddings only")
    parser.add_argument("--token-budget", type=int, default=8000)
    parser.add_argument("--require-acceptance", action="store_true",
                        help="Exit 2 unless the reviewed retrieval acceptance gate passes")
    args = parser.parse_args()
    if args.token_budget < 1:
        parser.error("token budget must be positive")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    work_dir = PROJECT_ROOT / "data" / "evaluation" / ("run-" + run_id)
    try:
        report = evaluate_dataset(args.cases.resolve(), PROJECT_ROOT, work_dir,
                                  semantic=args.semantic, token_budget=args.token_budget)
        output = work_dir / "report.json"
        with output.open("x", encoding="utf-8") as target:
            json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        # Never print exception text/chains: they may contain source filenames,
        # question text, identifiers or schema values from a clinical document.
        print(f"Evaluation failed ({type(exc).__name__}); check local inputs and annotation schema.",
              file=sys.stderr)
        return 1
    print(json.dumps({
        "runId": run_id, "report": str(output.relative_to(PROJECT_ROOT)),
        "metrics": report["allCases"], "reviewedUniqueSubjects": report["reviewedUniqueSubjects"],
        "retrievalAcceptance": report["retrievalAcceptance"],
        "semanticStatus": report["index"]["semantic"], "modelApiCalls": 0,
    }, ensure_ascii=False, indent=2))
    return 2 if args.require_acceptance and not report["retrievalAcceptance"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
