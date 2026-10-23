"""Validate experimental AI Oracle reviews, source bindings, and manifest hashes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_ai_selective_context.ai_oracle import validate_ai_review
from medical_ai_selective_context.oracle_benchmark import (
    load_case_record,
    read_jsonl,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--ai-reviews", type=Path, required=True)
    parser.add_argument("--ai-manifest", type=Path, required=True)
    parser.add_argument("--cco-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    reviews = {row["caseId"]: row for row in read_jsonl(args.ai_reviews)}
    manifest = json.loads(args.ai_manifest.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    if manifest.get("experimental") is not True or manifest.get("complete") is not True:
        errors.append("ai_manifest_not_complete_or_experimental")
    for path in (args.cases, args.ai_reviews, args.base_manifest):
        expected = manifest.get("files", {}).get(path.name)
        if not expected or sha256_file(path) != expected:
            errors.append(f"manifest_hash_mismatch:{path.name}")
    case_ids = {row["caseId"] for row in cases}
    if len(cases) != 20 or set(reviews) != case_ids:
        errors.append("ai_review_case_set_mismatch")
    relevant = irrelevant = 0
    for case in cases:
        review = reviews.get(case["caseId"])
        if review is None:
            continue
        try:
            record = load_case_record(case, args.cco_dir)
            validate_ai_review(case=case, record=record, review=review)
            relevant += len(review["provisionalRelevantEvidence"])
            irrelevant += len(review["provisionalIrrelevantEvidence"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{case['caseId']}:{exc}")
    report = {
        "schemaVersion": "cco-ai-oracle-validation/0.1.0",
        "valid": not errors,
        "experimental": True,
        "cases": len(cases),
        "validatedReviews": len(reviews) if not errors else len(cases) - len(errors),
        "reviewStatusCounts": dict(sorted(Counter(
            row.get("reviewStatus", "missing") for row in reviews.values()
        ).items())),
        "relevantEvidence": relevant,
        "irrelevantEvidence": irrelevant,
        "humanVerified": False,
        "gold": False,
        "trainable": False,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
