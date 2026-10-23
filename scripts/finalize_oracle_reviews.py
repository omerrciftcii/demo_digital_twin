"""Validate human Oracle reviews and seal their current hash in the manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_ai_selective_context.oracle_benchmark import (
    load_case_record,
    read_jsonl,
    sha256_file,
    verified_oracle_annotations,
)
from medical_ai_selective_context.oracle_eval import build_oracle_abc_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cco-dir", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    reviews = {row["caseId"]: row for row in read_jsonl(args.reviews)}
    if set(reviews) != {row["caseId"] for row in cases}:
        raise SystemExit("case_review_set_mismatch")
    verified = 0
    approved = 0
    errors: list[str] = []
    for case in cases:
        review = reviews[case["caseId"]]
        if review.get("questionReviewStatus") == "approved":
            approved += 1
        if review.get("reviewStatus") != "verified":
            continue
        try:
            annotations = verified_oracle_annotations(review)
            record = load_case_record(case, args.cco_dir)
            build_oracle_abc_plan(
                record, case["question"], "B", oracle_annotations=annotations
            )
            verified += 1
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{case['caseId']}:{exc}")
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 1
    if args.require_all and (approved != len(cases) or verified != len(cases)):
        raise SystemExit("not_all_oracle_cases_are_human_verified")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    manifest["files"][args.cases.name] = sha256_file(args.cases)
    manifest["files"][args.reviews.name] = sha256_file(args.reviews)
    manifest["humanReviewSeal"] = {
        "sealedAt": datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(),
        "cases": len(cases), "approvedQuestions": approved,
        "verifiedOracleCases": verified,
        "reviewStatusCounts": dict(sorted(Counter(
            row.get("reviewStatus", "missing") for row in reviews.values()
        ).items())),
    }
    manifest["packageStatus"] = (
        "human_review_complete" if approved == verified == len(cases)
        else "human_review_partial"
    )
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "valid": True, "cases": len(cases), "approvedQuestions": approved,
        "verifiedOracleCases": verified,
        "packageStatus": manifest["packageStatus"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
