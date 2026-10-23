"""Validate Oracle pilot identity, evidence, review gates, and artifact hashes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_ai_selective_context.oracle_benchmark import (
    load_case_record,
    read_jsonl,
    sha256_file,
)
from medical_ai_selective_context.text_document import build_evidence_spans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--cco-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases_path = args.package_dir / "oracle_cases.jsonl"
    reviews_path = args.package_dir / "oracle_evidence_reviews.jsonl"
    manifest_path = args.package_dir / "oracle_run_manifest.json"
    cases = read_jsonl(cases_path)
    reviews = read_jsonl(reviews_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    if len(cases) != 20 or len({row["caseId"] for row in cases}) != 20:
        errors.append("pilot_must_have_20_unique_cases")
    if len({row["sourceFile"] for row in cases}) != len(cases):
        errors.append("pilot_sources_must_be_distinct")
    review_by_id = {row["caseId"]: row for row in reviews}
    if set(review_by_id) != {row["caseId"] for row in cases}:
        errors.append("case_review_set_mismatch")
    evidence_refs = 0
    for case in cases:
        try:
            record = load_case_record(case, args.cco_dir)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{case.get('caseId', 'unknown')}:source:{type(exc).__name__}")
            continue
        available = {span.evidence_id: span.location() for span in build_evidence_spans(record)}
        relevant = case.get("candidateEvidence", [])
        irrelevant = case.get("candidateIrrelevantEvidence", [])
        relevant_ids = {item.get("evidenceId") for item in relevant}
        irrelevant_ids = {item.get("evidenceId") for item in irrelevant}
        if relevant_ids & irrelevant_ids:
            errors.append(f"{case['caseId']}:candidate_evidence_overlap")
        for item in relevant + irrelevant:
            evidence_refs += 1
            if available.get(item.get("evidenceId")) != item.get("sourceSpan"):
                errors.append(f"{case['caseId']}:evidence_source_mismatch")
        review = review_by_id.get(case["caseId"], {})
        if review.get("snapshotId") != case["snapshotId"]:
            errors.append(f"{case['caseId']}:review_snapshot_mismatch")
        if review.get("contentHash") != case["contentHash"]:
            errors.append(f"{case['caseId']}:review_content_hash_mismatch")
        if review.get("question") != case["question"]:
            errors.append(f"{case['caseId']}:review_question_mismatch")
        if case.get("trainable") is not False or review.get("trainable") is not False:
            errors.append(f"{case['caseId']}:unreviewed_item_marked_trainable")
        if review.get("reviewStatus") == "pending" and review.get("gold") is not False:
            errors.append(f"{case['caseId']}:pending_item_marked_gold")
    for name, expected in manifest.get("files", {}).items():
        path = args.package_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"artifact_hash_mismatch:{name}")
    for name, expected in manifest.get("promptTemplates", {}).items():
        path = ROOT / "oracle_prompts" / name
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"prompt_hash_mismatch:{name}")
    report = {
        "schemaVersion": "cco-oracle-package-validation/0.1.0",
        "valid": not errors,
        "cases": len(cases),
        "uniqueCcos": len({row.get("sourceFile") for row in cases}),
        "evidenceReferencesChecked": evidence_refs,
        "questionReviewStatus": dict(sorted(Counter(
            row.get("questionReviewStatus", "missing") for row in reviews
        ).items())),
        "evidenceReviewStatus": dict(sorted(Counter(
            row.get("reviewStatus", "missing") for row in reviews
        ).items())),
        "oracleBReadyCases": sum(
            row.get("reviewStatus") == "verified"
            and row.get("questionReviewStatus") == "approved"
            for row in reviews
        ),
        "sourceCcosModified": False,
        "realGemmaBenchmarkExecuted": False,
        "errors": errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
