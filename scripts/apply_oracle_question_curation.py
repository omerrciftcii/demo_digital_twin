"""Apply language-only question curation while preserving Luna candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_ai_selective_context.oracle_benchmark import read_jsonl, sha256_file, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--curation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    curation = json.loads(args.curation.read_text(encoding="utf-8-sig"))
    if curation.get("clinicalLabelsChanged") is not False:
        raise SystemExit("curation_must_not_change_clinical_labels")
    replacements = curation["questions"]
    cases = read_jsonl(args.cases)
    reviews = read_jsonl(args.reviews)
    case_ids = {row["caseId"] for row in cases}
    if case_ids != set(replacements) or case_ids != {row["caseId"] for row in reviews}:
        raise SystemExit("curation_case_set_mismatch")
    for row in cases:
        row.setdefault("lunaQuestionCandidate", row["question"])
        row["question"] = replacements[row["caseId"]].strip()
        row["questionCuration"] = {
            "source": curation["curationSource"],
            "scope": "language_only",
            "clinicalLabelsChanged": False,
            "reviewStatus": "pending",
        }
        row["reviewStatus"] = "pending"
        row["trainable"] = False
    case_by_id = {row["caseId"]: row for row in cases}
    for row in reviews:
        case = case_by_id[row["caseId"]]
        row.setdefault("lunaQuestionCandidate", case["lunaQuestionCandidate"])
        row["question"] = case["question"]
        row["questionReviewStatus"] = "pending"
        row["reviewStatus"] = "pending"
        row["gold"] = False
        row["trainable"] = False
    write_jsonl(args.cases, cases)
    write_jsonl(args.reviews, reviews)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    manifest["pilot"]["questionCuration"] = {
        "source": curation["curationSource"],
        "scope": "language_only",
        "clinicalLabelsChanged": False,
        "reviewStatus": "pending",
    }
    manifest["files"][args.cases.name] = sha256_file(args.cases)
    manifest["files"][args.reviews.name] = sha256_file(args.reviews)
    manifest["files"][args.curation.name] = sha256_file(args.curation)
    manifest["promptTemplates"] = {
        path.name: sha256_file(path)
        for path in sorted((ROOT / "oracle_prompts").glob("*"))
        if path.is_file()
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "cases": len(cases), "questionsCurated": len(replacements),
        "clinicalLabelsChanged": False, "reviewStatus": "pending",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
