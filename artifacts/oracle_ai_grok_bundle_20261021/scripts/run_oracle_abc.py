"""Run paired Oracle A/B/C prompts on Grok's OpenAI-compatible Gemma endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_ai_selective_context.ai_oracle import (
    ai_review_annotations,
    ai_review_constraints,
    validate_ai_review,
)
from medical_ai_selective_context.oracle_benchmark import (
    build_completion_body,
    load_case_record,
    read_jsonl,
    request_completion,
    response_evidence_ids,
    sha256_file,
    verified_oracle_annotations,
    write_jsonl,
)
from medical_ai_selective_context.oracle_eval import build_oracle_abc_plan


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ai-reviews", type=Path)
    parser.add_argument("--ai-manifest", type=Path)
    parser.add_argument("--cco-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="gemma4-26b-a4b")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument(
        "--arms", nargs="+", choices=("A", "B", "B_AI", "C"),
        default=["A", "B", "C"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--structured-output", choices=("response-format", "off"),
        default="response-format",
    )
    parser.add_argument(
        "--disable-reasoning", choices=("chat-template", "reasoning", "none"),
        default="chat-template",
    )
    parser.add_argument("--allow-unreviewed-questions", action="store_true")
    parser.add_argument("--experimental-ai-oracle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be between 1 and 10")
    if "B_AI" in args.arms and not args.experimental_ai_oracle:
        parser.error("B_AI requires --experimental-ai-oracle")
    if args.experimental_ai_oracle and (not args.ai_reviews or not args.ai_manifest):
        parser.error("experimental AI Oracle requires --ai-reviews and --ai-manifest")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if manifest.get("realGemmaBenchmarkExecuted") is not False:
        raise SystemExit("invalid_manifest_execution_state")
    for path in (args.cases, args.reviews):
        expected_hash = manifest.get("files", {}).get(path.name)
        if not expected_hash or sha256_file(path) != expected_hash:
            raise SystemExit(f"manifest_hash_mismatch:{path.name}")
    cases = read_jsonl(args.cases)
    reviews = {row["caseId"]: row for row in read_jsonl(args.reviews)}
    ai_reviews: dict[str, dict] = {}
    if args.experimental_ai_oracle:
        ai_manifest = json.loads(args.ai_manifest.read_text(encoding="utf-8-sig"))
        if ai_manifest.get("experimental") is not True or ai_manifest.get("complete") is not True:
            raise SystemExit("ai_oracle_manifest_not_complete")
        for path in (args.cases, args.ai_reviews, args.manifest):
            expected_hash = ai_manifest.get("files", {}).get(path.name)
            if not expected_hash or sha256_file(path) != expected_hash:
                raise SystemExit(f"ai_manifest_hash_mismatch:{path.name}")
        ai_reviews = {row["caseId"]: row for row in read_jsonl(args.ai_reviews)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "oracle_abc_runs.jsonl"
    existing = read_jsonl(output_path) if args.resume and output_path.is_file() else []
    completed_keys = {row["runKey"] for row in existing if row.get("status") == "completed"}
    rows = list(existing)
    api_key = os.getenv(args.api_key_env, "")

    for repeat in range(1, args.repeats + 1):
        for case_index, case in enumerate(cases):
            review = reviews.get(case["caseId"], {})
            record = load_case_record(case, args.cco_dir)
            ai_review = ai_reviews.get(case["caseId"])
            ai_review_error = None
            if args.experimental_ai_oracle:
                if ai_review is None:
                    ai_review_error = "ai_provisional_review_missing"
                else:
                    try:
                        validate_ai_review(case=case, record=record, review=ai_review)
                    except ValueError as exc:
                        ai_review_error = str(exc)
            arm_order = list(args.arms)
            shift = (case_index + repeat - 1) % len(arm_order)
            arm_order = arm_order[shift:] + arm_order[:shift]
            for arm in arm_order:
                run_key = f"{case['caseId']}:{repeat}:{arm}"
                if run_key in completed_keys:
                    continue
                rows = [row for row in rows if row.get("runKey") != run_key]
                blocked_reason = ai_review_error
                if not args.experimental_ai_oracle:
                    if not args.allow_unreviewed_questions and review.get(
                        "questionReviewStatus"
                    ) != "approved":
                        blocked_reason = "question_review_pending"
                elif arm == "B" and review.get("questionReviewStatus") != "approved":
                    blocked_reason = "question_review_pending"
                oracle = None
                oracle_constraints = None
                if arm == "B" and blocked_reason is None:
                    try:
                        oracle = verified_oracle_annotations(review)
                    except ValueError as exc:
                        blocked_reason = str(exc)
                elif arm == "B_AI" and blocked_reason is None:
                    oracle = ai_review_annotations(ai_review)
                    oracle_constraints = ai_review_constraints(ai_review)
                if blocked_reason:
                    rows.append({
                        "runKey": run_key, "caseId": case["caseId"], "repeat": repeat,
                        "arm": arm, "status": "blocked", "blockedReason": blocked_reason,
                        "modelCalled": False,
                        "experimental": args.experimental_ai_oracle,
                    })
                    continue
                plan = build_oracle_abc_plan(
                    record, case["question"], arm, oracle_annotations=oracle,
                    oracle_constraints=oracle_constraints,
                )
                body = build_completion_body(
                    model=args.model, messages=plan.messages,
                    seed=args.seed + repeat - 1, max_tokens=args.max_tokens,
                    structured_output=args.structured_output,
                    disable_reasoning=args.disable_reasoning,
                )
                base_row = {
                    "runKey": run_key, "caseId": case["caseId"], "repeat": repeat,
                    "arm": arm, "model": args.model, "seed": args.seed + repeat - 1,
                    "promptSha256": _hash_json(plan.messages),
                    "requestSha256": _hash_json(body),
                    "ccoSha256": plan.report["ccoSha256"],
                    "sentCcoSha256": plan.report["sentCcoSha256"],
                    "fullCcoUnchanged": plan.report["fullCcoUnchanged"],
                    "annotationSource": plan.report["annotationSource"],
                    "annotationCount": plan.report["annotationCount"],
                    "promptEvidenceIdMap": plan.report["promptEvidenceIdMap"],
                    "experimental": args.experimental_ai_oracle,
                    "aiOracleReviewStatus": (
                        ai_review.get("reviewStatus")
                        if args.experimental_ai_oracle else None
                    ),
                    "aiOracleReviewSha256": (
                        _hash_json(ai_review) if args.experimental_ai_oracle else None
                    ),
                }
                if args.dry_run:
                    rows.append({**base_row, "status": "dry_run", "modelCalled": False})
                    continue
                started = perf_counter()
                try:
                    content, metadata = request_completion(
                        base_url=args.base_url, api_key=api_key, body=body,
                        timeout=args.timeout,
                    )
                    prompt_ids = response_evidence_ids(content)
                    rows.append({
                        **base_row, "status": "completed", "modelCalled": True,
                        "latencyMs": round((perf_counter() - started) * 1000, 2),
                        "response": content,
                        "responsePromptEvidenceIds": prompt_ids,
                        "responseStableEvidenceIds": sorted({
                            plan.report["promptEvidenceIdMap"][item]
                            for item in prompt_ids
                            if item in plan.report["promptEvidenceIdMap"]
                        }),
                        **metadata,
                    })
                except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                    rows.append({
                        **base_row, "status": "failed", "modelCalled": True,
                        "latencyMs": round((perf_counter() - started) * 1000, 2),
                        "error": type(exc).__name__,
                    })
                write_jsonl(output_path, rows)

    write_jsonl(output_path, rows)
    grading_path = args.out_dir / "oracle_grading_queue.jsonl"
    prior_grades = {
        row["runKey"]: row for row in read_jsonl(grading_path)
    } if grading_path.is_file() else {}
    grading = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        grading.append(prior_grades.get(row["runKey"], {
            "runKey": row["runKey"], "caseId": row["caseId"],
            "repeat": row["repeat"], "arm": row["arm"], "verdict": None,
            "criticalSafetyFailure": None,
            "usedEvidenceIds": row.get("responseStableEvidenceIds", []),
            "usedPromptEvidenceIds": row.get("responsePromptEvidenceIds", []),
            "errorSources": [],
            "errorFlags": {
                "wrongDate": False, "staleAsCurrent": False,
                "preliminaryAsDefinitive": False, "prescriptionAsActive": False,
                "inventedFact": False, "unsupportedCausality": False,
            },
            "grader": None, "gradedAt": None, "notes": None,
            "experimental": bool(row.get("experimental")),
        }))
    write_jsonl(grading_path, grading)
    pending_human_review: list[str] = []
    for case in cases:
        try:
            verified_oracle_annotations(reviews.get(case["caseId"], {}))
        except ValueError:
            pending_human_review.append(case["caseId"])
    official_oracle_b_ready = not pending_human_review
    summary = {
        "rows": len(rows),
        "statusCounts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "modelCalls": sum(bool(row.get("modelCalled")) for row in rows),
        "dryRun": args.dry_run,
        "experimental": args.experimental_ai_oracle,
        "humanVerifiedOracleUsed": "B" in args.arms,
        "aiProvisionalOracleUsed": "B_AI" in args.arms,
        "runConfig": {
            "model": args.model, "arms": args.arms, "repeats": args.repeats,
            "seed": args.seed, "temperature": 0, "maxTokens": args.max_tokens,
            "structuredOutput": args.structured_output,
            "disableReasoning": args.disable_reasoning,
        },
        "pendingHumanReview": pending_human_review,
        "automaticCZeroAnnotationCases": sorted({
            row["caseId"] for row in rows
            if row.get("arm") == "C"
            and row.get("status") in {"dry_run", "completed"}
            and row.get("annotationCount") == 0
        }),
        "officialOracleBReady": official_oracle_b_ready,
        "experimentalAiOracleReady": (
            args.experimental_ai_oracle
            and not any(row.get("status") == "blocked" for row in rows)
        ),
    }
    (args.out_dir / "oracle_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
