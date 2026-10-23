"""Generate source-validated, experimental AI Oracle reviews with OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_ai_selective_context.ai_oracle import (
    AIOracleError,
    build_ai_manifest,
    build_ai_oracle_request,
    compile_ai_oracle_review,
    request_ai_oracle_review,
    validate_ai_review,
)
from medical_ai_selective_context.oracle_benchmark import (
    load_case_record,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            if len(value) > 1 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--cco-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-6-astra")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.retries <= 4:
        parser.error("--retries must be between 0 and 4")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    _load_env(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    cases = read_jsonl(args.cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = args.out_dir / "oracle_ai_reviews.jsonl"
    drafts_path = args.out_dir / "oracle_ai_drafts.jsonl"
    errors_path = args.out_dir / "oracle_ai_errors.jsonl"
    reviews = read_jsonl(reviews_path) if args.resume and reviews_path.is_file() else []
    drafts = read_jsonl(drafts_path) if args.resume and drafts_path.is_file() else []
    completed = {row["caseId"]: row for row in reviews}
    draft_index = {row["caseId"]: row for row in drafts}
    errors: list[dict] = []

    pending: list[tuple[int, dict, object]] = []
    for case_number, case in enumerate(cases, 1):
        record = load_case_record(case, args.cco_dir)
        existing = completed.get(case["caseId"])
        if existing is not None:
            try:
                validate_ai_review(case=case, record=record, review=existing)
                print(f"[{case_number}/{len(cases)}] {case['caseId']} resume-ok", flush=True)
                continue
            except ValueError:
                reviews = [row for row in reviews if row["caseId"] != case["caseId"]]
                completed.pop(case["caseId"], None)
        pending.append((case_number, case, record))

    def process(item: tuple[int, dict, object]) -> dict:
        case_number, case, record = item
        last_error = ""
        for attempt in range(args.retries + 1):
            case_seed = args.seed + case_number * 10 + attempt
            body = build_ai_oracle_request(
                case=case, record=record, model=args.model, seed=case_seed,
                reasoning_effort=args.reasoning_effort,
            )
            if last_error:
                body["messages"].append({
                    "role": "user",
                    "content": (
                        "Önceki taslak deterministik kaynak kapısından geçmedi: "
                        f"{last_error}. Aynı vakayı yeniden üret. Özellikle katalogdaki "
                        "deterministicExplicitClaims desteklemiyorsa explicit aktiflik, "
                        "kesinlik, iyileşme veya nedensellik işaretleme."
                    ),
                })
            try:
                draft, metadata = request_ai_oracle_review(
                    api_key=api_key, body=body, base_url=base_url, timeout=args.timeout
                )
                review = compile_ai_oracle_review(
                    case=case, record=record, draft=draft, model=args.model,
                    seed=case_seed, reasoning_effort=args.reasoning_effort,
                    metadata=metadata,
                )
                validate_ai_review(case=case, record=record, review=review)
                raw_draft = {
                    "caseId": case["caseId"],
                    "attempt": attempt + 1,
                    "model": args.model,
                    "seed": case_seed,
                    "requestSha256": metadata["requestSha256"],
                    "draft": draft.model_dump(mode="json", by_alias=True),
                }
                print(
                    f"[{case_number}/{len(cases)}] {case['caseId']} accepted "
                    f"attempt={attempt + 1}",
                    flush=True,
                )
                return {"case": case, "review": review, "draft": raw_draft}
            except (AIOracleError, KeyError, TypeError, ValueError) as exc:
                last_error = str(exc) or type(exc).__name__
                print(
                    f"[{case_number}/{len(cases)}] {case['caseId']} rejected "
                    f"attempt={attempt + 1} code={last_error}",
                    flush=True,
                )
                if last_error in {"ai_oracle_rate_limited", "ai_oracle_server_busy"}:
                    time.sleep(min(30, 5 * (2 ** attempt)))
                if last_error in {
                    "ai_oracle_invalid_request",
                    "ai_oracle_insufficient_credits",
                    "ai_oracle_no_compatible_endpoint",
                    "ai_oracle_request_rejected",
                }:
                    break
        return {
            "case": case,
            "error": {
                "caseId": case["caseId"],
                "error": last_error,
                "attempts": attempt + 1,
            },
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, item) for item in pending]
        for future in as_completed(futures):
            result = future.result()
            case = result["case"]
            if "error" in result:
                errors.append(result["error"])
            else:
                completed[case["caseId"]] = result["review"]
                draft_index[case["caseId"]] = result["draft"]
            reviews = [
                completed[row["caseId"]]
                for row in cases if row["caseId"] in completed
            ]
            drafts = [
                draft_index[row["caseId"]]
                for row in cases if row["caseId"] in draft_index
            ]
            write_jsonl(reviews_path, reviews)
            write_jsonl(drafts_path, drafts)
            write_jsonl(errors_path, sorted(errors, key=lambda row: row["caseId"]))

    ordered_reviews = [completed[row["caseId"]] for row in cases if row["caseId"] in completed]
    write_jsonl(reviews_path, ordered_reviews)
    write_jsonl(drafts_path, [
        draft_index[row["caseId"]] for row in cases if row["caseId"] in draft_index
    ])
    if errors:
        write_jsonl(errors_path, errors)
    else:
        write_jsonl(errors_path, [])
    manifest = build_ai_manifest(
        cases_path=args.cases, reviews_path=reviews_path,
        base_manifest_path=args.base_manifest, reviews=ordered_reviews,
        model=args.model, seed=args.seed, reasoning_effort=args.reasoning_effort,
    )
    manifest["files"][drafts_path.name] = sha256_file(drafts_path)
    manifest["complete"] = len(ordered_reviews) == len(cases) and not errors
    manifest_path = args.out_dir / "oracle_ai_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schemaVersion": "cco-ai-oracle-generation-report/0.1.0",
        "cases": len(cases), "accepted": len(ordered_reviews),
        "rejected": len(errors), "complete": manifest["complete"],
        "experimental": True, "humanVerified": False,
        "gold": False, "trainable": False,
        "models": manifest["models"],
        "reasoningEfforts": manifest["reasoningEfforts"],
        "reportedUsageCost": manifest["reportedUsageCost"],
    }
    (args.out_dir / "oracle_ai_generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
