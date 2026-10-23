"""Create the 20-case V3 real-CCO question pilot; never creates gold labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_ai_selective_context.oracle_eval import (
    OracleQuestionBatch,
    audit_cco_directory,
    build_evidence_review_queue,
    build_openrouter_question_request,
    compile_question_candidates,
    pilot_summary,
    request_openrouter_questions,
    select_oracle_pilot_records,
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cco-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-6-luna")
    parser.add_argument("--provider", choices=("azure", "amazon-bedrock"), default="azure")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--authorize-external-anonymous-cco", action="store_true")
    parser.add_argument(
        "--reuse-question-drafts", type=Path,
        help="Reuse a previously schema-validated Luna draft without another API call.",
    )
    args = parser.parse_args()
    if not args.authorize_external_anonymous_cco:
        parser.error("external real-CCO processing requires explicit authorization flag")

    _load_env(ROOT / ".env")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_rows, eligible = audit_cco_directory(args.cco_dir)
    if len(eligible) < 20:
        raise SystemExit("not_enough_anonymous_valid_ccos: no external request was made")
    selected = select_oracle_pilot_records(eligible)
    if args.reuse_question_drafts:
        batch = OracleQuestionBatch.model_validate_json(
            args.reuse_question_drafts.read_text(encoding="utf-8-sig")
        )
        request_body = build_openrouter_question_request(
            selected, model=args.model, seed=args.seed, provider_slug=args.provider
        )
        request_hash = hashlib.sha256(
            json.dumps(
                request_body, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest()
        actual_provider = None
    else:
        batch, request_hash, actual_provider = request_openrouter_questions(
            api_key=os.getenv("OPENROUTER_API_KEY", ""), selected=selected,
            model=args.model, seed=args.seed, provider_slug=args.provider,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        if actual_provider is not None:
            normalized = actual_provider.casefold().replace(" ", "-")
            if args.provider not in normalized:
                raise SystemExit("provider_attestation_mismatch: outputs were not accepted")
    _write_json(
        args.out_dir / "question-drafts-unreviewed.json",
        batch.model_dump(mode="json", by_alias=True),
    )
    cases = compile_question_candidates(selected, batch)
    (args.out_dir / "oracle_cases.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in cases) + "\n",
        encoding="utf-8",
    )
    reviews = build_evidence_review_queue(cases)
    (args.out_dir / "oracle_evidence_reviews.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in reviews) + "\n",
        encoding="utf-8",
    )
    audit_path = args.out_dir / "source-audit.json"
    _write_json(audit_path, {
        "examined": len(audit_rows),
        "eligible": len(eligible),
        "files": audit_rows,
    })
    summary = pilot_summary(
        audit_rows, cases, request_sha256=request_hash,
        requested_provider=args.provider, actual_provider=actual_provider,
    )
    summary["externalProcessing"]["reusedSchemaValidatedDraft"] = bool(
        args.reuse_question_drafts
    )
    report_path = args.out_dir / "pilot-report.json"
    _write_json(report_path, summary)
    cases_path = args.out_dir / "oracle_cases.jsonl"
    reviews_path = args.out_dir / "oracle_evidence_reviews.jsonl"
    prompt_paths = sorted((ROOT / "oracle_prompts").glob("*"))
    manifest = {
        "schemaVersion": "cco-oracle-run-manifest/0.1.0",
        "generatedAt": datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(),
        "packageStatus": "human_review_pending",
        "sourceContract": {
            "fields": ["id", "subjectId", "asOf", "text"],
            "sourceCcosModified": False,
            "snapshotValidationRequired": True,
            "evidenceReference": "evidenceId+sourceSpan",
            "legacyFactRole": "legacy_audit_only",
        },
        "pilot": {
            "cases": len(cases),
            "uniqueCcos": len({row["sourceFile"] for row in cases}),
            "questionsGeneratedBy": args.model,
            "questionsAreGold": False,
            "questionReviewStatus": "pending",
            "oracleBReviewStatus": "pending",
        },
        "targetRuntime": {
            "owner": "Grok",
            "model": "gemma4-26b-a4b",
            "quantization": "AWQ 4-bit",
            "server": "vLLM 0.20.2",
            "reasoning": "disabled",
            "gpu": "NVIDIA L40S-24Q vGPU (24 GB slice of physical L40S 48 GB)",
            "localVramObservedGb": 18.8,
        },
        "pairedDesign": {
            "arms": {
                "A": "full_cco_no_relevance_guide",
                "B": "full_cco_human_verified_oracle_guide",
                "C": "full_cco_repo_lexical_ranker_guide",
            },
            "sameWeights": True,
            "sameFullCco": True,
            "sameQuestion": True,
            "temperature": 0,
            "repeats": 3,
            "seed": args.seed,
            "armOrder": "rotated_per_case_and_repeat",
        },
        "reviewGates": {
            "allOfficialArmsRequireApprovedQuestion": True,
            "oracleBRequiresHumanVerifiedEvidence": True,
            "pendingCasesAreReportedAndNotRun": True,
        },
        "files": {
            "oracle_cases.jsonl": _sha256(cases_path),
            "oracle_evidence_reviews.jsonl": _sha256(reviews_path),
            "source-audit.json": _sha256(audit_path),
            "pilot-report.json": _sha256(report_path),
        },
        "promptTemplates": {
            path.name: _sha256(path) for path in prompt_paths if path.is_file()
        },
        "realGemmaBenchmarkExecuted": False,
        "modelPerformanceClaimsAvailable": False,
        "loraStarted": False,
        "syntheticScaleExpanded": False,
    }
    _write_json(args.out_dir / "oracle_run_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
