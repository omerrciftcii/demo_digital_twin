"""Grok-facing Oracle A/B/C runner contracts and deterministic analysis.

This module does not know how to reach the user's GPU.  It builds paired
requests, enforces review gates, and summarizes already-produced results.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Literal

import httpx

from .text_document import parse_cco_text
from .text_verify import ANSWER_JSON_SCHEMA

Arm = Literal["A", "B", "C"]
VerdictName = Literal["pass", "partial", "fail"]

ERROR_SOURCES = frozenset({
    "cco_source_renderer",
    "evidence_extraction",
    "evidence_selection_ranking",
    "prompt_context_construction",
    "llm_answer_generation",
    "temporal_grounding",
    "assertion_medication_state",
    "unsupported_causal_inference",
    "numeric_calculation",
    "evaluation_judge_uncertainty",
})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    Path(path).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case_record(case: dict[str, Any], cco_dir: Path) -> Any:
    source = Path(cco_dir) / case["sourceFile"]
    record = parse_cco_text(json.loads(source.read_text(encoding="utf-8-sig")))
    if str(record.id) != case["snapshotId"]:
        raise ValueError("case_snapshot_mismatch")
    if record.content_hash != case["contentHash"]:
        raise ValueError("case_content_hash_mismatch")
    return record


def review_is_question_approved(review: dict[str, Any]) -> bool:
    return review.get("questionReviewStatus") == "approved"


def verified_oracle_annotations(review: dict[str, Any]) -> list[dict[str, Any]]:
    """Return human oracle annotations or fail closed, including valid empty evidence."""
    if review.get("reviewStatus") != "verified":
        raise ValueError("human_oracle_review_pending")
    if not review_is_question_approved(review):
        raise ValueError("question_review_pending")
    if not review.get("reviewer") or not review.get("reviewedAt"):
        raise ValueError("human_reviewer_attestation_missing")
    evidence = review.get("verifiedRelevantEvidence")
    if not isinstance(evidence, list):
        raise ValueError("verified_relevant_evidence_missing")
    if not isinstance(review.get("verifiedIrrelevantEvidence"), list):
        raise ValueError("verified_irrelevant_evidence_missing")
    if not isinstance(review.get("verifiedForbiddenInferences"), list):
        raise ValueError("verified_forbidden_inferences_missing")
    for field in (
        "temporalStatusReview", "diagnosisStatusReview", "medicationActivityReview"
    ):
        if review.get(field) in {None, "pending"}:
            raise ValueError("clinical_status_review_pending")
    return evidence


def build_completion_body(
    *, model: str, messages: list[dict[str, str]], seed: int,
    max_tokens: int, structured_output: Literal["response-format", "off"],
    disable_reasoning: Literal["chat-template", "reasoning", "none"],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
    }
    if structured_output == "response-format":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "oracle_abc_answer",
                "strict": True,
                "schema": ANSWER_JSON_SCHEMA,
            },
        }
    if disable_reasoning == "chat-template":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif disable_reasoning == "reasoning":
        body["reasoning"] = {"enabled": False}
    return body


def request_completion(
    *, base_url: str, api_key: str, body: dict[str, Any], timeout: float,
    transport: httpx.BaseTransport | None = None,
) -> tuple[str, dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    with httpx.Client(timeout=timeout, transport=transport) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/chat/completions", json=body, headers=headers
        )
        response.raise_for_status()
        payload = response.json()
    choice = payload["choices"][0]
    if choice.get("finish_reason") not in {None, "stop"}:
        raise ValueError("model_response_incomplete")
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model_response_empty")
    metadata = {
        "servedModel": payload.get("model"),
        "finishReason": choice.get("finish_reason"),
        "usage": payload.get("usage", {}),
    }
    return content, metadata


def response_evidence_ids(content: str) -> list[str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "evidence_ids" and isinstance(child, list):
                    found.update(item for item in child if isinstance(item, str))
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return sorted(found)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return round(ordered[lower], 2)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return round(value, 2)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def analyze_results(
    runs: list[dict[str, Any]], grades: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    grade_index = {(row["runKey"]): row for row in grades}
    review_index = {row["caseId"]: row for row in reviews}
    completed = [row for row in runs if row.get("status") == "completed"]
    graded = [
        row for row in completed
        if grade_index.get(row["runKey"], {}).get("verdict")
        in {"pass", "partial", "fail"}
    ]
    arms: dict[str, Any] = {}
    for arm in ("A", "B", "B_AI", "C"):
        arm_rows = [row for row in graded if row["arm"] == arm]
        verdicts = Counter(grade_index[row["runKey"]].get("verdict") for row in arm_rows)
        critical = sum(
            bool(grade_index[row["runKey"]].get("criticalSafetyFailure"))
            for row in arm_rows
        )
        latencies = [float(row["latencyMs"]) for row in arm_rows]
        error_sources: Counter[str] = Counter()
        error_flags: Counter[str] = Counter()
        relevant_seen = relevant_total = selected_total = 0
        evidence_cases = 0
        for row in arm_rows:
            grade = grade_index[row["runKey"]]
            sources = grade.get("errorSources", [])
            unknown = set(sources) - ERROR_SOURCES
            if unknown:
                raise ValueError("unknown_error_source")
            error_sources.update(sources)
            error_flags.update(
                key for key, value in grade.get("errorFlags", {}).items() if value
            )
            review = review_index.get(row["caseId"], {})
            oracle = review.get("verifiedRelevantEvidence")
            if not isinstance(oracle, list):
                continue
            gold_ids = {item["evidenceId"] for item in oracle}
            used_ids = set(grade.get(
                "usedEvidenceIds", row.get("responseStableEvidenceIds", [])
            ))
            relevant_seen += len(gold_ids & used_ids)
            relevant_total += len(gold_ids)
            selected_total += len(used_ids)
            evidence_cases += 1
        arms[arm] = {
            "graded": len(arm_rows),
            "verdictCounts": {key: verdicts.get(key, 0) for key in ("pass", "partial", "fail")},
            "passRate": _safe_rate(verdicts.get("pass", 0), len(arm_rows)),
            "criticalSafetyFailures": critical,
            "criticalSafetyFailureRate": _safe_rate(critical, len(arm_rows)),
            "evidenceRecall": _safe_rate(relevant_seen, relevant_total),
            "evidencePrecision": _safe_rate(relevant_seen, selected_total),
            "evidenceScoredCases": evidence_cases,
            "errorSources": dict(sorted(error_sources.items())),
            "errorFlags": dict(sorted(error_flags.items())),
            "latencyMs": {
                "median": round(median(latencies), 2) if latencies else None,
                "p95": _percentile(latencies, 0.95),
            },
        }
    by_pair: dict[tuple[str, int], dict[str, VerdictName]] = defaultdict(dict)
    for row in graded:
        by_pair[(row["caseId"], int(row["repeat"]))][row["arm"]] = grade_index[
            row["runKey"]
        ]["verdict"]
    score = {"fail": 0, "partial": 1, "pass": 2}
    deltas: dict[str, Any] = {}
    for left, right in (
        ("A", "B"), ("A", "B_AI"), ("A", "C"), ("B", "C"), ("B_AI", "C")
    ):
        comparable = [pair for pair in by_pair.values() if left in pair and right in pair]
        wins = sum(score[pair[right]] > score[pair[left]] for pair in comparable)
        losses = sum(score[pair[right]] < score[pair[left]] for pair in comparable)
        deltas[f"{right}-{left}"] = {
            "pairs": len(comparable), "wins": wins, "losses": losses,
            "ties": len(comparable) - wins - losses,
        }
    return {
        "schemaVersion": "cco-oracle-analysis/0.1.0",
        "runRows": len(runs),
        "completedRows": len(completed),
        "gradedRows": len(graded),
        "ungradedCompletedRows": len(completed) - len(graded),
        "blockedRows": sum(row.get("status") == "blocked" for row in runs),
        "experimental": any(row.get("experimental") is True for row in runs),
        "arms": arms,
        "pairedDeltas": deltas,
        "interpretationReady": len(completed) > 0 and len(graded) == len(completed),
    }
