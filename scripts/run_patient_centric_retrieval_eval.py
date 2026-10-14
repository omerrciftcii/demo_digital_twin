"""Offline retrieval evaluation for the private patient-centric benchmark.

The script never calls an LLM or a network service.  It validates that every
gold fact points to the exact CCO source line, evaluates the current evidence
ranking, and optionally adds a cache-only CPU E5 ranking as a small hybrid PoC.
Outputs contain questions, opaque fact ids, rankings, and aggregate metrics;
raw CCO/fact text is deliberately excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_embed import (
    embedding_model_name,
    passage_matrix,
    query_vector,
)
from medical_ai_selective_context.text_evidence import build_evidence
from medical_ai_selective_context.text_qa import _rank_evidence

K_VALUES = (1, 3, 5, 10, 20)
RRF_K = 60


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(payload: object) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _fact_map(package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = package.get("facts")
    if not isinstance(rows, list):
        raise ValueError("facts_missing")
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        fact_id = str(row.get("fact_id") or "").strip()
        if not fact_id or fact_id in mapped:
            raise ValueError("invalid_or_duplicate_fact_id")
        mapped[fact_id] = row
    return mapped


def _profile_groups(profile: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for role in ("direct", "indirect", "medication", "missingness"):
        for row in profile.get(role, []):
            item = dict(row)
            item["role"] = role
            groups.append(item)
    return groups


def _case_groups(case: dict[str, Any], profiles: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for profile_id in case["profile_refs"]:
        if profile_id not in profiles:
            raise ValueError(f"unknown_profile:{profile_id}")
        for group in _profile_groups(profiles[profile_id]):
            group_id = str(group.get("group_id") or "")
            if not group_id or group_id in seen:
                continue
            seen.add(group_id)
            groups.append(group)
    wanted = case.get("group_ids")
    if wanted is not None:
        wanted_set = set(wanted)
        unknown = wanted_set - seen
        if unknown:
            raise ValueError(f"unknown_group_override:{','.join(sorted(unknown))}")
        groups = [group for group in groups if group["group_id"] in wanted_set]
    return groups


def _group_fact_ids(group: dict[str, Any]) -> set[str]:
    any_of = {str(item) for item in group.get("any_of", []) if str(item)}
    all_of = {str(item) for item in group.get("all_of", []) if str(item)}
    if bool(any_of) == bool(all_of):
        raise ValueError(f"group_requires_exactly_one_operator:{group.get('group_id')}")
    return any_of or all_of


def _satisfied(group: dict[str, Any], found: set[str]) -> bool:
    ids = _group_fact_ids(group)
    if group.get("all_of"):
        return ids.issubset(found)
    return bool(ids & found)


def _recall(groups: list[dict[str, Any]], found: set[str]) -> float | None:
    if not groups:
        return None
    return sum(_satisfied(group, found) for group in groups) / len(groups)


def _mean(values: list[float]) -> float | None:
    return None if not values else round(statistics.fmean(values), 4)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    return round(statistics.quantiles(values, n=20, method="inclusive")[18], 3)


def _rank_selective(evidence: list[Any], question: str) -> list[Any]:
    return _rank_evidence(evidence, question)


def _semantic_order(evidence: list[Any], query: str) -> tuple[list[Any], bool]:
    passages = tuple(
        " | ".join(
            part for part in (
                item.display_name,
                item.concept.replace("_", " "),
                item.source_section,
                item.source_text,
            ) if part
        )
        for item in evidence
    )
    matrix = passage_matrix(passages)
    vector = query_vector(query)
    if matrix is None or vector is None:
        return [], False
    scores = matrix @ vector
    ordered = sorted(
        zip(evidence, scores, strict=True),
        key=lambda pair: (-float(pair[1]), pair[0].fact_id or pair[0].id),
    )
    return [item for item, _score in ordered], True


def _rank_hybrid(
    selective: list[Any], semantic: list[Any],
) -> list[Any]:
    by_id = {item.fact_id: item for item in selective if item.fact_id}
    scores: dict[str, float] = defaultdict(float)
    for rank, item in enumerate(selective, start=1):
        if item.fact_id:
            scores[item.fact_id] += 1.0 / (RRF_K + rank)
    for rank, item in enumerate(semantic, start=1):
        if item.fact_id:
            scores[item.fact_id] += 1.0 / (RRF_K + rank)
            by_id.setdefault(item.fact_id, item)
    return [
        by_id[fact_id]
        for fact_id in sorted(scores, key=lambda item: (-scores[item], item))
    ]


def _history_query(case: dict[str, Any]) -> str:
    history = [
        str(item).strip() for item in case.get("user_history", []) if str(item).strip()
    ]
    return "\n".join([*history[-4:], case["question"]])


def _validate_provenance(
    patient_id: str,
    cco: dict[str, Any],
    facts: dict[str, dict[str, Any]],
    gold_ids: set[str],
) -> dict[str, Any]:
    lines = cco["text"].splitlines()
    all_exact = 0
    shared: dict[int, int] = defaultdict(int)
    for row in facts.values():
        line = row.get("source_line")
        text = row.get("source_text")
        if not isinstance(line, int) or not isinstance(text, str):
            continue
        if 1 <= line <= len(lines) and lines[line - 1] == text:
            all_exact += 1
        shared[line] += 1
    missing = sorted(gold_ids - set(facts))
    if missing:
        raise ValueError(f"gold_fact_missing:{patient_id}:{','.join(missing)}")
    invalid_gold = []
    hashes: dict[str, str] = {}
    for fact_id in sorted(gold_ids):
        row = facts[fact_id]
        line = row.get("source_line")
        text = row.get("source_text")
        exact = isinstance(line, int) and isinstance(text, str) and (
            1 <= line <= len(lines) and lines[line - 1] == text
        )
        if not exact:
            invalid_gold.append(fact_id)
        else:
            hashes[fact_id] = _sha256_bytes(text.encode("utf-8"))
    if invalid_gold:
        raise ValueError(
            f"gold_provenance_mismatch:{patient_id}:{','.join(invalid_gold)}"
        )
    return {
        "factCount": len(facts),
        "exactSourceMappings": all_exact,
        "goldFactCount": len(gold_ids),
        "goldSourceHashes": hashes,
        "sharedSourceLineCount": sum(count > 1 for count in shared.values()),
    }


def _metric_row(
    approach: str,
    case: dict[str, Any],
    groups: list[dict[str, Any]],
    ranking: list[Any],
    latency_ms: float,
    *,
    cold_start: bool = False,
) -> dict[str, Any]:
    ranked_ids = [item.fact_id for item in ranking if item.fact_id]
    result: dict[str, Any] = {
        "approach": approach,
        "caseId": case["case_id"],
        "patientId": case["patient_id"],
        "symptom": case["symptom"],
        "conversationId": case.get("conversation_id"),
        "turn": case.get("turn"),
        "latencyMs": round(latency_ms, 3),
        "coldStart": cold_start,
        "hasCriticalGroups": any(group.get("critical") for group in groups),
        "rankedFactIds": ranked_ids[: max(K_VALUES)],
        "metrics": {},
    }
    for k in K_VALUES:
        found = set(ranked_ids[:k])
        overall = _recall(groups, found)
        critical = [group for group in groups if group.get("critical")]
        roles = {
            role: [group for group in groups if group["role"] == role]
            for role in ("direct", "indirect", "medication", "missingness")
        }
        result["metrics"][str(k)] = {
            "recall": None if overall is None else round(overall, 4),
            "criticalMiss": any(not _satisfied(group, found) for group in critical),
            **{
                f"{role}Recall": (
                    None if (value := _recall(role_groups, found)) is None
                    else round(value, 4)
                )
                for role, role_groups in roles.items()
            },
        }
    return result


def _aggregate_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_k: dict[str, Any] = {}
    for k in K_VALUES:
        key = str(k)
        recalls = [
            row["metrics"][key]["recall"] for row in rows
            if row["metrics"][key]["recall"] is not None
        ]
        eligible = [row for row in rows if row["hasCriticalGroups"]]
        role_metrics = {}
        for role in ("direct", "indirect", "medication", "missingness"):
            name = f"{role}Recall"
            values = [
                row["metrics"][key][name] for row in rows
                if row["metrics"][key][name] is not None
            ]
            role_metrics[name] = _mean(values)
        by_k[key] = {
            "macroRecall": _mean(recalls),
            "criticalEvidenceQuestionCount": len(eligible),
            "criticalEvidenceMissRate": (
                None if not eligible else round(
                    sum(row["metrics"][key]["criticalMiss"] for row in eligible)
                    / len(eligible),
                    4,
                )
            ),
            **role_metrics,
        }
    latency = [row["latencyMs"] for row in rows]
    warm_latency = [row["latencyMs"] for row in rows if not row["coldStart"]]
    cold_latency = [row["latencyMs"] for row in rows if row["coldStart"]]
    return {
        "caseCount": len(rows),
        "recallAtK": by_k,
        "latencyMs": {
            "all": {"mean": _mean(latency), "p95": _p95(latency)},
            "warm": {"mean": _mean(warm_latency), "p95": _p95(warm_latency)},
            "cold": {"mean": _mean(cold_latency), "p95": _p95(cold_latency)},
        },
    }


def _aggregate(rows: list[dict[str, Any]], semantic_available: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "semanticAvailable": semantic_available,
        "embeddingModel": embedding_model_name(),
        "approaches": {},
    }
    for approach in sorted({row["approach"] for row in rows}):
        selected = [row for row in rows if row["approach"] == approach]
        payload["approaches"][approach] = {
            "all": _aggregate_slice(selected),
            "singleTurn": _aggregate_slice([
                row for row in selected if row["conversationId"] is None
            ]),
            "conversationTurns": _aggregate_slice([
                row for row in selected if row["conversationId"] is not None
            ]),
        }
    return payload


def _cross_patient(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    single = [row for row in rows if row["conversationId"] is None]
    for approach in sorted({row["approach"] for row in single}):
        for symptom in sorted({row["symptom"] for row in single}):
            selected = [
                row for row in single
                if row["approach"] == approach and row["symptom"] == symptom
            ]
            results.append({
                "approach": approach,
                "symptom": symptom,
                "patientCount": len(selected),
                "patientSpecificRecallAt10": {
                    row["patientId"]: row["metrics"]["10"]["recall"]
                    for row in selected
                },
                "criticalMissPatientsAt10": [
                    row["patientId"] for row in selected
                    if row["metrics"]["10"]["criticalMiss"]
                ],
            })
    return results


def expand_cases(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    """Single-turn cases, then every conversation turn in order with its prior questions."""
    cases = [*benchmark["single_turn_cases"]]
    for conversation in benchmark.get("conversations", []):
        history: list[str] = []
        for turn_number, turn in enumerate(conversation["turns"], start=1):
            cases.append({
                **turn,
                "case_id": f"{conversation['conversation_id']}.t{turn_number}",
                "patient_id": conversation["patient_id"],
                "conversation_id": conversation["conversation_id"],
                "turn": turn_number,
                "user_history": list(history),
            })
            history.append(turn["question"])
    return cases


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Force cache-only CPU behavior even when the caller has CUDA available.
    os.environ["TEXT_EMBEDDING_DEVICE"] = "cpu"
    benchmark = _load_json(args.benchmark)
    profiles = benchmark["evidence_profiles"]
    cases = expand_cases(benchmark)

    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_patient[case["patient_id"]].append(case)

    provenance: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    semantic_available = True
    for patient_id, patient_cases in sorted(by_patient.items()):
        patient = benchmark["patients"][patient_id]
        cco_path = args.data_root / patient["cco_file"]
        facts_path = args.data_root / patient["facts_file"]
        cco = _load_json(cco_path)
        facts_package = _load_json(facts_path)
        facts = _fact_map(facts_package)
        record = parse_cco_text(cco)
        evidence = build_evidence(record, structured=facts_package)
        evidence_ids = {item.fact_id for item in evidence}
        if evidence_ids != set(facts):
            raise ValueError(f"evidence_adapter_loss:{patient_id}")

        gold_ids: set[str] = set()
        for case in patient_cases:
            for group in _case_groups(case, profiles):
                gold_ids.update(_group_fact_ids(group))
        provenance[patient_id] = {
            **_validate_provenance(patient_id, cco, facts, gold_ids),
            "ccoSha256": _sha256_bytes(cco_path.read_bytes()),
            "factsSha256": _sha256_bytes(facts_path.read_bytes()),
        }

        for case_index, case in enumerate(patient_cases):
            groups = _case_groups(case, profiles)
            started = perf_counter()
            selective = _rank_selective(evidence, case["question"])
            selective_ms = (perf_counter() - started) * 1000
            rows.append(_metric_row("selective", case, groups, selective, selective_ms))

            started = perf_counter()
            semantic, available = _semantic_order(evidence, _history_query(case))
            hybrid_ms = (perf_counter() - started) * 1000
            semantic_available = semantic_available and available
            if available:
                hybrid = _rank_hybrid(selective, semantic)
                rows.append(_metric_row(
                    "hybrid", case, groups, hybrid, hybrid_ms,
                    cold_start=case_index == 0,
                ))

    summary = _aggregate(rows, semantic_available)
    result = {
        "schemaVersion": "patient-centric-retrieval-result/1",
        "createdAt": datetime.now(UTC).isoformat(),
        "benchmarkHash": _canonical_hash(benchmark),
        "benchmarkCaseCount": len(cases),
        "singleTurnCaseCount": len(benchmark["single_turn_cases"]),
        "conversationTurnCount": len(cases) - len(benchmark["single_turn_cases"]),
        "gpuOrLlmCalled": False,
        "productionServiceCalled": False,
        "provenance": provenance,
        "summary": summary,
        "crossPatient": _cross_patient(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "benchmarkHash": result["benchmarkHash"],
        "caseCount": result["benchmarkCaseCount"],
        "gpuOrLlmCalled": result["gpuOrLlmCalled"],
        "summary": result["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
