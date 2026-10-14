"""Isolated four-arm experiment; never imported by the production application.

Run only from a private source snapshot on the inference host. This deliberately
calls the verified reasoning stage, not the public safety gate; gate decisions
are recorded separately. Raw attempts and final answers are sensitive artifacts.
Every completion, including retry, is guarded by a served-tokenizer 16k budget.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from medical_ai_selective_context import text_qa as qa
from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_evidence import build_evidence
from medical_ai_selective_context.text_llm import ModelError, TextModelClient, TextModelSettings
from medical_ai_selective_context.text_safety import classify_question
from medical_ai_selective_context.text_verify import parse_model_payload, verify_payload
from scripts.run_patient_centric_retrieval_eval import (
    _canonical_hash,
    _case_groups,
    _fact_map,
    _group_fact_ids,
    _history_query,
    _load_json,
    _rank_hybrid,
    _rank_selective,
    _recall,
    _satisfied,
    _semantic_order,
    _validate_provenance,
    expand_cases,
)

ARMS = ("oracle", "full", "selective", "hybrid")
LIMIT = 16384


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    path.chmod(0o600)


def source_manifest(root: Path) -> dict[str, str]:
    paths = sorted((root / "src").rglob("*.py"))
    paths += sorted((root / "scripts").glob("run_patient_centric*eval.py"))
    paths += sorted((root / "data" / "terminology").glob("*.json"))
    paths += sorted((root / "data").glob("drug_classes.json"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def coverage(groups: list[dict], found: set[str]) -> dict:
    critical = [g for g in groups if g.get("critical")]
    return {
        "groupRecall": _recall(groups, found),
        "criticalMiss": any(not _satisfied(g, found) for g in critical) if critical else None,
        "missingGroups": [g["group_id"] for g in groups if not _satisfied(g, found)],
    }


def compose_plan(
    arm: str,
    record,
    question: str,
    history: list[dict],
    facts: dict,
    groups: list[dict],
    ranked_ids: list[str],
    output_tokens: int,
):
    """Use current builders without editing them. Hybrid is a top-20 source-line PoC."""
    package = build_evidence(record, structured=facts)
    gold = {fid for group in groups for fid in _group_fact_ids(group)}
    if arm in {"full", "oracle", "hybrid"}:
        wanted = None if arm == "full" else gold if arm == "oracle" else set(ranked_ids[:20])
        plan = qa.build_full_context(record, question, history, facts, oracle_fact_ids=wanted)
        body = qa._sent_chart(
            plan.messages[1]["content"], qa.render_compact_catalog(plan.prompt_package)
        )
    elif arm == "selective":
        messages = qa.build_messages(
            record, question, history, budget=qa._prompt_budget(output_tokens)
        )
        body = qa._chart_from_record_message(messages[1]["content"])
        package = build_evidence(record, body, structured=facts)
        messages, prompt = qa._fit_evidence(messages, package, output_tokens, question=question)
        plan = qa.FullContextPlan(
            messages,
            prompt,
            package,
            {
                "mode": "selective",
                "ccoComplete": body == record.text,
                "history": qa._history_report(history),
                "trimmed": body != record.text,
            },
        )
    else:
        raise ValueError("unknown_arm")
    catalog_ids = {e.fact_id for e in plan.prompt_package if e.fact_id}
    # Whole, exact source-line membership, NOT substring matching of values.
    sent_lines = set(body.splitlines())
    raw_ids = {e.fact_id for e in package if e.source_text in sent_lines and e.fact_id}
    report = {
        **plan.report,
        "arm": arm,
        "sourceTextExactLines": True,
        "catalogFactIds": sorted(catalog_ids),
        "rawTextFactIds": sorted(raw_ids),
        "selectedSeedFactIds": ranked_ids[:20] if arm == "hybrid" else None,
        "catalogCoverage": coverage(groups, catalog_ids),
        "rawTextCoverage": coverage(groups, raw_ids),
        "history": qa._history_report(history),
    }
    return dataclasses.replace(plan, report=report)


class AuditedClient(TextModelClient):
    def __init__(self):
        super().__init__(
            TextModelSettings(
                base_url="http://127.0.0.1:8000/v1",
                model="gemma4-26b-a4b",
                require_zdr=False,
                local_inference=True,
                allow_external_real_data=False,
                answer_max_tokens=2000,
                reasoning=False,
                disable_thinking="true",
                retry_max=0,
                allow_fallbacks=False,
            )
        )
        self.attempts: list[dict] = []

    async def count_tokens(self, messages=None, prompt=None):
        counted = await super().count_tokens(messages, prompt)
        if counted is None:
            raise ModelError("tokenizer_unavailable_eval_closed")
        return counted

    async def complete(self, messages, json_schema=None):
        counted = await self.count_tokens(messages)
        if counted["count"] + self.settings.answer_max_tokens + 128 > LIMIT:
            raise ModelError("context_overflow_eval_guard")
        attempt = {"messages": messages, "tokenCount": counted}
        self.attempts.append(attempt)
        async for event in super().complete(messages, json_schema):
            if event["type"] != "token":
                attempt["completion"] = event
            yield event


async def run(args):
    os.umask(0o077)
    # Runtime-only env; no .env or serving configuration is touched.
    os.environ.update(
        {
            "QA_CONTEXT_TOKENS": str(LIMIT),
            "QA_FULL_CONTEXT_ENABLED": "true",
            "DETERMINISTIC_EVIDENCE_VERIFICATION_ENABLED": "true",
            "TEXT_EMBEDDING_DEVICE": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    benchmark = _load_json(args.benchmark)
    cases = expand_cases(benchmark)
    patients, input_hashes, provenance = {}, {}, {}
    for pid, spec in benchmark["patients"].items():
        cco_path, fact_path = (args.data_root / spec[k] for k in ("cco_file", "facts_file"))
        cco, facts = _load_json(cco_path), _load_json(fact_path)
        record = parse_cco_text(cco)
        if record.text != cco["text"]:
            raise ValueError("source_text_normalized")
        fmap = _fact_map(facts)
        gold = {
            f
            for c in cases
            if c["patient_id"] == pid
            for g in _case_groups(c, benchmark["evidence_profiles"])
            for f in _group_fact_ids(g)
        }
        provenance[pid] = _validate_provenance(pid, cco, fmap, gold)
        mapped = {f.get("source_line") for f in fmap.values()}
        provenance[pid]["unmappedNonemptyContentLines"] = [
            n
            for n, line in enumerate(cco["text"].splitlines(), 1)
            if line.strip() and not line.startswith("#") and n not in mapped
        ]
        input_hashes[pid] = {"cco": _canonical_hash(cco), "facts": _canonical_hash(facts)}
        patients[pid] = (record, facts, build_evidence(record, structured=facts))
    manifest = {
        "benchmarkHash": _canonical_hash(benchmark),
        "sourceHashes": source_manifest(ROOT),
        "inputHashes": input_hashes,
        "arms": list(ARMS),
        "effectiveContextLimit": LIMIT,
        "hybrid": "E5 CPU + current ranking equal RRF60; top20; co-line closure",
        "gateMode": "reasoning_stage_only; default safety decision recorded, not enforced",
        "sampling": "current chat_body; reasoning off; max_output2000; serial; no network retries",
        "provenance": provenance,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and _load_json(manifest_path) != manifest:
        raise ValueError("resume_manifest_mismatch")
    dump(manifest_path, manifest)
    row_path = args.out_dir / "rows.jsonl"
    rows = (
        [json.loads(s) for s in row_path.read_text(encoding="utf-8").splitlines()]
        if row_path.exists()
        else []
    )
    old = {r["key"]: r for r in rows}
    histories = defaultdict(list)
    client, answerer = AuditedClient(), None
    answerer = qa.TextAnswerer(client)
    for ci, case in enumerate(cases):
        record, facts, evidence = patients[case["patient_id"]]
        groups = _case_groups(case, benchmark["evidence_profiles"])
        retrieval_start = perf_counter()
        lexical = _rank_selective(evidence, case["question"])
        semantic, available = _semantic_order(evidence, _history_query(case))
        if not available:
            raise ValueError("cpu_embedding_unavailable_no_fake_hybrid")
        hybrid = _rank_hybrid(lexical, semantic)
        retrieval_ms = round((perf_counter() - retrieval_start) * 1000, 3)
        # Rotate execution order; conversation histories remain arm/patient scoped.
        arms = ARMS[ci % 4 :] + ARMS[: ci % 4]
        for arm in arms:
            key = arm + ":" + case["case_id"]
            hkey = (arm, case["patient_id"], case.get("conversation_id"))
            history = list(histories[hkey]) if case.get("conversation_id") else []
            if key in old:
                row = old[key]
            else:
                client.attempts = []
                ranked = hybrid if arm == "hybrid" else lexical
                ranked_ids = [e.fact_id for e in ranked]
                started = perf_counter()
                plan = compose_plan(
                    arm,
                    record,
                    case["question"],
                    history,
                    facts,
                    groups,
                    ranked_ids,
                    client.settings.answer_max_tokens,
                )
                context = {
                    **plan.report,
                    **qa.fit_report(
                        plan.messages,
                        await client.count_tokens(plan.messages),
                        client.settings.answer_max_tokens,
                    ),
                }
                decision = classify_question(case["question"])
                result = {}
                # Evaluation process only. No source edit, service reload, public endpoint or gate change.
                with patch.object(qa, "build_full_context", return_value=plan):
                    client.guard(record.data_class)
                    async for event in answerer._verified_events(
                        record,
                        case["question"],
                        history,
                        perf_counter(),
                        facts,
                    ):
                        if event["type"] == "done":
                            result = event
                attempts = []
                for attempt in client.attempts:
                    completion = attempt.get("completion", {})
                    verdict = verify_payload(
                        parse_model_payload(completion.get("text", "")),
                        plan.prompt_package,
                        catalog=plan.package,
                    )
                    attempts.append({**attempt, "verdict": dataclasses.asdict(verdict)})
                validation, metrics = result.get("validation", {}), result.get("metrics", {})
                cited = set(validation.get("trace", {}).get("cited_fact_ids", []))
                row = {
                    "key": key,
                    "arm": arm,
                    "case": case,
                    "history": history,
                    "context": context,
                    "defaultSafetyDecision": dataclasses.asdict(decision),
                    "retrievalAtK": {
                        str(k): coverage(groups, set(ranked_ids[:k])) for k in (1, 3, 5, 10, 20)
                    },
                    "retrievalMsCombined": retrieval_ms,
                    "compositionAndAnswerMs": round((perf_counter() - started) * 1000, 3),
                    "result": result,
                    "attempts": attempts,
                    "citationCoverage": coverage(groups, cited),
                    "modelCalls": metrics.get("modelCalls", 0),
                }
                with row_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_path.chmod(0o600)
                rows.append(row)
                print(
                    f"{len(rows)}/{len(cases) * 4} {key} {result.get('status')} "
                    f"calls={row['modelCalls']}",
                    flush=True,
                )
            if case.get("conversation_id"):
                histories[hkey].extend(
                    [
                        {"role": "user", "content": case["question"]},
                        {"role": "assistant", "content": row["result"].get("answer", "")},
                    ]
                )
    if source_manifest(ROOT) != manifest["sourceHashes"]:
        raise ValueError("source_changed_during_run")
    dump(
        args.out_dir / "complete.json",
        {"rows": len(rows), "manifestHash": _canonical_hash(manifest)},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.out_dir.resolve().is_relative_to(ROOT / "data" / "evaluation"):
        parser.error("sensitive output must remain inside data/evaluation")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
