"""Build bounded evidence packets without mutating or impersonating a CCO.

This layer never interprets a number as abnormal, infers disease activity, or
treats terminology relationships as patient evidence. Its job is preservation,
source closure, temporal grouping, and honest disclosure of missing coverage.
"""

import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

from .terminology import definition_base


def estimate_tokens(value: Any) -> int:
    """A documented UTF-8 estimate, not the provider's actual tokenizer count."""
    return math.ceil(len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()) / 3)


def _identity(entry: dict[str, Any]) -> tuple[str, str, str] | None:
    if entry.get("sourceKind") == "insight":
        return None
    base = definition_base(entry.get("definition")) or entry.get("definitionId", "")
    if not base:
        return None
    # latest/series are structural representations of the exact same canonical
    # definition stem, not a clinical disease -> laboratory association.
    stem, _, suffix = base.rpartition(".")
    if suffix in {"latest", "series"}:
        base = stem
    value = entry.get("value") or {}
    if not isinstance(value, dict):
        return (base, "", "")
    if value.get("type") == "coded":
        # Different medications/diagnoses sharing one definition are not a
        # longitudinal group. In particular a prescription does not prove why
        # it was prescribed or that the patient is currently taking it.
        return (base, str(value.get("system", "")), str(value.get("code", "")))
    return (base, str(value.get("type", "")), str(value.get("unit", "")))


def _date(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value)
        if len(value) == 10:
            return result.replace(tzinfo=UTC)
        return result if result.tzinfo is not None else None
    except ValueError:
        return None


def _date_match(entry: dict[str, Any], plan: dict[str, Any]) -> bool:
    start, end = _date(plan.get("dateFrom")), _date(plan.get("dateTo"))
    if end and len(str(plan.get("dateTo"))) == 10:
        end += timedelta(days=1, microseconds=-1)
    occurred_from, occurred_to = _date(entry.get("occurredFrom")), _date(entry.get("occurredTo"))
    if not start and not end:
        return False
    # producedAt and recordedAt are never substituted for an event date.
    if not occurred_from and not occurred_to:
        return False
    earliest, latest = occurred_from or occurred_to, occurred_to or occurred_from
    return (not start or latest >= start) and (not end or earliest <= end)


def _glossary(
    document: dict[str, Any], selected: list[dict[str, Any]], warning_index: list[dict[str, Any]]
) -> dict[str, Any]:
    glossary = document.get("glossary", {})
    families = {entry.get("section") for entry in [*selected, *warning_index]}
    definitions = {
        definition_base(entry.get("definition")) for entry in selected
        if entry.get("sourceKind") != "insight"
    }
    rules = {
        definition_base(entry.get("definition")) for entry in selected
        if entry.get("sourceKind") == "insight"
    } | {definition_base(entry.get("definition")) for entry in warning_index}
    # Frozen dependencies can be filed under the parent insight's section even
    # when their canonical definitions belong to another family.
    families.update(
        family for family in glossary.get("families", {})
        if any(code.startswith(family + ".") for code in definitions | rules)
    )
    return {
        group: {
            key: deepcopy(value) for key, value in glossary.get(group, {}).items() if key in wanted
        }
        for group, wanted in (("families", families), ("facts", definitions), ("rules", rules))
    }


def _model_entry(entry: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "ref", "sourceKind", "section", "sourcePointer", "definition", "definitionId",
        "factId", "insightId", "value", "occurredFrom", "occurredTo", "recordedAt",
        "producedAt", "validUntil", "evaluationKey", "evaluationStatus", "comparisons",
        "evidence", "severity", "dependencies", "parentRef", "selectedReason",
    )
    return {key: deepcopy(entry[key]) for key in fields if key in entry}


def select_context(
    document: dict[str, Any],
    question: str,
    entries: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    *,
    snapshot_id: str,
    plan: dict[str, Any] | None = None,
    token_budget: int = 8000,
    graph_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select indivisible source/longitudinal bundles under an estimated budget.

    ``modelContext`` is the only model-facing portion. UI diagnostics are not
    included in the token budget. Callers must not send it when canAnswer=false.
    Critical source insights and their frozen dependencies are never cut to fit.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    by_ref: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.get("snapshotId") != snapshot_id:
            raise ValueError("evidence snapshot scope mismatch")
        ref = entry.get("ref")
        if not isinstance(ref, str) or not ref or ref in by_ref:
            raise ValueError("evidence references must be unique nonempty strings")
        by_ref[ref] = entry
    plan = deepcopy(plan) if plan is not None else {
        "queries": [question], "concepts": [], "timeIntent": "overview",
        "dateFrom": None, "dateTo": None, "needsClarification": False,
        "clarificationQuestion": None,
    }
    withheld = set(document.get("consent", {}).get("withheldFamilies", []))
    denied = {
        ref for ref, item in by_ref.items()
        if item.get("section") in withheld
        or any(definition_base(item.get("definition")).startswith(family + ".")
               for family in withheld)
    }
    # A cross-family insight contains its frozen values in source/text too.
    # Deny the whole insight and its other frozen children, not just the one
    # prohibited dependency. Otherwise a source preview could leak that value.
    while True:
        expanded_denied = denied | {
            ref for ref, item in by_ref.items()
            if any(dependency in denied for dependency in item.get("dependencies", []))
            or item.get("parentRef") in denied
        }
        if expanded_denied == denied:
            break
        denied = expanded_denied
    allowed = {ref: item for ref, item in by_ref.items() if ref not in denied}
    reverse: dict[str, set[str]] = defaultdict(set)
    related: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for ref, entry in allowed.items():
        for dependency in entry.get("dependencies", []):
            reverse[dependency].add(ref)
        identity = _identity(entry)
        if identity is not None:
            related[identity].add(ref)

    def closure(seeds: set[str], longitudinal: bool = False) -> tuple[set[str], set[str]]:
        found: set[str] = set()
        missing: set[str] = set()
        pending = list(seeds)
        while pending:
            ref = pending.pop()
            if ref in found or ref in missing:
                continue
            if ref not in allowed:
                missing.add(ref)
                continue
            entry = allowed[ref]
            found.add(ref)
            pending.extend(entry.get("dependencies", []))
            # An independently retrieved frozen value retains the insight that
            # gives it meaning, plus all of that insight's other source values.
            if entry.get("sourceKind") == "insightSnapshot":
                pending.extend(reverse.get(ref, set()))
                if entry.get("parentRef"):
                    pending.append(entry["parentRef"])
            if longitudinal:
                identity = _identity(entry)
                if identity is not None:
                    pending.extend(related[identity])
        return found, missing

    critical = {
        ref for ref, entry in allowed.items()
        if entry.get("sourceKind") == "insight" and entry.get("severity") == "critical"
    }
    mandatory, mandatory_missing = closure(critical)
    warning_index = []
    for ref, entry in allowed.items():
        if entry.get("sourceKind") != "insight" or entry.get("severity") != "warning":
            continue
        base = definition_base(entry.get("definition"))
        text = document.get("glossary", {}).get("rules", {}).get(base, {})
        warning_index.append({
            "ref": ref, "insightId": entry.get("insightId"),
            "definition": entry.get("definition"), "section": entry.get("section"),
            "label": text.get("label", base), "severity": "warning",
            "producedAt": entry.get("producedAt"),
            "dependencies": list(entry.get("dependencies", [])),
            "detailStatus": "retrieve_sources_before_interpreting",
        })

    coverage: list[dict[str, Any]] = [{
        "state": "history_unknown",
        "message": "Bu CCO snapshot'ı tüm yaşam boyu kayıtların mevcut olduğunu garanti etmez.",
    }]
    warnings: list[dict[str, Any]] = []
    for family in sorted(withheld):
        coverage.append({"state": "withheld", "family": family,
                         "message": "Bu veri ailesi erişim/rıza nedeniyle saklıdır."})
    if document.get("limits", {}).get("truncated"):
        coverage.append({"state": "source_truncated", "message": "CCO üreticisi kayıt kırpmıştır."})
    narrative_cut = [
        ref for ref, entry in allowed.items()
        if isinstance(entry.get("value"), dict) and entry["value"].get("truncated") is True
    ]
    if narrative_cut:
        coverage.append({"state": "source_content_truncated", "count": len(narrative_cut),
                         "message": "Bazı kaynak metinleri CCO üretilirken kırpılmıştır."})
    missing_data = {}
    for family, section in document.get("sections", {}).items():
        if family in withheld:
            continue
        notes = [
            deepcopy(note) for note in section.get("missingData", [])
            if not any(definition_base(note.get("code")).startswith(hidden + ".")
                       for hidden in withheld)
        ]
        if notes:
            missing_data[family] = notes
    if missing_data:
        coverage.append({"state": "source_missing_data", "families": sorted(missing_data),
                         "message": "Kaynakta veri eksikliği veya süresi dolmuş sonuçlar bildirilmiş."})
    if document.get("release", {}).get("mode") == "preview":
        warnings.append({"code": "preview_release", "message":
                         "Önizleme CCO: içerik incelenmiş klinik öneri gibi sunulmamalıdır."})
    if document.get("release", {}).get("unreviewedDefinitions"):
        warnings.append({"code": "unreviewed_definitions", "message":
                         "Kaynakta klinik incelemesi tamamlanmamış tanımlar bulunuyor."})
    if document.get("safety", {}).get("requiresHumanReview") or critical:
        warnings.append({"code": "human_review_required", "message":
                         "CCO güvenlik bilgisi veya kritik insight insan incelemesi gerektiriyor."})
    if graph_status and graph_status.get("status") != "available":
        warnings.append({"code": "terminology_unavailable", "message":
                         "Neo4j terminoloji desteği yok; yerel kanıt araması kullanıldı."})

    reasons = {ref: "critical_safety" for ref in mandatory}
    selected = set(mandatory)
    blockers = []
    if any(by_ref[ref].get("sourceKind") == "insight"
           and by_ref[ref].get("severity") == "critical" for ref in denied):
        blockers.append("critical_source_withheld")
    if mandatory_missing:
        blockers.append("mandatory_dependency_missing")
    if plan.get("needsClarification"):
        blockers.append("query_needs_clarification")
    eligible: dict[str, dict[str, Any]] = {}
    for item in ranked:
        ref = item.get("ref")
        if ref not in allowed or not isinstance(item.get("score"), (int, float)):
            continue
        if not math.isfinite(item["score"]) or item["score"] <= 0:
            continue
        if ref not in eligible or eligible[ref]["score"] < item["score"]:
            eligible[ref] = item
    # RRF is a rank, not clinical relevance confidence. Until a local evaluation
    # calibrates semantic-only acceptance, keep those candidates inspectable but
    # require lexical/exact corroboration before using them as bundle roots.
    # They can still enter through explicit timeline/dependency closure.
    uncertain = {
        ref for ref, item in eligible.items()
        if not set(item.get("channels", [])) & {"exact", "lexical"}
    }
    ordered = sorted(set(eligible) - uncertain, key=lambda ref: (
        -int(_date_match(allowed[ref], plan)), -eligible[ref]["score"], ref
    ))
    if uncertain:
        coverage.append({"state": "retrieval_uncertain", "count": len(uncertain), "message":
                         "Yalnız anlamsal benzerliği bulunan adayların soru ilgisi doğrulanmadı."})
    if not ordered and uncertain:
        blockers.append("retrieval_uncertain")
    elif not ordered:
        coverage.append({"state": "not_found_in_snapshot", "message":
                         "Soru için eldeki snapshot'ta arama sonucu bulunamadı; yokluk kanıtı değildir."})
        blockers.append("no_relevant_evidence")
    else:
        coverage.append({"state": "retrieved_in_snapshot", "count": len(ordered),
                         "message": "Arama adayları bulundu; bütün geçmişin kapsandığı anlamına gelmez."})

    def materialize(refs: set[str]) -> list[dict[str, Any]]:
        return [
            {**deepcopy(allowed[ref]), "selectedReason": reasons.get(ref, "longitudinal_same_definition")}
            for ref in sorted(refs, key=lambda value: (
                0 if value in mandatory else 1, allowed[value].get("sourcePointer", ""), value
            ))
        ]

    def model_payload(refs: set[str], statuses: list[dict[str, Any]]) -> dict[str, Any]:
        selected_entries = materialize(refs)
        insight_ids = {
            item.get("insightId") for item in selected_entries if item.get("sourceKind") == "insight"
        }
        return {
            "snapshotId": snapshot_id, "asOf": document.get("asOf"),
            "evidence": [_model_entry(entry) for entry in selected_entries],
            "warningIndex": deepcopy(warning_index),
            "coverage": deepcopy(statuses), "warnings": deepcopy(warnings),
            "glossary": _glossary(document, selected_entries, warning_index),
            "safety": deepcopy(document.get("safety", {})),
            "release": deepcopy(document.get("release", {})),
            "consent": deepcopy(document.get("consent", {})),
            "limits": deepcopy(document.get("limits", {})),
            "missingData": deepcopy(missing_data),
            "contentReferences": [
                deepcopy(item) for item in document.get("contentReferences", [])
                if item.get("insightId") in insight_ids
            ],
        }

    # Reserve the bounded diagnostic fields that can arise during selection.
    reserve_coverage = [*coverage, {"state": "budget_excluded", "count": len(allowed),
        "message": "Bazı kanıt grupları token bütçesine sığmadı; sonuç kısmi olabilir."}]
    reserve_tokens = 160
    mandatory_estimate = estimate_tokens(model_payload(mandatory, coverage))
    budget_excluded: set[str] = set()
    dependency_excluded: set[str] = set()
    accepted_query = False
    if mandatory_estimate > token_budget:
        blockers.append("mandatory_context_exceeds_budget")
        budget_excluded = set(ordered) - mandatory
    else:
        for ref in ordered:
            bundle, missing = closure({ref}, longitudinal=True)
            if missing:
                dependency_excluded.update(bundle)
                continue
            if bundle.issubset(selected):
                accepted_query = True
                continue
            proposed = selected | bundle
            if estimate_tokens(model_payload(proposed, reserve_coverage)) + reserve_tokens > token_budget:
                budget_excluded.update(bundle - selected)
                continue
            selected = proposed
            accepted_query = True
            for member in bundle:
                if member not in reasons:
                    if member == ref:
                        reasons[member] = "query_match"
                    elif member in closure({ref})[0]:
                        reasons[member] = "insight_dependency"
                    else:
                        reasons[member] = "longitudinal_same_definition"
        budget_excluded -= selected
        dependency_excluded -= selected
    if budget_excluded:
        coverage.append({"state": "budget_excluded", "count": len(budget_excluded),
                         "message": "Bazı kanıt grupları token bütçesine sığmadı; sonuç kısmi olabilir."})
    if dependency_excluded:
        coverage.append({"state": "dependency_missing", "count": len(dependency_excluded),
                         "message": "Kaynak bağımlılıkları tam olmayan kanıt grupları kullanılmadı."})
    if ordered and not accepted_query:
        blockers.append("no_complete_relevant_bundle")
    selected_entries = materialize(selected)
    if any(entry.get("sourceKind") == "insightSnapshot" and not entry.get("occurredFrom")
           and not entry.get("occurredTo") for entry in selected_entries):
        warnings.append({"code": "frozen_event_date_unknown", "message":
                         "Insight içi donmuş değerlerin olay tarihi bilinmiyor; producedAt ölçüm tarihi değildir."})
    payload = model_payload(selected, coverage)
    estimated = estimate_tokens(payload)
    if estimated > token_budget and "mandatory_context_exceeds_budget" not in blockers:
        # Never silently trim a source bundle if metadata itself consumed more
        # than anticipated. Fail closed and keep diagnostic evidence inspectable.
        blockers.append("context_exceeds_budget")
    excluded = []
    for ref, entry in by_ref.items():
        if ref in selected:
            continue
        reason = "not_retrieved"
        if ref in denied:
            reason = "access_withheld"
        elif ref in budget_excluded:
            reason = "bundle_exceeds_budget"
        elif ref in dependency_excluded:
            reason = "dependency_missing"
        elif ref in uncertain:
            reason = "semantic_relevance_unverified"
        diagnostic = {"ref": ref, "reason": reason}
        # UI source navigation is useful only for permitted evidence. A denied
        # dependency can taint its entire parent/sibling bundle, so the derived
        # denied set (not just the entry's section) controls metadata exposure.
        if ref not in denied and reason not in {"access_withheld", "bundle_contains_withheld_fact"}:
            diagnostic.update({
                key: deepcopy(entry[key]) for key in ("sourcePointer", "definition", "sourceKind")
                if key in entry
            })
        excluded.append(diagnostic)
    fingerprint = json.dumps(
        {"snapshot": snapshot_id, "question": question, "payload": payload},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    packet = {
        "schemaVersion": "context-packet/1.0",
        "packetId": hashlib.sha256(fingerprint.encode()).hexdigest(),
        "snapshotId": snapshot_id, "subjectId": str(document.get("subjectId", "")),
        "asOf": document.get("asOf"), "canAnswer": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "evidence": selected_entries, "excluded": excluded,
        **{key: deepcopy(payload[key]) for key in (
            "warningIndex", "coverage", "warnings", "glossary", "safety", "release",
            "consent", "limits", "contentReferences", "missingData",
        )},
        "counts": {
            "sourceEvidence": len(entries), "selectedEvidence": len(selected_entries),
            "excludedEvidence": len(excluded),
            "sourceFacts": sum(entry.get("sourceKind") == "fact" for entry in entries),
            "selectedFacts": sum(entry.get("sourceKind") == "fact" for entry in selected_entries),
        },
        "tokenEstimate": estimated, "tokenBudget": token_budget,
        "tokenEstimationMethod": "utf8_bytes_divided_by_3_not_provider_tokenizer",
        "retrieval": {
            "plan": plan, "graphStatus": deepcopy(graph_status),
            "semanticPolicy": "lexical_or_exact_corroboration_required_until_calibrated",
            "candidates": [deepcopy(item) for item in eligible.values()],
        },
        "modelContext": payload,
    }
    return packet
