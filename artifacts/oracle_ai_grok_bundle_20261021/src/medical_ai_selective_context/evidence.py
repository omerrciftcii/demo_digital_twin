"""Lossless, source-addressed evidence projection of a validated CCO snapshot."""

import hashlib
import json
import re
from copy import deepcopy
from typing import Any


def pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def evidence_ref(snapshot_id: str, pointer: str) -> str:
    return "ev_" + hashlib.sha256(f"{snapshot_id}\n{pointer}".encode()).hexdigest()[:32]


def resolve_pointer(document: Any, pointer: str) -> Any:
    """Resolve RFC 6901 JSON pointers without accepting Python negative indices."""
    if pointer == "":
        return deepcopy(document)
    if not pointer.startswith("/"):
        raise ValueError("A source pointer must be an RFC 6901 JSON pointer")
    current = document
    for encoded in pointer[1:].split("/"):
        if re.search(r"~(?![01])", encoded):
            raise ValueError("Invalid JSON pointer escape")
        component = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", component):
                raise ValueError("Invalid JSON pointer array index")
            current = current[int(component)]
        elif isinstance(current, dict):
            current = current[component]
        else:
            raise ValueError("JSON pointer traverses a scalar")
    return deepcopy(current)


def _glossary_text(document: dict[str, Any], section: str, definition: str,
                   source_kind: str) -> str:
    glossary = document.get("glossary", {})
    entries = [glossary.get("families", {}).get(section, {})]
    group = "rules" if source_kind == "insight" else "facts"
    entries.append(glossary.get(group, {}).get(definition.split("|")[0], {}))
    return " ".join(str(entry.get(key, "")) for entry in entries
                    for key in ("label", "description"))


def _entry(document: dict[str, Any], snapshot_id: str, section: str,
           source: dict[str, Any], pointer: str, source_kind: str,
           parent: dict[str, Any] | None = None) -> dict[str, Any]:
    definition = source["definition"]
    is_fact = source_kind == "fact"
    is_insight = source_kind == "insight"
    entry = {
        "ref": evidence_ref(snapshot_id, pointer), "snapshotId": snapshot_id,
        "sourcePointer": pointer, "sourceKind": source_kind, "section": section,
        "definition": definition, "definitionId": definition.split("|")[0],
        "factId": source.get("factId"),
        "insightId": source.get("insightId") if is_insight else (
            parent.get("insightId") if parent else None
        ),
        "value": deepcopy(source.get("value")),
        # An insight's producedAt is not the event time of its frozen facts.
        "occurredFrom": source.get("occurredFrom") if is_fact else None,
        "occurredTo": source.get("occurredTo") if is_fact else None,
        "recordedAt": source.get("recordedAt") if is_fact else None,
        "producedAt": source.get("producedAt") if is_insight else None,
        "validUntil": source.get("validUntil"),
        "evaluationKey": source.get("evaluationKey"),
        "evaluationStatus": source.get("evaluationStatus"),
        "comparisons": deepcopy(source.get("comparisons", [])),
        "evidence": deepcopy(source.get("evidence", [])),
        "severity": source.get("severity") if is_insight else None,
        "dependencies": [], "parentRef": parent.get("ref") if parent else None,
        "source": deepcopy(source),
    }
    # Search text is derived only from supplied source fields and the glossary;
    # there is no inferred diagnosis, anomaly flag, or synthetic event date.
    search_value = source.get("facts", []) if is_insight else source.get("value")
    entry["text"] = " ".join([
        section, definition,
        _glossary_text(document, section, definition, source_kind),
        json.dumps(search_value, ensure_ascii=False, sort_keys=True),
        json.dumps(source.get("comparisons", []), ensure_ascii=False, sort_keys=True),
        " ".join(str(entry[key] or "") for key in
                 ("occurredFrom", "occurredTo", "recordedAt", "producedAt", "severity")),
    ])
    return entry


def build_evidence(document: dict[str, Any], snapshot_id: str) -> list[dict[str, Any]]:
    """Index all facts and every frozen insight fact, even when factIds collide."""
    entries = []
    for section, contents in document["sections"].items():
        base = f"/sections/{pointer_escape(section)}"
        for number, fact in enumerate(contents["facts"]):
            entries.append(_entry(document, snapshot_id, section, fact,
                                  f"{base}/facts/{number}", "fact"))
        for number, insight in enumerate(contents["insights"]):
            pointer = f"{base}/insights/{number}"
            parent = _entry(document, snapshot_id, section, insight, pointer, "insight")
            entries.append(parent)
            for frozen_number, frozen in enumerate(insight["facts"]):
                child = _entry(document, snapshot_id, section, frozen,
                               f"{pointer}/facts/{frozen_number}", "insightSnapshot", parent)
                parent["dependencies"].append(child["ref"])
                entries.append(child)
    return entries
