"""Privacy-safe audit of text CCO sources and their raw fact snapshot.

The production model consumes text CCOs, while raw facts are useful as an
independent provenance and consistency check.  This module deliberately emits
aggregate counts only: no subject identifier, clinical text, diagnosis name,
or measurement value is copied into its report.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SOURCE_AUDIT_VERSION = "cco-text-fact-source-audit/0.2.0"
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_DEMOGRAPHICS = re.compile(
    r"^\d{1,3}\s+yaşında(?:\s+(?:kadın|erkek)|;\s+cinsiyeti bilinmiyor)\.",
    re.IGNORECASE,
)
_PATIENT_REPORTED_HEADINGS = {
    "Hastalıklar (hasta beyanı)": "patientReportedConditionRecords",
    "Kullandığı ilaçlar (hasta beyanı)": "patientReportedMedicationRecords",
    "Alerjiler": "allergyRecords",
    "Alerjiler (hasta beyanı)": "allergyRecords",
    "Aile öyküsü": "familyHistoryRecords",
    "Aile öyküsü (hasta beyanı)": "familyHistoryRecords",
}


class SourceAuditError(ValueError):
    """Stable error code that never includes source content or identifiers."""


@dataclass(frozen=True)
class TextCcoSource:
    record_id: str
    subject_id: str
    as_of: datetime
    text: str


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceAuditError("duplicate_json_property")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceAuditError("invalid_json_source") from exc


def _aware_datetime(value: Any, code: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SourceAuditError(code)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SourceAuditError(code) from exc
    if parsed.tzinfo is None:
        raise SourceAuditError(code)
    return parsed


def _optional_datetime(value: Any, code: str) -> datetime | None:
    if value is None:
        return None
    return _aware_datetime(value, code)


def _parse_text_cco(payload: Any) -> TextCcoSource:
    if not isinstance(payload, dict):
        raise SourceAuditError("invalid_text_cco")
    required = {"id", "subjectId", "asOf", "text"}
    optional = {"format", "language", "dataClass", "contentHash"}
    if not required.issubset(payload) or set(payload) - required - optional:
        raise SourceAuditError("invalid_text_cco")
    record_id = payload.get("id")
    subject_id = payload.get("subjectId")
    text = payload.get("text")
    if not isinstance(record_id, str) or not _UUID.fullmatch(record_id):
        raise SourceAuditError("invalid_text_cco")
    if not isinstance(subject_id, str) or not _UUID.fullmatch(subject_id):
        raise SourceAuditError("invalid_text_cco")
    if not isinstance(text, str) or not text.strip():
        raise SourceAuditError("invalid_text_cco")
    if payload.get("format") not in {None, "cco-text/1"}:
        raise SourceAuditError("unsupported_text_cco_format")
    if payload.get("language") not in {None, "tr"}:
        raise SourceAuditError("unsupported_text_cco_language")
    return TextCcoSource(
        record_id=record_id.lower(),
        subject_id=subject_id.lower(),
        as_of=_aware_datetime(payload.get("asOf"), "invalid_text_cco_timestamp"),
        text=text.replace("\r\n", "\n").replace("\r", "\n"),
    )


def _corpus_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name.lower()):
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _fact_family(canonical_id: Any) -> str:
    if not isinstance(canonical_id, str) or not canonical_id:
        return "invalid"
    return canonical_id.partition(".")[0]


def _diagnosis_text_status(line: str) -> str:
    if "— kesin tanı;" in line:
        return "definitive"
    if "— ön tanı;" in line:
        return "preliminary"
    return "unknown"


def _matching_diagnosis_line(text: str, display: str, code: str) -> str | None:
    marker = f"{display} ({code})"
    return next((line for line in text.splitlines() if marker in line), None)


def _load_text_corpus(directory: Path) -> tuple[dict[str, TextCcoSource], list[Path]]:
    if not directory.is_dir():
        raise SourceAuditError("text_cco_directory_missing")
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise SourceAuditError("text_cco_corpus_empty")
    records: dict[str, TextCcoSource] = {}
    record_ids: set[str] = set()
    for path in paths:
        record = _parse_text_cco(_read_json(path))
        if record.record_id in record_ids:
            raise SourceAuditError("duplicate_text_cco_id")
        if record.subject_id in records:
            raise SourceAuditError("duplicate_text_cco_subject")
        record_ids.add(record.record_id)
        records[record.subject_id] = record
    return records, paths


def _fact_definition_keys(facts_directory: Path) -> set[tuple[str, str]] | None:
    path = facts_directory / "_fact-definitions.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise SourceAuditError("invalid_fact_definitions")
    result: set[tuple[str, str]] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise SourceAuditError("invalid_fact_definitions")
        canonical_id = item.get("canonicalId")
        version = item.get("version")
        if not isinstance(canonical_id, str) or not isinstance(version, str):
            raise SourceAuditError("invalid_fact_definitions")
        result.add((canonical_id, version))
    return result


def _safe_manifest(facts_directory: Path) -> dict[str, Any]:
    path = facts_directory / "_manifest.json"
    if not path.is_file():
        return {}
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise SourceAuditError("invalid_fact_manifest")
    return payload


def audit_text_cco_and_facts(cco_directory: Path, facts_directory: Path) -> dict[str, Any]:
    """Audit two source corpora and return only aggregate, non-clinical metadata."""
    records, cco_paths = _load_text_corpus(Path(cco_directory))
    facts_directory = Path(facts_directory)
    if not facts_directory.is_dir():
        raise SourceAuditError("facts_directory_missing")
    fact_paths = sorted(
        path for path in facts_directory.glob("*.json") if not path.name.startswith("_")
    )
    if not fact_paths:
        raise SourceAuditError("fact_corpus_empty")

    manifest = _safe_manifest(facts_directory)
    definitions = _fact_definition_keys(facts_directory)
    manifest_completed_at = _optional_datetime(
        manifest.get("completedAt"), "invalid_fact_manifest_timestamp"
    )

    statuses: Counter[str] = Counter()
    families: Counter[str] = Counter()
    patient_sections: Counter[str] = Counter()
    diagnosis_source_status: Counter[str] = Counter()
    diagnosis_text_status: Counter[str] = Counter()
    diagnosis_status_pairs: Counter[str] = Counter()
    fact_ids: set[str] = set()
    subject_files: set[str] = set()
    duplicate_fact_ids = 0
    invalid_intervals = 0
    missing_definition_links = 0
    empty_subjects = 0
    facts_valid_after_cco = 0
    future_window_endpoints = 0
    diagnosis_eligible = 0
    diagnosis_rendered = 0
    preliminary_rendered_definitive = 0
    text_bytes = 0

    for record in records.values():
        text_bytes += len(record.text.encode("utf-8"))
        headings = {
            line[3:].strip() for line in record.text.splitlines() if line.startswith("## ")
        }
        present_patient_sections = {
            report_key
            for heading, report_key in _PATIENT_REPORTED_HEADINGS.items()
            if heading in headings
        }
        for report_key in present_patient_sections:
            patient_sections[report_key] += 1
        first_nonempty = next((line.strip() for line in record.text.splitlines() if line.strip()), "")
        if _DEMOGRAPHICS.match(first_nonempty):
            patient_sections["demographicsRecords"] += 1

    for path in fact_paths:
        subject_id = path.stem.lower()
        if not _UUID.fullmatch(subject_id):
            raise SourceAuditError("invalid_fact_filename")
        if subject_id in subject_files:
            raise SourceAuditError("duplicate_fact_subject")
        subject_files.add(subject_id)
        payload = _read_json(path)
        if not isinstance(payload, list):
            raise SourceAuditError("invalid_fact_subject_file")
        if not payload:
            empty_subjects += 1
        record = records.get(subject_id)

        for fact in payload:
            if not isinstance(fact, dict):
                raise SourceAuditError("invalid_fact_row")
            fact_id = fact.get("factId")
            if not isinstance(fact_id, str) or not _UUID.fullmatch(fact_id):
                raise SourceAuditError("invalid_fact_row")
            lowered_fact_id = fact_id.lower()
            if lowered_fact_id in fact_ids:
                duplicate_fact_ids += 1
            fact_ids.add(lowered_fact_id)

            status = fact.get("evaluationStatus")
            statuses[str(status)] += 1
            canonical_id = fact.get("canonicalId")
            version = fact.get("version")
            families[_fact_family(canonical_id)] += 1
            if definitions is not None and (canonical_id, version) not in definitions:
                missing_definition_links += 1

            valid_from = _aware_datetime(
                fact.get("validFrom"), "invalid_fact_validity_timestamp"
            )
            valid_until = _optional_datetime(
                fact.get("validUntil"), "invalid_fact_validity_timestamp"
            )
            occurred_from = _optional_datetime(
                fact.get("occurredFrom"), "invalid_fact_occurrence_timestamp"
            )
            occurred_to = _optional_datetime(
                fact.get("occurredTo"), "invalid_fact_occurrence_timestamp"
            )
            if valid_until is not None and valid_until < valid_from:
                invalid_intervals += 1
            if (
                occurred_from is not None
                and occurred_to is not None
                and occurred_to < occurred_from
            ):
                invalid_intervals += 1
            if (
                manifest_completed_at is not None
                and occurred_to is not None
                and occurred_to > manifest_completed_at
            ):
                future_window_endpoints += 1
            if record is not None and valid_from > record.as_of:
                facts_valid_after_cco += 1

            if (
                record is None
                or canonical_id != "diagnosis.recorded-item"
                or status != "success"
                or valid_from > record.as_of
                or (valid_until is not None and valid_until <= record.as_of)
            ):
                continue
            value = fact.get("value")
            coded = value.get("coded") if isinstance(value, dict) else None
            attributes = coded.get("attributes") if isinstance(coded, dict) else None
            display = coded.get("display") if isinstance(coded, dict) else None
            source_status = attributes.get("type") if isinstance(attributes, dict) else None
            evaluation_key = fact.get("evaluationKey")
            if not isinstance(display, str) or not isinstance(evaluation_key, str):
                continue
            code = evaluation_key.removeprefix("code:icd-10:")
            if not code or code == evaluation_key:
                continue
            diagnosis_eligible += 1
            diagnosis_source_status[str(source_status)] += 1
            line = _matching_diagnosis_line(record.text, display, code)
            if line is None:
                continue
            diagnosis_rendered += 1
            rendered_status = _diagnosis_text_status(line)
            diagnosis_text_status[rendered_status] += 1
            diagnosis_status_pairs[f"{source_status}->{rendered_status}"] += 1
            if source_status == "preliminary" and rendered_status == "definitive":
                preliminary_rendered_definitive += 1

    cco_subjects = set(records)
    matched_subjects = cco_subjects & subject_files
    missing_fact_subjects = cco_subjects - subject_files
    orphan_fact_subjects = subject_files - cco_subjects
    historical_reconstruction = manifest.get("historicalCcoReconstruction") is True
    include_earlier_days = manifest.get("includeEarlierDays") is True

    integrity_errors = (
        duplicate_fact_ids
        + invalid_intervals
        + missing_definition_links
        + len(missing_fact_subjects)
        + len(orphan_fact_subjects)
    )
    return {
        "schemaVersion": SOURCE_AUDIT_VERSION,
        "valid": integrity_errors == 0,
        "disposition": "legacy_audit_only" if integrity_errors == 0 else "blocked",
        "privacy": {"aggregateOnly": True, "clinicalContentEmitted": False},
        "textCco": {
            "records": len(records),
            "uniqueSubjects": len(cco_subjects),
            "bytes": text_bytes,
            "asOfMin": min(record.as_of for record in records.values()).isoformat(),
            "asOfMax": max(record.as_of for record in records.values()).isoformat(),
            "corpusSha256": _corpus_digest(cco_paths),
            **{
                key: patient_sections[key]
                for key in sorted(_PATIENT_REPORTED_HEADINGS.values())
            },
            "demographicsRecords": patient_sections["demographicsRecords"],
        },
        "facts": {
            "sourceRole": "legacy_audit_only",
            "goldLabelsAllowed": False,
            "clinicalStateInferenceAllowed": False,
            "activeMedicationInferenceAllowed": False,
            "assumedSameSnapshotAsTextCco": False,
            "assumedSameRendererVersion": False,
            "subjectFiles": len(subject_files),
            "totalRows": sum(statuses.values()),
            "evaluationStatus": dict(sorted(statuses.items())),
            "families": dict(sorted(families.items())),
            "emptySubjects": empty_subjects,
            "duplicateFactIds": duplicate_fact_ids,
            "invalidIntervals": invalid_intervals,
            "missingDefinitionLinks": missing_definition_links,
            "validAfterCcoAsOf": facts_valid_after_cco,
            "windowEndsAfterExport": future_window_endpoints,
            "historicalCcoReconstruction": historical_reconstruction,
            "includeEarlierDays": include_earlier_days,
            "corpusSha256": _corpus_digest(fact_paths),
        },
        "alignment": {
            "matchedSubjects": len(matched_subjects),
            "missingFactSubjects": len(missing_fact_subjects),
            "orphanFactSubjects": len(orphan_fact_subjects),
        },
        "legacyComparisonDiagnostics": {
            "eligibleSourceFacts": diagnosis_eligible,
            "renderedWithExactCode": diagnosis_rendered,
            "sourceStatus": dict(sorted(diagnosis_source_status.items())),
            "textStatus": dict(sorted(diagnosis_text_status.items())),
            "statusPairs": dict(sorted(diagnosis_status_pairs.items())),
            "preliminaryRenderedDefinitive": preliminary_rendered_definitive,
        },
        "legacyObservations": {
            "validAfterCcoAsOf": facts_valid_after_cco,
            "preliminaryRenderedDefinitive": preliminary_rendered_definitive,
            "windowEndsAfterExport": future_window_endpoints,
            "emptySubjects": empty_subjects,
            "historicalCcoReconstruction": historical_reconstruction,
            "includeEarlierDays": include_earlier_days,
        },
        "warnings": [],
        "notices": [
            "legacy_fact_source_not_eligible_for_gold_labels",
            "legacy_fact_differences_are_not_text_cco_errors",
        ],
        "clinicalCorrectnessEvaluated": False,
    }
