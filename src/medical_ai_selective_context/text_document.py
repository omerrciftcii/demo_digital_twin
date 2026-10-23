"""cco-text/1 records: one Turkish markdown summary, not a CCO 0.7 fact graph.

The chart is one user message. A prompt that would pass the context window
drops the least relevant sections first. Sections exist so a citation can name
a heading and a line; they are not a second copy of the chart.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .store import canonical_json

FORMAT = "cco-text/1"
EVIDENCE_ID_VERSION = "cco-text-evidence/1"
MAX_TEXT_BYTES = 64 * 1024
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^##\s+(\S.*?)\s*$")
_REQUIRED_FIELDS = ("id", "subjectId", "asOf", "text")
_OPTIONAL_FIELDS = ("format", "language")
_FIELDS = (*_REQUIRED_FIELDS, *_OPTIONAL_FIELDS)
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "text" / "patients"
EXAMPLE_FIXTURE = FIXTURE_DIR.parent / "example-facts.json"


class TextRecordError(ValueError):
    """Stable code only. The message must not echo clinical text."""


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TextRecordError("invalid_text_record")
        result[key] = value
    return result


@dataclass(frozen=True)
class SourceLine:
    number: int
    text: str
    section_index: int


@dataclass(frozen=True)
class EvidenceSpan:
    evidence_id: str
    snapshot_id: str
    content_hash: str
    section_heading: str
    start_line: int
    end_line: int
    text_sha256: str
    text: str

    def location(self) -> dict[str, Any]:
        """Return provenance without copying clinical text into annotations."""
        return {
            "snapshotId": self.snapshot_id,
            "contentHash": self.content_hash,
            "sectionHeading": self.section_heading,
            "startLine": self.start_line,
            "endLine": self.end_line,
            "textSha256": self.text_sha256,
        }


@dataclass
class Section:
    heading: str
    start_line: int
    end_line: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class TextRecord:
    id: str
    subject_id: str
    as_of: str
    language: str
    text: str
    content_hash: str
    data_class: str
    sections: list[Section]
    lines: list[SourceLine]

    def summary(self) -> dict[str, Any]:
        return {
            "recordId": self.id,
            "subjectId": self.subject_id,
            "asOf": self.as_of,
            "language": self.language,
            "format": FORMAT,
            "bytes": len(self.text.encode("utf-8")),
            "dataClass": self.data_class,
            "contentHash": self.content_hash,
            "sectionCount": len(self.sections),
        }

    def public_document(self) -> dict[str, Any]:
        body = self.summary()
        body["text"] = self.text
        body["sections"] = [
            {"heading": section.heading, "startLine": section.start_line, "endLine": section.end_line}
            for section in self.sections
        ]
        return body


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_sections(text: str) -> tuple[list[Section], list[SourceLine]]:
    sections: list[Section] = []
    lines: list[SourceLine] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        heading = _HEADING.match(raw)
        if heading or not sections:
            sections.append(Section(
                heading=heading.group(1) if heading else "Giriş",
                start_line=number, end_line=number,
            ))
        section = sections[-1]
        section.end_line = number
        section.lines.append(raw)
        lines.append(SourceLine(number, raw, len(sections) - 1))
    return sections, lines


def _require_uuid(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        raise TextRecordError(code)
    return value.lower()


def parse_cco_text(payload: Any) -> TextRecord:
    """Validate the wrapper. Missing fields stay missing; nothing is invented."""
    if not isinstance(payload, dict):
        raise TextRecordError("invalid_text_record")
    supplied = set(payload)
    if not set(_REQUIRED_FIELDS).issubset(supplied):
        raise TextRecordError("invalid_text_record")
    if supplied - set(_REQUIRED_FIELDS) - set(_OPTIONAL_FIELDS):
        raise TextRecordError("invalid_text_record")
    if payload.get("format", FORMAT) != FORMAT:
        raise TextRecordError("unsupported_text_format")
    if payload.get("language", "tr") != "tr":
        raise TextRecordError("unsupported_language")
    record_id = _require_uuid(payload.get("id"), "invalid_text_record")
    subject_id = _require_uuid(payload.get("subjectId"), "invalid_text_record")
    as_of = payload.get("asOf")
    if not isinstance(as_of, str) or not as_of.strip() or len(as_of) > 40:
        raise TextRecordError("invalid_text_timestamp")
    try:
        parsed = datetime.fromisoformat(as_of)
    except ValueError as exc:
        raise TextRecordError("invalid_text_timestamp") from exc
    if parsed.tzinfo is None:
        raise TextRecordError("invalid_text_timestamp")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise TextRecordError("invalid_text_record")
    text = _normalize_newlines(text)
    size = len(text.encode("utf-8"))
    if size > MAX_TEXT_BYTES:
        raise TextRecordError("text_too_large")
    canonical = {
        "format": FORMAT,
        "id": record_id,
        "subjectId": subject_id,
        "asOf": as_of,
        "language": "tr",
        "text": text,
    }
    canonical["text"] = text
    try:
        digest = hashlib.sha256(canonical_json(canonical).encode("utf-8")).hexdigest()
    except (TypeError, ValueError) as exc:
        raise TextRecordError("invalid_text_record") from exc
    sections, lines = _split_sections(text)
    return TextRecord(
        id=record_id, subject_id=subject_id, as_of=as_of, language="tr", text=text,
        content_hash=digest, data_class="real", sections=sections, lines=lines,
    )


def build_evidence_spans(record: TextRecord) -> list[EvidenceSpan]:
    """Create stable, snapshot-bound evidence IDs for non-heading source lines."""
    spans: list[EvidenceSpan] = []
    for source_line in record.lines:
        text = source_line.text.strip()
        if not text or _HEADING.match(text):
            continue
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identity = canonical_json({
            "version": EVIDENCE_ID_VERSION,
            "snapshotId": record.id,
            "contentHash": record.content_hash,
            "startLine": source_line.number,
            "endLine": source_line.number,
            "textSha256": text_sha256,
        })
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        spans.append(EvidenceSpan(
            evidence_id=f"evi_{digest}",
            snapshot_id=record.id,
            content_hash=record.content_hash,
            section_heading=record.sections[source_line.section_index].heading,
            start_line=source_line.number,
            end_line=source_line.number,
            text_sha256=text_sha256,
            text=text,
        ))
    return spans


def evidence_span_index(record: TextRecord) -> dict[str, EvidenceSpan]:
    return {span.evidence_id: span for span in build_evidence_spans(record)}


def extract_explicit_claims(section_heading: str, text: str) -> set[str]:
    """Extract only literal status markers; absence always remains unspecified."""
    folded_heading = section_heading.casefold()
    folded = text.casefold()
    claims: set[str] = set()
    if "hasta beyanı" in folded_heading:
        claims.add("patient_reported")
    if "reçete kaydı" in folded_heading:
        claims.add("medication_prescribed")
    if "— kesin tanı" in folded:
        claims.add("assertion_definitive")
    if "— ön tanı" in folded:
        claims.add("assertion_preliminary")
    if any(marker in folded for marker in (
        "bıraktığını belirtti", "kullanımı sonlandırıldı", "ilaç kesildi",
    )):
        claims.add("medication_stopped")
    if any(marker in folded for marker in (
        "halen kullandığını belirtti", "aktif olarak kullanıyor", "kullanmaya devam ediyor",
    )):
        claims.add("medication_active")
    if any(marker in folded for marker in (
        "şikayetlerinde gerileme oldu", "belirgin gerileme", "klinik iyileşme",
    )):
        claims.add("course_improved")
    if any(marker in folded for marker in (
        "tamamen düzeldi", "tam olarak iyileşti", "durum sonlandı",
    )):
        claims.add("course_resolved")
    if any(marker in folded for marker in (
        "nedeni olduğu belirtildi", "nedensel ilişki kuruldu", "bağlı olduğu belirtildi",
    )):
        claims.add("causality_explicit")
    return claims


def loads_text_payload(raw: bytes) -> Any:
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_keys)
    except (json.JSONDecodeError, UnicodeError, TextRecordError) as exc:
        raise TextRecordError("invalid_text_record") from exc
    if not isinstance(payload, dict):
        raise TextRecordError("invalid_text_record")
    return payload


def fixture_hashes() -> set[str]:
    """Content hashes of the shipped synthetic patients. Clients cannot self-label."""
    found: set[str] = set()
    paths = sorted(FIXTURE_DIR.glob("*.json")) if FIXTURE_DIR.is_dir() else []
    if EXAMPLE_FIXTURE.is_file():
        paths.append(EXAMPLE_FIXTURE)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        found.add(parse_cco_text(payload).content_hash)
    return found


class TextRecordStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.records: dict[str, TextRecord] = {}
        self._synthetic = fixture_hashes()

    @property
    def count(self) -> int:
        return len(self.records)

    def load(self) -> None:
        self.records.clear()
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
            document = {key: payload[key] for key in _REQUIRED_FIELDS}
            document.update({key: payload[key] for key in _OPTIONAL_FIELDS if key in payload})
            record = parse_cco_text(document)
            # The on-disk label is not trusted. Only the shipped fixture hash is synthetic.
            record.data_class = "synthetic" if record.content_hash in self._synthetic else "real"
            self.records[record.id] = record

    def import_payload(self, payload: Any) -> tuple[TextRecord, bool]:
        record = parse_cco_text(payload)
        record.data_class = "synthetic" if record.content_hash in self._synthetic else "real"
        existing = self.records.get(record.id)
        if existing is not None:
            if existing.content_hash != record.content_hash:
                raise TextRecordError("record_id_conflict")
            return existing, False
        self.records[record.id] = record
        self._write(record)
        return record, True

    def get(self, record_id: str) -> TextRecord | None:
        return self.records.get(record_id)

    def list_summaries(self) -> list[dict[str, Any]]:
        return [record.summary() for record in self.records.values()]

    def _write(self, record: TextRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        body = {
            "format": FORMAT, "id": record.id, "subjectId": record.subject_id,
            "asOf": record.as_of, "language": record.language, "text": record.text,
            "dataClass": record.data_class, "contentHash": record.content_hash,
        }
        target = self.directory / f"{record.id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
