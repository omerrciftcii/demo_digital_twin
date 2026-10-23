"""Deterministic event-graph -> text CCO -> gold annotation pipeline.

No external model is called. Synthetic gold is attached to graph event IDs,
then resolved against the rendered text spans so graph/render drift is caught.
"""

from __future__ import annotations

import json
import random
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .text_document import (
    EvidenceSpan,
    build_evidence_spans,
    extract_explicit_claims,
    parse_cco_text,
)
from .training_dataset import MANIFEST_VERSION, SCHEMA_VERSION

EVENT_GRAPH_VERSION = "cco-synthetic-event-graph/0.1.0"
_NAMESPACE = uuid.UUID("728f696e-8508-4bda-bde3-fc16042dc8f8")


class SyntheticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SyntheticEvent(SyntheticModel):
    event_id: str = Field(alias="eventId", min_length=3, max_length=100)
    kind: Literal[
        "diagnosis", "medication_prescription", "medication_reported", "laboratory",
        "wearable", "procedure",
    ]
    display: str = Field(min_length=1, max_length=240)
    occurred_on: date = Field(alias="occurredOn")
    code: str | None = Field(default=None, max_length=30)
    assertion: Literal["definitive", "preliminary"] | None = None
    diagnosis_group: Literal["chronic", "other", "acute"] | None = Field(
        default=None, alias="diagnosisGroup"
    )
    medication_state: Literal["active", "stopped"] | None = Field(
        default=None, alias="medicationState"
    )
    course: Literal["improved", "resolved"] | None = None
    value: str | None = Field(default=None, max_length=80)
    unit: str | None = Field(default=None, max_length=40)
    requiredness: Literal["must_use", "helpful", "not_needed", "harmful"]
    use_role: Literal[
        "direct_fact", "background", "safety", "counterevidence", "status_check", "none",
    ] = Field(alias="useRole")
    causal_strength: Literal["none", "possible", "supported"] = Field(
        default="none", alias="causalStrength"
    )
    safety_critical: bool = Field(default=False, alias="safetyCritical")
    reason_code: str = Field(alias="reasonCode", min_length=3, max_length=80)

    @model_validator(mode="after")
    def event_semantics(self) -> SyntheticEvent:
        if self.kind == "diagnosis" and (self.assertion is None or self.diagnosis_group is None):
            raise ValueError("Diagnosis events require explicit assertion and group")
        if self.kind != "diagnosis" and (
            self.assertion is not None or self.diagnosis_group is not None
        ):
            raise ValueError("Only diagnosis events may carry diagnosis status")
        if self.medication_state is not None and self.kind != "medication_reported":
            raise ValueError("Medication state requires an explicit patient-reported event")
        if self.requiredness in {"not_needed", "harmful"} and self.use_role != "none":
            raise ValueError("Excluded events cannot have a use role")
        if self.safety_critical and self.requiredness != "must_use":
            raise ValueError("Safety-critical events must be required")
        return self


class SyntheticScenario(SyntheticModel):
    schema_version: Literal[EVENT_GRAPH_VERSION] = Field(
        default=EVENT_GRAPH_VERSION, alias="schemaVersion"
    )
    scenario_id: str = Field(alias="scenarioId", min_length=3, max_length=127)
    snapshot_id: str = Field(alias="snapshotId", pattern=r"^[0-9a-fA-F-]{36}$")
    subject_id: str = Field(alias="subjectId", pattern=r"^[0-9a-fA-F-]{36}$")
    as_of: datetime = Field(alias="asOf")
    age: int = Field(ge=18, le=100)
    sex: Literal["kadın", "erkek", "bilinmiyor"]
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=8000)
    intents: list[Literal[
        "lookup", "interpretation", "guidance", "triage", "clarification", "follow_up",
    ]] = Field(min_length=1, max_length=4)
    subjects: list[Literal[
        "symptom", "wearable", "laboratory", "medication", "daily_activity", "diagnosis",
        "procedure", "general_health",
    ]] = Field(min_length=1, max_length=4)
    personalization_decision: Literal["required", "helpful", "not_supported"] = Field(
        alias="personalizationDecision"
    )
    events: list[SyntheticEvent] = Field(min_length=1, max_length=30)
    followup_questions: list[str] = Field(
        default_factory=list, alias="followupQuestions", max_length=2
    )
    causal_claim_allowed: bool = Field(default=False, alias="causalClaimAllowed")
    uncertainty_required: bool = Field(default=True, alias="uncertaintyRequired")

    @model_validator(mode="after")
    def unique_events(self) -> SyntheticScenario:
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("Synthetic event IDs must be unique")
        positive = [
            event for event in self.events
            if event.requiredness in {"must_use", "helpful"}
        ]
        if self.personalization_decision == "required" and not any(
            event.requiredness == "must_use" for event in positive
        ):
            raise ValueError("Required personalization needs required graph evidence")
        if self.personalization_decision == "helpful" and (
            not positive or any(event.requiredness == "must_use" for event in positive)
        ):
            raise ValueError("Helpful personalization needs only helpful graph evidence")
        if self.personalization_decision == "not_supported" and positive:
            raise ValueError("Unsupported personalization cannot have positive graph evidence")
        return self


@dataclass(frozen=True)
class RenderedScenario:
    cco: dict[str, Any]
    event_evidence: dict[str, EvidenceSpan]


@dataclass(frozen=True)
class SyntheticPackage:
    project_root: Path
    package_root: Path
    annotations: Path
    event_graphs: Path
    manifest: Path


@dataclass(frozen=True)
class GoldReviewPackage:
    package_root: Path
    tasks: Path
    manifest: Path


def _date(value: date) -> str:
    return value.strftime("%d.%m.%Y")


def _diagnosis_line(event: SyntheticEvent) -> str:
    code = f" ({event.code})" if event.code else ""
    assertion = "kesin tanı" if event.assertion == "definitive" else "ön tanı"
    suffix = ""
    if event.course == "improved":
        suffix = " Klinik iyileşme açıkça kaydedildi."
    elif event.course == "resolved":
        suffix = " Durumun tamamen düzeldiği açıkça kaydedildi."
    return f"- {event.display}{code} — {assertion}; son kayıt {_date(event.occurred_on)}.{suffix}"


def _event_line(event: SyntheticEvent) -> str:
    if event.kind == "diagnosis":
        return _diagnosis_line(event)
    if event.kind == "medication_prescription":
        return f"- {event.display} — son reçete {_date(event.occurred_on)}, 1 reçete."
    if event.kind == "medication_reported":
        state = ""
        if event.medication_state == "active":
            state = " halen kullandığını belirtti"
        elif event.medication_state == "stopped":
            state = " bıraktığını belirtti"
        return f"- {event.display} ({_date(event.occurred_on)} tarihinde{state})."
    if event.kind == "laboratory":
        measurement = " ".join(part for part in (event.value, event.unit) if part)
        return f"- {event.display} {measurement} ({_date(event.occurred_on)}).".replace("  ", " ")
    if event.kind == "wearable":
        measurement = " ".join(part for part in (event.value, event.unit) if part)
        return f"- {event.display}: {measurement} ({_date(event.occurred_on)}).".replace("  ", " ")
    return f"- {event.display} — {_date(event.occurred_on)}."


def _section_for(event: SyntheticEvent) -> str:
    return {
        "diagnosis": "Tanılar (hastane kaydı)",
        "medication_prescription": "İlaçlar (reçete kaydı)",
        "medication_reported": "Kullandığı ilaçlar (hasta beyanı)",
        "laboratory": "Laboratuvar (hastane)",
        "wearable": "Uyku (yüzük)",
        "procedure": "Ameliyat ve işlemler",
    }[event.kind]


def render_event_graph(scenario: SyntheticScenario) -> RenderedScenario:
    """Render only explicit graph fields and resolve every event to its source span."""
    sex = scenario.sex if scenario.sex != "bilinmiyor" else "cinsiyeti bilinmiyor"
    intro = f"{scenario.age} yaşında {sex}." if scenario.sex != "bilinmiyor" else (
        f"{scenario.age} yaşında; cinsiyeti bilinmiyor."
    )
    lines = [intro]
    event_lines: dict[str, int] = {}
    section_order = [
        "Tanılar (hastane kaydı)", "İlaçlar (reçete kaydı)",
        "Kullandığı ilaçlar (hasta beyanı)", "Ameliyat ve işlemler",
        "Laboratuvar (hastane)", "Uyku (yüzük)",
    ]
    by_section: dict[str, list[SyntheticEvent]] = {name: [] for name in section_order}
    for event in scenario.events:
        by_section[_section_for(event)].append(event)
    for section in section_order:
        events = by_section[section]
        if not events:
            continue
        lines.extend(["", f"## {section}"])
        if section == "Tanılar (hastane kaydı)":
            group_labels = {"chronic": "Kronik", "other": "Diğer", "acute": "Akut"}
            for group in ("chronic", "other", "acute"):
                grouped = [event for event in events if event.diagnosis_group == group]
                if not grouped:
                    continue
                lines.extend([f"{group_labels[group]}:"])
                for event in grouped:
                    lines.append(_event_line(event))
                    event_lines[event.event_id] = len(lines)
        else:
            for event in events:
                lines.append(_event_line(event))
                event_lines[event.event_id] = len(lines)
    cco = {
        "id": scenario.snapshot_id,
        "subjectId": scenario.subject_id,
        "asOf": scenario.as_of.isoformat(),
        "text": "\n".join(lines),
    }
    record = parse_cco_text(cco)
    spans_by_line = {span.start_line: span for span in build_evidence_spans(record)}
    event_evidence: dict[str, EvidenceSpan] = {}
    for event_id, line_number in event_lines.items():
        if line_number not in spans_by_line:
            raise ValueError("Rendered event could not be resolved to a text source span")
        event_evidence[event_id] = spans_by_line[line_number]
    return RenderedScenario(cco=cco, event_evidence=event_evidence)


def build_synthetic_example(
    scenario: SyntheticScenario,
    rendered: RenderedScenario,
    *,
    split: Literal["train", "dev", "test"],
    cco_file: str | None = None,
    root_id: str | None = None,
    variant: Literal[
        "base", "relevant_counterfactual", "irrelevant_counterfactual",
        "order_paraphrase", "controlled_corruption", "human_gold",
    ] = "base",
    generator: str = "controlled-event-graph-renderer",
    parent_example_id: str | None = None,
    trainable: bool = False,
) -> dict[str, Any]:
    """Build gold from graph events and verify claims against rendered spans."""
    record = parse_cco_text(rendered.cco)
    annotations = []
    for event in scenario.events:
        span = rendered.event_evidence[event.event_id]
        claims = sorted(extract_explicit_claims(span.section_heading, span.text))
        annotations.append({
            "evidenceId": span.evidence_id,
            "sourceSpan": span.location(),
            "factId": None,
            "requiredness": event.requiredness,
            "useRole": event.use_role,
            "causalStrength": event.causal_strength,
            "safetyCritical": event.safety_critical,
            "reasonCode": event.reason_code,
            "explicitClaims": claims,
            "independentClinicalValidation": False,
        })
    return {
        "schemaVersion": SCHEMA_VERSION,
        "exampleId": scenario.scenario_id,
        "rootId": root_id or scenario.scenario_id,
        "ccoFile": cco_file or f"training_v2/generated/sources/{scenario.snapshot_id}.json",
        "snapshotId": scenario.snapshot_id,
        "contentHash": record.content_hash,
        "split": split,
        "variant": variant,
        "question": {
            "text": scenario.question,
            "intents": scenario.intents,
            "subjects": scenario.subjects,
        },
        "personalizationDecision": scenario.personalization_decision,
        "evidenceAnnotations": annotations,
        "alternativeEvidenceSets": [],
        "criticalEvidenceBundles": [
            [rendered.event_evidence[event.event_id].evidence_id]
            for event in scenario.events if event.safety_critical
        ],
        "followupPlan": {
            "needed": bool(scenario.followup_questions),
            "questions": scenario.followup_questions,
        },
        "answerPolicy": {
            "personalizedAnswerExpected": scenario.personalization_decision != "not_supported",
            "causalClaimAllowed": scenario.causal_claim_allowed,
            "uncertaintyDisclosureRequired": scenario.uncertainty_required,
            "directAnswerFirst": True,
            "unspecifiedStatePolicy": "remain_unspecified",
            "sourceTextClinicallyValidated": False,
            "legacyFactCanDefineGold": False,
        },
        "trainable": trainable,
        "candidateAnswer": scenario.answer if not trainable else None,
        "goldAnswer": scenario.answer if trainable else None,
        "provenance": {
            "sourceType": "synthetic",
            "reviewStatus": "unreviewed",
            "generator": generator,
            "parentExampleId": parent_example_id,
            "parserVersion": "cco-text-evidence/1",
        },
        "corruption": None,
    }


def _ids(index: int, seed: int) -> tuple[str, str]:
    key = f"{seed}:{index}"
    return str(uuid.uuid5(_NAMESPACE, f"snapshot:{key}")), str(
        uuid.uuid5(_NAMESPACE, f"subject:{key}")
    )


def _scenario_for(index: int, seed: int, rng: random.Random) -> SyntheticScenario:
    snapshot_id, subject_id = _ids(index, seed)
    scenario_id = f"synthetic-{index + 1:06d}"
    as_of = datetime.fromisoformat("2026-10-21T12:00:00+03:00")
    recent = as_of.date() - timedelta(days=rng.randint(2, 40))
    old = as_of.date() - timedelta(days=rng.randint(900, 2200))
    distractor = SyntheticEvent(
        eventId=f"{scenario_id}-old-acute",
        kind="diagnosis",
        display="Geçmiş akut solunum yolu enfeksiyonu",
        occurredOn=old,
        code="J06.9",
        assertion="definitive",
        diagnosisGroup="acute",
        requiredness="not_needed",
        useRole="none",
        reasonCode="unrelated_old_acute_record",
    )
    template = index % 6
    common = {
        "scenarioId": scenario_id,
        "snapshotId": snapshot_id,
        "subjectId": subject_id,
        "asOf": as_of,
        "age": 30 + index % 51,
        "sex": "kadın" if index % 2 == 0 else "erkek",
        "personalizationDecision": "required",
    }
    if index % 20 == 19:
        return SyntheticScenario(
            **{**common, "personalizationDecision": "not_supported"},
            question="Kayıtlarımda lateks alerjisi bulunuyor mu?",
            answer="Kayıtta lateks alerjisini destekleyen bir ifade bulunmuyor.",
            intents=["lookup"],
            subjects=["general_health"],
            events=[distractor],
        )
    if template == 0:
        value = str(5 + index % 3).replace(".", ",")
        event = SyntheticEvent(
            eventId=f"{scenario_id}-lab", kind="laboratory", display="HbA1c",
            occurredOn=recent, value=value, unit="%", requiredness="must_use",
            useRole="direct_fact", reasonCode="latest_hba1c_value",
        )
        return SyntheticScenario(
            **common, question="Son HbA1c sonucum neydi?",
            answer=f"Son HbA1c kaydın %{value} ve tarihi {_date(recent)}.",
            intents=["lookup"], subjects=["laboratory"], events=[event, distractor],
            uncertaintyRequired=False,
        )
    if template == 1:
        event = SyntheticEvent(
            eventId=f"{scenario_id}-sleep", kind="wearable",
            display="Son ana uyku süresi", occurredOn=old, value="5 sa 42 dk",
            requiredness="must_use", useRole="status_check",
            reasonCode="sleep_duration_with_old_date",
        )
        return SyntheticScenario(
            **common, question="Dün gece ne kadar uyudum?",
            answer=(f"Dün geceye ait kayıt yok. Son uyku kaydın {_date(old)} tarihindeki "
                    "5 saat 42 dakikalık ölçüm."),
            intents=["lookup", "interpretation"], subjects=["wearable"],
            events=[event, distractor], uncertaintyRequired=False,
        )
    if template == 2:
        event = SyntheticEvent(
            eventId=f"{scenario_id}-rx", kind="medication_prescription",
            display="Örnek ağrı kesici", occurredOn=old, requiredness="must_use",
            useRole="status_check", safetyCritical=True,
            reasonCode="old_prescription_not_current_use",
        )
        return SyntheticScenario(
            **common, question="Eski ağrı kesiciyi bugün kullanabilir miyim?",
            answer=(f"Kayıt yalnız {_date(old)} tarihli reçeteyi gösteriyor; bugün aktif "
                    "kullandığını veya yeniden başlamanın uygun olduğunu göstermiyor."),
            intents=["guidance", "clarification"], subjects=["medication"],
            events=[event, distractor],
            followupQuestions=["Bu ilacı şu anda kullanıyor musun?"],
        )
    if template == 3:
        event = SyntheticEvent(
            eventId=f"{scenario_id}-preliminary", kind="diagnosis", display="Migren",
            occurredOn=recent, code="G43", assertion="preliminary", diagnosisGroup="other",
            requiredness="must_use", useRole="direct_fact",
            reasonCode="explicit_preliminary_diagnosis",
        )
        return SyntheticScenario(
            **common, question="Migren tanım kesinleşmiş mi?",
            answer=(f"Hayır. Migren {_date(recent)} tarihli kayıtta ön tanı olarak geçiyor; "
                    "kesinleşmiş tanı olarak sunamam."),
            intents=["lookup", "interpretation"], subjects=["diagnosis"],
            events=[event, distractor], uncertaintyRequired=False,
        )
    if template == 4:
        event = SyntheticEvent(
            eventId=f"{scenario_id}-procedure", kind="procedure",
            display="Safra kesesi ameliyatı", occurredOn=old, requiredness="must_use",
            useRole="direct_fact", reasonCode="recorded_procedure",
        )
        return SyntheticScenario(
            **common, question="Safra kesesi ameliyatı geçirmiş miyim?",
            answer=f"Evet, kayıtta {_date(old)} tarihli safra kesesi ameliyatı bulunuyor.",
            intents=["lookup"], subjects=["procedure"], events=[event, distractor],
            uncertaintyRequired=False,
        )
    event = SyntheticEvent(
        eventId=f"{scenario_id}-history", kind="diagnosis",
        display="Kronik tiroid işlev bozukluğu",
        occurredOn=old, assertion="definitive", diagnosisGroup="chronic",
        requiredness="must_use", useRole="background", causalStrength="possible",
        reasonCode="relevant_persistent_history",
    )
    return SyntheticScenario(
        **common, question="Yorgunluğumda tiroid geçmişim önemli olabilir mi?",
        answer=("Tiroid geçmişin değerlendirmede önemli olabilir; ancak kayıt tek başına "
                "bugünkü yorgunluğun nedeninin bu olduğunu göstermiyor."),
        intents=["interpretation", "clarification"], subjects=["symptom", "diagnosis"],
        events=[event, distractor], causalClaimAllowed=False,
        followupQuestions=["Güncel TSH veya serbest T4 sonucun var mı?"],
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate_synthetic_pilot(
    project_root: Path,
    *,
    count: int = 480,
    seed: int = 20261021,
    package_name: str = "generated",
) -> SyntheticPackage:
    """Generate a local synthetic package without network or real patient data."""
    if count < 1:
        raise ValueError("Synthetic pilot count must be positive")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,39}", package_name):
        raise ValueError("Synthetic package name is invalid")
    project_root = Path(project_root).resolve()
    package_root = project_root / "data" / "training_v2" / package_name
    sources = package_root / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    annotations_path = package_root / "annotations.jsonl"
    graphs_path = package_root / "event-graphs.jsonl"
    manifest_path = package_root / "manifest.json"
    rng = random.Random(seed)
    annotations: list[str] = []
    graphs: list[str] = []
    for index in range(count):
        scenario = _scenario_for(index, seed, rng)
        rendered = render_event_graph(scenario)
        relative_source = (
            f"training_v2/{package_name}/sources/{scenario.snapshot_id}.json"
        )
        _write_json(sources / f"{scenario.snapshot_id}.json", rendered.cco)
        split: Literal["train", "dev", "test"]
        bucket = index % 10
        split = "dev" if bucket == 8 else "test" if bucket == 9 else "train"
        annotation = build_synthetic_example(
            scenario, rendered, split=split, cco_file=relative_source
        )
        annotations.append(json.dumps(annotation, ensure_ascii=False, separators=(",", ":")))
        graphs.append(json.dumps(
            scenario.model_dump(by_alias=True, mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        ))
    annotations_path.write_text("\n".join(annotations) + "\n", encoding="utf-8")
    graphs_path.write_text("\n".join(graphs) + "\n", encoding="utf-8")
    _write_json(manifest_path, {
        "schemaVersion": MANIFEST_VERSION,
        "datasetId": f"controlled-{package_name}-{count}-{seed}",
        "purpose": "answer_sft",
        "policy": {
            "personalizationPriority": True,
            "primarySource": "text_cco",
            "legacyFactRole": "legacy_audit_only",
            "maxNotSupportedFraction": 0.1,
            "minReviewedFraction": 0.0,
        },
    })
    return SyntheticPackage(
        project_root=project_root,
        package_root=package_root,
        annotations=annotations_path,
        event_graphs=graphs_path,
        manifest=manifest_path,
    )


def prepare_gold_review_queue(
    cco_directory: Path,
    output_directory: Path,
    *,
    count: int = 60,
    questions_path: Path | None = None,
) -> GoldReviewPackage:
    """Create a local metadata-only review queue; it does not create gold labels."""
    if count < 1:
        raise ValueError("Gold review count must be positive")
    cco_directory = Path(cco_directory)
    paths = sorted(cco_directory.glob("*.json"))
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    tasks_path = output_directory / "tasks.jsonl"
    manifest_path = output_directory / "manifest.json"
    records = []
    records_by_id = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        record = parse_cco_text(payload)
        records.append((record.content_hash, path, record))
        records_by_id[record.id] = (path, record)
    records.sort(key=lambda item: item[0])
    candidates: list[tuple[Path, Any, dict[str, Any] | None]] = []
    if questions_path is not None:
        with Path(questions_path).open(encoding="utf-8-sig") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Invalid gold review question row")
                cco_id = row.get("cco_id")
                question = row.get("question")
                if not isinstance(cco_id, str) or not isinstance(question, str):
                    raise ValueError("Invalid gold review question row")
                matched = records_by_id.get(cco_id.lower())
                if matched is not None:
                    candidates.append((*matched, row))
        candidates.sort(key=lambda item: str(item[2].get("benchmark_id", "")))
    else:
        candidates = [(path, record, None) for _, path, record in records]
    if len(candidates) < count:
        raise ValueError("Not enough sources for requested gold review count")
    tasks: list[str] = []
    for index, (path, record, question_row) in enumerate(candidates[:count], start=1):
        evidence_catalog = [
            {"evidenceId": span.evidence_id, **span.location()}
            for span in build_evidence_spans(record)
        ]
        task = {
            "schemaVersion": "cco-gold-review-task/0.1.0",
            "taskId": f"gold-review-{index:03d}",
            "sourceFile": path.name,
            "snapshotId": record.id,
            "contentHash": record.content_hash,
            "sourceContract": "cco-text/1",
            "legacyFactRole": "legacy_audit_only",
            "reviewStatus": "pending",
            "trainable": False,
            "question": question_row.get("question") if question_row else None,
            "reviewHints": ({
                "source": "unreviewed_benchmark_seed",
                "benchmarkId": question_row.get("benchmark_id"),
                "category": question_row.get("category"),
                "expectedCriteria": question_row.get("expected_criteria", []),
                "failureModes": question_row.get("failure_modes", []),
            } if question_row else None),
            "goldLabels": None,
            "evidenceCatalog": evidence_catalog,
        }
        tasks.append(json.dumps(task, ensure_ascii=False, separators=(",", ":")))
    tasks_path.write_text("\n".join(tasks) + "\n", encoding="utf-8")
    _write_json(manifest_path, {
        "schemaVersion": "cco-gold-review-manifest/0.1.0",
        "tasks": count,
        "containsSourceCcoText": False,
        "containsReviewQuestions": questions_path is not None,
        "containsSubjectIds": False,
        "reviewStatus": "pending",
        "trainable": False,
        "legacyFactRole": "legacy_audit_only",
        "goldLabelsCreated": False,
    })
    return GoldReviewPackage(
        package_root=output_directory,
        tasks=tasks_path,
        manifest=manifest_path,
    )
