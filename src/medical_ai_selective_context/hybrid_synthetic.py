"""Hybrid fictional-draft -> deterministic event-graph training-data pipeline.

The OpenAI boundary accepts only a root count, seed and model name. It cannot
accept CCO text or paths. Model output is a candidate design; Python owns split
assignment, event semantics, counterfactuals, rendering and source grounding.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import Field, model_validator

from .llm import strict_output_schema
from .synthetic_training import (
    SyntheticEvent,
    SyntheticModel,
    SyntheticScenario,
    _date,
    _write_json,
    build_synthetic_example,
    render_event_graph,
)
from .training_dataset import MANIFEST_VERSION, validate_training_dataset

DRAFT_SCHEMA_VERSION = "cco-fictional-scenario-drafts/0.2.0"
HYBRID_REPORT_VERSION = "cco-hybrid-pilot-report/0.2.0"
TAXONOMY_CONTRACT_VERSION = "cco-hybrid-taxonomy/1.0.0"
_NAMESPACE = uuid.UUID("d95f9f75-e89d-4fa8-9485-9e8bfab10619")
_VARIANTS = (
    "base",
    "relevant_counterfactual",
    "irrelevant_counterfactual",
    "order_paraphrase",
)

Template = Literal[
    "lab_lookup",
    "stale_wearable",
    "old_prescription",
    "diagnosis_status",
    "procedure_history",
    "persistent_history",
]
Variant = Literal[
    "base", "relevant_counterfactual", "irrelevant_counterfactual", "order_paraphrase"
]
EventType = Literal[
    "laboratory_result",
    "sleep_wearable_measurement",
    "activity_wearable_measurement",
    "cardiac_wearable_measurement",
    "prescription_record",
    "diagnosis_record",
    "procedure_record",
    "patient_reported_condition",
    "family_history",
    "allergy_report",
]
RecordSection = Literal[
    "Laboratuvar (hastane)",
    "Uyku (yüzük)",
    "Hareket",
    "Kalp (yüzük ve saat)",
    "İlaçlar (reçete kaydı)",
    "Tanılar (hastane kaydı)",
    "Ameliyat ve işlemler",
    "Hastalıklar (hasta beyanı)",
    "Aile öyküsü (hasta beyanı)",
    "Alerjiler (hasta beyanı)",
]
SubjectScope = Literal["patient", "family_member", "other"]


@dataclass(frozen=True)
class TaxonomyContract:
    event_type: EventType
    record_section: RecordSection
    subject_scope: SubjectScope
    graph_kind: str


_TAXONOMY_CONTRACTS: dict[Template, TaxonomyContract] = {
    "lab_lookup": TaxonomyContract(
        "laboratory_result", "Laboratuvar (hastane)", "patient", "laboratory"
    ),
    "stale_wearable": TaxonomyContract(
        "sleep_wearable_measurement", "Uyku (yüzük)", "patient", "wearable"
    ),
    "old_prescription": TaxonomyContract(
        "prescription_record", "İlaçlar (reçete kaydı)", "patient",
        "medication_prescription",
    ),
    "diagnosis_status": TaxonomyContract(
        "diagnosis_record", "Tanılar (hastane kaydı)", "patient", "diagnosis"
    ),
    "procedure_history": TaxonomyContract(
        "procedure_record", "Ameliyat ve işlemler", "patient", "procedure"
    ),
    "persistent_history": TaxonomyContract(
        "diagnosis_record", "Tanılar (hastane kaydı)", "patient", "diagnosis"
    ),
}

_SLEEP_TERMS = (
    "uyku", "gece", "uyuma", "uyanma", "uyanik", "uyaniklik", "rem", "derin uyku",
    "yatak", "kestirme",
)
_NON_SLEEP_WEARABLE_TERMS = (
    "adim", "hareket", "aktivite", "egzersiz", "vo2", "dinlenik nabiz", "hrv",
    "kalp hizi", "kalp atim",
)
_NON_PATIENT_DIAGNOSIS_TERMS = (
    "aile oyk", "anne", "baba", "kardes", "cocuk", "akraba", "alerji", "reaksiyon",
    "sigara", "tutun", "alkol",
)


def _fold(value: str) -> str:
    return value.casefold().translate(str.maketrans("çğıöşü", "cgiosu"))


def draft_taxonomy_errors(draft: FictionalScenarioDraft) -> list[str]:
    """Return deterministic contract violations without trusting model labels."""
    expected = _TAXONOMY_CONTRACTS[draft.template]
    errors = []
    if draft.event_type != expected.event_type:
        errors.append(f"eventType:{draft.event_type}!={expected.event_type}")
    if draft.record_section != expected.record_section:
        errors.append(f"recordSection:{draft.record_section}!={expected.record_section}")
    if draft.subject_scope != expected.subject_scope:
        errors.append(f"subjectScope:{draft.subject_scope}!={expected.subject_scope}")

    concept = _fold(f"{draft.clinical_domain} {draft.primary_label}")
    if draft.template == "stale_wearable":
        if not any(term in concept for term in _SLEEP_TERMS):
            errors.append("semantic:stale_wearable_requires_sleep_concept")
        if any(term in concept for term in _NON_SLEEP_WEARABLE_TERMS):
            errors.append("semantic:non_sleep_wearable_concept")
    if draft.template == "persistent_history" and any(
        term in concept for term in _NON_PATIENT_DIAGNOSIS_TERMS
    ):
        errors.append("semantic:persistent_history_requires_patient_diagnosis")
    return errors


class DraftTaxonomyError(ValueError):
    """Raised before compilation when a model draft violates the closed taxonomy."""

    def __init__(self, rejections: list[dict[str, Any]]):
        self.rejections = rejections
        super().__init__(f"draft_taxonomy_rejected:{len(rejections)}")


def reject_incompatible_drafts(batch: FictionalDraftBatch) -> FictionalDraftBatch:
    """Reject the whole candidate batch if any root has a taxonomy mismatch."""
    rejections = [
        {"draftId": draft.draft_id, "errors": errors}
        for draft in batch.drafts
        if (errors := draft_taxonomy_errors(draft))
    ]
    if rejections:
        raise DraftTaxonomyError(rejections)
    return batch


class FictionalScenarioDraft(SyntheticModel):
    """Unreviewed model suggestion. None of its clinical ideas are accepted as facts."""

    draft_id: str = Field(alias="draftId", pattern=r"^[a-z0-9][a-z0-9_-]{2,79}$")
    template: Template
    event_type: EventType = Field(alias="eventType")
    record_section: RecordSection = Field(alias="recordSection")
    subject_scope: SubjectScope = Field(alias="subjectScope")
    clinical_domain: str = Field(alias="clinicalDomain", min_length=3, max_length=80)
    age_band: Literal["18_29", "30_44", "45_64", "65_plus"] = Field(alias="ageBand")
    sex: Literal["kadın", "erkek", "bilinmiyor"]
    primary_label: str = Field(alias="primaryLabel", min_length=3, max_length=100)
    distractor_label: str = Field(alias="distractorLabel", min_length=3, max_length=100)
    question_variants: list[str] = Field(
        alias="questionVariants", min_length=3, max_length=3
    )
    clinical_variety_candidates: list[str] = Field(
        alias="clinicalVarietyCandidates", min_length=1, max_length=4
    )
    language_style: Literal["kısa", "gündelik", "ayrıntılı"] = Field(alias="languageStyle")

    @model_validator(mode="after")
    def coherent_draft(self) -> FictionalScenarioDraft:
        questions = [question.strip() for question in self.question_variants]
        if any(not question for question in questions) or len(set(questions)) != 3:
            raise ValueError("Draft question variants must be three distinct nonblank strings")
        if self.primary_label.casefold() == self.distractor_label.casefold():
            raise ValueError("Primary and distractor labels must differ")
        self.question_variants = questions
        return self


class FictionalDraftBatch(SyntheticModel):
    schema_version: Literal[DRAFT_SCHEMA_VERSION] = Field(
        alias="schemaVersion"
    )
    drafts: list[FictionalScenarioDraft] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def unique_drafts(self) -> FictionalDraftBatch:
        ids = [draft.draft_id for draft in self.drafts]
        if len(ids) != len(set(ids)):
            raise ValueError("Draft IDs must be unique")
        return self


class OpenAIDraftError(RuntimeError):
    """Stable provider error that does not echo credentials or response bodies."""


@dataclass(frozen=True)
class HybridSyntheticPackage:
    project_root: Path
    package_root: Path
    drafts: Path
    annotations: Path
    event_graphs: Path
    manifest: Path
    report: Path


_SYSTEM_PROMPT = """Design completely fictional Turkish medical-record scenario drafts.
Return only the requested JSON schema. Produce varied domains and natural Turkish question
phrasings. The suggestions are candidates for later review and are not medically validated.

Every draft must use exactly one supported taxonomy tuple:
- lab_lookup: laboratory_result | Laboratuvar (hastane) | patient
- stale_wearable: sleep_wearable_measurement | Uyku (yüzük) | patient
- old_prescription: prescription_record | İlaçlar (reçete kaydı) | patient
- diagnosis_status: diagnosis_record | Tanılar (hastane kaydı) | patient
- procedure_history: procedure_record | Ameliyat ve işlemler | patient
- persistent_history: diagnosis_record | Tanılar (hastane kaydı) | patient

stale_wearable is limited to sleep metrics. Do not use steps, exercise, activity, heart rate,
HRV or VO2max. persistent_history is limited to the patient's own persistent hospital diagnosis.
Do not use family history, allergy, medication reaction, tobacco/alcohol history or a
patient-reported condition for that template. Unsupported concepts must not be emitted."""


def build_openai_draft_request(*, model: str, root_count: int, seed: int) -> dict[str, Any]:
    """Build a strict Responses API request from synthetic controls only."""
    if not model.strip():
        raise ValueError("OpenAI model is required")
    if not 1 <= root_count <= 50:
        raise ValueError("Root count must be between 1 and 50")
    schema = strict_output_schema(
        FictionalDraftBatch.model_json_schema(by_alias=True)
    )
    user_prompt = (
        f"Create exactly {root_count} independent fictional roots. Use seed label {seed}. "
        "Distribute the six template enum values as evenly as possible. Each root must have "
        "three distinct Turkish question variants and 1-4 clinical diversity ideas. Copy the "
        "eventType, recordSection and subjectScope values from the supported tuple for the "
        "chosen template; never infer or substitute those fields."
    )
    return {
        "model": model.strip(),
        "input": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "fictional_scenario_drafts",
                "strict": True,
                "schema": schema,
            }
        },
        "max_output_tokens": 16000,
        "store": False,
    }


def build_openrouter_draft_request(
    *, model: str, root_count: int, seed: int
) -> dict[str, Any]:
    """Build the equivalent strict request for OpenRouter Chat Completions."""
    openai_body = build_openai_draft_request(
        model=model, root_count=root_count, seed=seed
    )
    text_format = openai_body["text"]["format"]
    return {
        "model": openai_body["model"],
        "messages": openai_body["input"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": text_format["name"],
                "strict": text_format["strict"],
                "schema": text_format["schema"],
            },
        },
        "max_tokens": openai_body["max_output_tokens"],
        "seed": seed,
        "stream": False,
        "provider": {"require_parameters": True},
    }


def _response_output_text(payload: dict[str, Any]) -> str:
    if payload.get("status") != "completed":
        raise OpenAIDraftError("openai_response_not_completed")
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for item in payload.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") == "refusal":
                raise OpenAIDraftError("openai_response_refused")
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise OpenAIDraftError("openai_response_missing_structured_output")


def request_openai_drafts(
    *,
    api_key: str,
    model: str,
    root_count: int,
    seed: int,
    base_url: str = "https://api.openai.com/v1",
    timeout_seconds: float = 120,
    transport: httpx.BaseTransport | None = None,
) -> tuple[FictionalDraftBatch, str]:
    """Request fictional candidates. No clinical source input is accepted by this API."""
    if not api_key.strip():
        raise OpenAIDraftError("openai_api_key_missing")
    body = build_openai_draft_request(model=model, root_count=root_count, seed=seed)
    request_hash = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    try:
        with httpx.Client(timeout=timeout_seconds, transport=transport) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/responses",
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
        batch = FictionalDraftBatch.model_validate_json(_response_output_text(payload))
    except OpenAIDraftError:
        raise
    except (httpx.HTTPError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise OpenAIDraftError("openai_draft_request_failed") from exc
    if len(batch.drafts) != root_count:
        raise OpenAIDraftError("openai_draft_count_mismatch")
    return batch, request_hash


def request_openrouter_drafts(
    *,
    api_key: str,
    model: str,
    root_count: int,
    seed: int,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout_seconds: float = 120,
    transport: httpx.BaseTransport | None = None,
) -> tuple[FictionalDraftBatch, str]:
    """Request the same fictional candidate schema through OpenRouter."""
    if not api_key.strip():
        raise OpenAIDraftError("openrouter_api_key_missing")
    body = build_openrouter_draft_request(
        model=model, root_count=root_count, seed=seed
    )
    request_hash = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    try:
        with httpx.Client(timeout=timeout_seconds, transport=transport) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
        choice = payload["choices"][0]
        message = choice["message"]
        if message.get("refusal"):
            raise OpenAIDraftError("openrouter_response_refused")
        if choice.get("finish_reason") not in {"stop", None}:
            raise OpenAIDraftError("openrouter_response_not_completed")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OpenAIDraftError("openrouter_response_missing_structured_output")
        batch = FictionalDraftBatch.model_validate_json(content)
    except OpenAIDraftError:
        raise
    except (httpx.HTTPError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
        raise OpenAIDraftError("openrouter_draft_request_failed") from exc
    if len(batch.drafts) != root_count:
        raise OpenAIDraftError("openrouter_draft_count_mismatch")
    return batch, request_hash


def _fixture_questions(template: Template, label: str) -> list[str]:
    mapping = {
        "lab_lookup": [
            f"Son {label} sonucum nedir?",
            f"Kayıttaki en yeni {label} değerini söyler misin?",
            f"{label} ölçümüm hangi tarihte ve kaç çıkmış?",
        ],
        "stale_wearable": [
            f"Dün gece {label} kaydım var mı?",
            f"En son {label} ölçümüm hangi geceye ait?",
            f"Son {label} verisini dün geceymiş gibi yorumlayabilir misin?",
        ],
        "old_prescription": [
            f"{label} için eski reçetem bugün kullanım anlamına gelir mi?",
            f"Kayıtta {label} şu anda aktif görünüyor mu?",
            f"Geçmiş {label} reçetesinden bugünkü kullanımı anlayabilir miyiz?",
        ],
        "diagnosis_status": [
            f"{label} kaydı kesin tanı mı?",
            f"Kayıtlarda {label} hangi tanı statüsünde?",
            f"{label} kesinleşmiş gibi görünüyor mu?",
        ],
        "procedure_history": [
            f"{label} işlemini geçirmiş miyim?",
            f"Kayıtta {label} tarihi nedir?",
            f"Geçmişimde {label} bulunuyor mu?",
        ],
        "persistent_history": [
            f"{label} geçmişim bugünkü değerlendirmede önemli mi?",
            f"Eski {label} kaydını bu soruda dikkate almak gerekir mi?",
            f"{label} öyküsü bugünkü yakınmanın kesin nedeni sayılır mı?",
        ],
    }
    return mapping[template]


def build_fixture_drafts(root_count: int = 20) -> FictionalDraftBatch:
    """Offline contract fixture used when no API credential is configured."""
    templates: list[Template] = [
        "lab_lookup", "stale_wearable", "old_prescription", "diagnosis_status",
        "procedure_history", "persistent_history",
    ]
    domains = [
        "metabolik izlem", "uyku süresi izlemi", "ağrı ilacı geçmişi",
        "nörolojik tanı kaydı", "karın cerrahisi geçmişi", "kas iskelet öyküsü",
        "kan hücresi izlemi", "uyku verimliliği izlemi", "antimikrobiyal reçete geçmişi",
        "denge bozukluğu kaydı", "diz işlemi geçmişi", "sindirim sistemi öyküsü",
        "elektrolit izlemi", "gece uyanma izlemi", "alerji ilacı geçmişi",
        "burun bulgusu kaydı", "göz işlemi geçmişi", "solunum yolu öyküsü",
        "enzim izlemi", "dinlenme süresi izlemi",
    ]
    labels = [
        "kurgusal glisemik belirteç", "ana uyku süresi", "örnek ağrı ilacı",
        "kurgusal baş ağrısı sendromu", "örnek karın işlemi", "kalıcı eklem öyküsü",
        "örnek kan hücresi belirteci", "uyku verimliliği", "örnek antimikrobiyal ilaç",
        "geçici denge bozukluğu", "örnek diz işlemi", "kalıcı mide yakınması öyküsü",
        "örnek elektrolit belirteci", "gece uyanık kalma süresi", "örnek alerji ilacı",
        "tekrarlayan burun bulgusu", "örnek göz işlemi", "kalıcı solunum öyküsü",
        "örnek karaciğer enzimi", "dinlendirici uyku süresi",
    ]
    drafts = []
    for index in range(root_count):
        template = templates[index % len(templates)]
        label = labels[index % len(labels)]
        taxonomy = _TAXONOMY_CONTRACTS[template]
        drafts.append(FictionalScenarioDraft(
            draftId=f"fixture-{index + 1:03d}",
            template=template,
            eventType=taxonomy.event_type,
            recordSection=taxonomy.record_section,
            subjectScope=taxonomy.subject_scope,
            clinicalDomain=domains[index % len(domains)],
            ageBand=("18_29", "30_44", "45_64", "65_plus")[index % 4],
            sex=("kadın", "erkek", "bilinmiyor")[index % 3],
            primaryLabel=label,
            distractorLabel=f"ilgisiz eski kayıt {index + 1}",
            questionVariants=_fixture_questions(template, label),
            clinicalVarietyCandidates=[
                "zaman yakınlığını değiştir",
                "kalıcılık sinyalini ayrıca incele",
            ],
            languageStyle=("kısa", "gündelik", "ayrıntılı")[index % 3],
        ))
    return FictionalDraftBatch(schemaVersion=DRAFT_SCHEMA_VERSION, drafts=drafts)


def _age(age_band: str, index: int) -> int:
    starts = {"18_29": 18, "30_44": 30, "45_64": 45, "65_plus": 65}
    widths = {"18_29": 12, "30_44": 15, "45_64": 20, "65_plus": 25}
    return starts[age_band] + index % widths[age_band]


def _uuid(root_id: str, variant: str, kind: str, seed: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{seed}:{root_id}:{variant}:{kind}"))


def _split_for_root(index: int) -> Literal["train", "dev", "test"]:
    bucket = index % 10
    return "dev" if bucket == 8 else "test" if bucket == 9 else "train"


def _answer_for(template: Template, event: SyntheticEvent, as_of: datetime) -> str:
    event_date = _date(event.occurred_on)
    if template == "lab_lookup":
        return f"Kayıttaki son {event.display} sonucu {event.value} {event.unit}; tarih {event_date}."
    if template == "stale_wearable":
        if event.occurred_on == as_of.date() - timedelta(days=1):
            return f"Dün geceki {event.display} kaydı {event.value} {event.unit}."
        return (
            f"Dün geceye ait kayıt yok. Son {event.display} kaydı {event_date} tarihindeki "
            f"{event.value} {event.unit} ölçümüdür."
        )
    if template == "old_prescription":
        return (
            f"Kayıt yalnız {event_date} tarihli {event.display} reçetesini gösteriyor; "
            "bu, ilacın bugün aktif kullanıldığını kanıtlamaz."
        )
    if template == "diagnosis_status":
        status = "kesin tanı" if event.assertion == "definitive" else "ön tanı"
        return f"{event.display}, {event_date} tarihli kayıtta {status} olarak geçiyor."
    if template == "procedure_history":
        return f"Kayıtta {event_date} tarihli {event.display} bulunuyor."
    if event.course == "resolved":
        return (
            f"{event.display} kaydında durumun düzeldiği açıkça yazıyor; bu nedenle bugünkü "
            "yakınmanın nedeni olduğu varsayılamaz."
        )
    return (
        f"{event.display} geçmişi değerlendirmede önemli olabilir; kayıt tek başına bugünkü "
        "yakınmanın nedenini kanıtlamaz."
    )


def _event_roles(template: Template) -> tuple[list[str], list[str], list[str]]:
    return {
        "lab_lookup": (["lookup"], ["laboratory"], []),
        "stale_wearable": (["lookup", "interpretation"], ["wearable"], []),
        "old_prescription": (
            ["guidance", "clarification"], ["medication"],
            ["Bu ilacı şu anda kullanıyor musun?"],
        ),
        "diagnosis_status": (["lookup", "interpretation"], ["diagnosis"], []),
        "procedure_history": (["lookup"], ["procedure"], []),
        "persistent_history": (
            ["interpretation", "clarification"], ["symptom", "diagnosis"],
            ["Bu yakınma ne zaman başladı?"],
        ),
    }[template]


def _scenario_for_variant(
    draft: FictionalScenarioDraft,
    *,
    root_id: str,
    index: int,
    seed: int,
    variant: Variant,
) -> SyntheticScenario:
    scenario_id = f"{root_id}-{variant.replace('_', '-')}"
    as_of = datetime.fromisoformat("2026-10-21T12:00:00+03:00")
    recent = as_of.date() - timedelta(days=2 + index % 35)
    old = as_of.date() - timedelta(days=800 + index * 23)
    event_id = f"{scenario_id}-primary"
    if draft.template == "lab_lookup":
        primary = SyntheticEvent(
            eventId=event_id, kind="laboratory", display=draft.primary_label,
            occurredOn=recent, value="5,7", unit="birim", requiredness="must_use",
            useRole="direct_fact", reasonCode="latest_recorded_measurement",
        )
    elif draft.template == "stale_wearable":
        primary = SyntheticEvent(
            eventId=event_id, kind="wearable", display=draft.primary_label,
            occurredOn=old, value="6 sa 12 dk", unit="", requiredness="must_use",
            useRole="status_check", reasonCode="wearable_date_must_be_stated",
        )
    elif draft.template == "old_prescription":
        primary = SyntheticEvent(
            eventId=event_id, kind="medication_prescription", display=draft.primary_label,
            occurredOn=old, requiredness="must_use", useRole="status_check",
            safetyCritical=True, reasonCode="prescription_does_not_prove_active_use",
        )
    elif draft.template == "diagnosis_status":
        primary = SyntheticEvent(
            eventId=event_id, kind="diagnosis", display=draft.primary_label,
            occurredOn=recent, assertion="preliminary", diagnosisGroup="other",
            requiredness="must_use", useRole="direct_fact",
            reasonCode="explicit_diagnosis_assertion",
        )
    elif draft.template == "procedure_history":
        primary = SyntheticEvent(
            eventId=event_id, kind="procedure", display=draft.primary_label,
            occurredOn=old, requiredness="must_use", useRole="direct_fact",
            reasonCode="recorded_procedure_history",
        )
    else:
        primary = SyntheticEvent(
            eventId=event_id, kind="diagnosis", display=draft.primary_label,
            occurredOn=old, assertion="definitive", diagnosisGroup="chronic",
            requiredness="must_use", useRole="background", causalStrength="possible",
            reasonCode="persistent_history_may_be_relevant",
        )

    distractor = SyntheticEvent(
        eventId=f"{scenario_id}-distractor", kind="diagnosis",
        display=draft.distractor_label, occurredOn=old - timedelta(days=370),
        assertion="definitive", diagnosisGroup="acute", requiredness="not_needed",
        useRole="none", reasonCode="unrelated_old_record",
    )

    if variant == "relevant_counterfactual":
        primary.event_id = f"{scenario_id}-primary"
        if draft.template == "lab_lookup":
            primary.value = "6,4"
            primary.occurred_on = recent + timedelta(days=1)
        elif draft.template == "stale_wearable":
            primary.value = "7 sa 03 dk"
            primary.occurred_on = as_of.date() - timedelta(days=1)
        elif draft.template == "old_prescription":
            primary.occurred_on = recent
        elif draft.template == "diagnosis_status":
            primary.assertion = "definitive"
        elif draft.template == "procedure_history":
            primary.occurred_on = old + timedelta(days=45)
        else:
            primary.course = "resolved"
            primary.causal_strength = "none"
            primary.reason_code = "explicitly_resolved_history"
    if variant == "irrelevant_counterfactual":
        distractor.display = f"{draft.distractor_label} alternatif"
        distractor.occurred_on = old - timedelta(days=740)

    question_index = 1 if variant == "order_paraphrase" else 0
    events = [primary, distractor]
    if variant == "order_paraphrase":
        events.reverse()
    intents, subjects, followups = _event_roles(draft.template)
    return SyntheticScenario(
        scenarioId=scenario_id,
        snapshotId=_uuid(root_id, variant, "snapshot", seed),
        subjectId=_uuid(root_id, variant, "subject", seed),
        asOf=as_of,
        age=_age(draft.age_band, index),
        sex=draft.sex,
        question=draft.question_variants[question_index],
        answer=_answer_for(draft.template, primary, as_of),
        intents=intents,
        subjects=subjects,
        personalizationDecision="required",
        events=events,
        followupQuestions=followups,
        causalClaimAllowed=False,
        uncertaintyRequired=draft.template in {"old_prescription", "persistent_history"},
    )


def generate_hybrid_pilot(
    project_root: Path,
    batch: FictionalDraftBatch,
    *,
    seed: int = 20261021,
    package_name: str = "hybrid-pilot",
    draft_provider: Literal["openai", "openrouter", "fixture"] = "fixture",
    external_api_calls: int = 0,
    request_sha256: str | None = None,
) -> HybridSyntheticPackage:
    """Compile candidate drafts into four unreviewed, non-trainable variants each."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,39}", package_name):
        raise ValueError("Synthetic package name is invalid")
    reject_incompatible_drafts(batch)
    project_root = Path(project_root).resolve()
    package_root = project_root / "data" / "training_v2" / package_name
    sources = package_root / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    drafts_path = package_root / "drafts.json"
    annotations_path = package_root / "annotations.jsonl"
    graphs_path = package_root / "event-graphs.jsonl"
    manifest_path = package_root / "manifest.json"
    report_path = package_root / "hybrid-report.json"
    _write_json(drafts_path, batch.model_dump(by_alias=True, mode="json"))

    annotations: list[str] = []
    graphs: list[str] = []
    root_splits: dict[str, set[str]] = defaultdict(set)
    root_example_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    taxonomy_render_mismatches: list[dict[str, str]] = []
    for index, draft in enumerate(batch.drafts):
        root_id = f"hybrid-root-{index + 1:04d}"
        split = _split_for_root(index)
        base_example_id = f"{root_id}-base"
        for variant_name in _VARIANTS:
            variant: Variant = variant_name  # type: ignore[assignment]
            scenario = _scenario_for_variant(
                draft, root_id=root_id, index=index, seed=seed, variant=variant
            )
            rendered = render_event_graph(scenario)
            primary_event_id = f"{scenario.scenario_id}-primary"
            primary_event = next(
                event for event in scenario.events if event.event_id == primary_event_id
            )
            primary_span = rendered.event_evidence[primary_event_id]
            contract = _TAXONOMY_CONTRACTS[draft.template]
            if (
                primary_event.kind != contract.graph_kind
                or primary_span.section_heading != contract.record_section
            ):
                taxonomy_render_mismatches.append({
                    "draftId": draft.draft_id,
                    "variant": variant,
                    "eventType": draft.event_type,
                    "expectedGraphKind": contract.graph_kind,
                    "actualGraphKind": primary_event.kind,
                    "expectedSection": contract.record_section,
                    "actualSection": primary_span.section_heading,
                })
            relative_source = (
                f"training_v2/{package_name}/sources/{scenario.snapshot_id}.json"
            )
            _write_json(sources / f"{scenario.snapshot_id}.json", rendered.cco)
            annotation = build_synthetic_example(
                scenario,
                rendered,
                split=split,
                cco_file=relative_source,
                root_id=root_id,
                variant=variant,
                generator="hybrid-openai-draft-python-compiler",
                parent_example_id=None if variant == "base" else base_example_id,
                trainable=False,
            )
            annotations.append(json.dumps(annotation, ensure_ascii=False, separators=(",", ":")))
            graphs.append(json.dumps({
                "rootId": root_id,
                "draftId": draft.draft_id,
                "split": split,
                "variant": variant,
                "taxonomy": {
                    "contractVersion": TAXONOMY_CONTRACT_VERSION,
                    "eventType": draft.event_type,
                    "recordSection": draft.record_section,
                    "subjectScope": draft.subject_scope,
                },
                "scenario": scenario.model_dump(by_alias=True, mode="json"),
            }, ensure_ascii=False, separators=(",", ":")))
            root_splits[root_id].add(split)
            root_example_counts[root_id] += 1
            variant_counts[variant] += 1

    if taxonomy_render_mismatches:
        raise DraftTaxonomyError(taxonomy_render_mismatches)

    annotations_path.write_text("\n".join(annotations) + "\n", encoding="utf-8")
    graphs_path.write_text("\n".join(graphs) + "\n", encoding="utf-8")
    _write_json(manifest_path, {
        "schemaVersion": MANIFEST_VERSION,
        "datasetId": f"hybrid-{len(batch.drafts)}-roots-{seed}",
        "purpose": "answer_sft",
        "policy": {
            "personalizationPriority": True,
            "primarySource": "text_cco",
            "legacyFactRole": "legacy_audit_only",
            "maxNotSupportedFraction": 0.1,
            "minReviewedFraction": 0.0,
        },
    })
    validation = validate_training_dataset(annotations_path, manifest_path, project_root)
    roots_with_four = sum(count == 4 for count in root_example_counts.values())
    root_split_counts = Counter(next(iter(splits)) for splits in root_splits.values())
    report = {
        "schemaVersion": HYBRID_REPORT_VERSION,
        "valid": validation["valid"],
        "pilot": {
            "roots": len(batch.drafts),
            "examples": len(annotations),
            "variantsPerRoot": 4,
            "variantCounts": dict(sorted(variant_counts.items())),
            "rootSplitCounts": dict(sorted(root_split_counts.items())),
            "exampleSplitCounts": validation["splits"],
        },
        "diversity": {
            "templateCounts": dict(sorted(Counter(draft.template for draft in batch.drafts).items())),
            "clinicalDomainCount": len({draft.clinical_domain for draft in batch.drafts}),
            "uniquePrimaryLabels": len({draft.primary_label.casefold() for draft in batch.drafts}),
            "uniqueQuestionCandidates": len({
                question.casefold()
                for draft in batch.drafts for question in draft.question_variants
            }),
            "clinicalVarietyCandidates": sum(
                len(draft.clinical_variety_candidates) for draft in batch.drafts
            ),
        },
        "correctness": {
            "jsonSchemaValidatedDrafts": len(batch.drafts),
            "eventGraphsValidated": len(graphs),
            "sourceGroundingValidated": validation["valid"],
            "taxonomyContractVersion": TAXONOMY_CONTRACT_VERSION,
            "taxonomyValidatedDrafts": len(batch.drafts),
            "taxonomyRejectedDrafts": 0,
            "taxonomyRenderMismatches": 0,
            "taxonomyErrors": 0,
            "modelClinicalRelationshipsAcceptedAsTrue": False,
            "clinicalCorrectnessEvaluated": False,
        },
        "sourceIntegrity": {
            "rootsWithExactlyFourVariants": roots_with_four,
            "rootSplitLeakage": sum(len(splits) > 1 for splits in root_splits.values()),
            "stableEvidenceReferencesValidated": validation["valid"],
            "legacyFactIdsPresent": validation["legacyFactIdsPresent"],
        },
        "dataLeakage": {
            "realCcoAcceptedByApiBoundary": False,
            "realCcoSentToExternalApi": False,
            "externalRequestInputClasses": ["fictional_generation_brief", "json_schema"],
            "draftProvider": draft_provider,
            "externalApiCalls": external_api_calls,
            "requestSha256": request_sha256,
        },
        "review": {
            "reviewStatus": "unreviewed",
            "unreviewedExamples": len(annotations),
            "trainableExamples": 0,
            "trainable": False,
        },
        "loraStarted": False,
    }
    _write_json(report_path, report)
    return HybridSyntheticPackage(
        project_root=project_root,
        package_root=package_root,
        drafts=drafts_path,
        annotations=annotations_path,
        event_graphs=graphs_path,
        manifest=manifest_path,
        report=report_path,
    )
