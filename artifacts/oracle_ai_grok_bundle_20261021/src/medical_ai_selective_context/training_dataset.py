"""Contracts and deterministic validation for the CCO V2 training dataset.

This module deliberately validates only things that can be established from the
annotation and its immutable CCO source.  It does not pretend to validate the
clinical quality of an answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .text_document import (
    evidence_span_index,
    extract_explicit_claims,
    parse_cco_text,
)

SCHEMA_VERSION = "cco-text-training-annotation/0.3.0"
MANIFEST_VERSION = "cco-text-training-manifest/0.3.0"
_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,127}$"
_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_EVIDENCE_ID_PATTERN = r"^evi_[0-9a-f]{32}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class QuestionSpec(StrictModel):
    text: str = Field(min_length=1, max_length=16000)
    intents: list[Literal[
        "lookup", "interpretation", "guidance", "triage", "clarification", "follow_up",
    ]] = Field(min_length=1, max_length=4)
    subjects: list[Literal[
        "symptom", "wearable", "laboratory", "medication", "daily_activity", "diagnosis",
        "procedure", "general_health",
    ]] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique_labels(self) -> QuestionSpec:
        if not self.text.strip():
            raise ValueError("Question text must not be blank")
        if len(set(self.intents)) != len(self.intents):
            raise ValueError("Question intents must be unique")
        if len(set(self.subjects)) != len(self.subjects):
            raise ValueError("Question subjects must be unique")
        return self


class EvidenceSourceSpan(StrictModel):
    snapshot_id: str = Field(alias="snapshotId", pattern=r"^[0-9a-fA-F-]{36}$")
    content_hash: str = Field(alias="contentHash", pattern=_SHA256_PATTERN)
    section_heading: str = Field(alias="sectionHeading", min_length=1, max_length=200)
    start_line: int = Field(alias="startLine", ge=1)
    end_line: int = Field(alias="endLine", ge=1)
    text_sha256: str = Field(alias="textSha256", pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def valid_line_range(self) -> EvidenceSourceSpan:
        if self.end_line < self.start_line:
            raise ValueError("Evidence source span ends before it starts")
        return self


class EvidenceAnnotation(StrictModel):
    evidence_id: str = Field(alias="evidenceId", pattern=_EVIDENCE_ID_PATTERN)
    source_span: EvidenceSourceSpan = Field(alias="sourceSpan")
    fact_id: str | None = Field(default=None, alias="factId", pattern=r"^[0-9a-fA-F-]{36}$")
    requiredness: Literal["must_use", "helpful", "not_needed", "harmful"]
    use_role: Literal[
        "direct_fact", "background", "safety", "counterevidence", "status_check", "none",
    ] = Field(alias="useRole")
    causal_strength: Literal["none", "possible", "supported"] = Field(alias="causalStrength")
    safety_critical: bool = Field(alias="safetyCritical")
    reason_code: str = Field(alias="reasonCode", min_length=3, max_length=80)
    explicit_claims: list[Literal[
        "patient_reported", "medication_prescribed", "medication_active",
        "medication_stopped", "assertion_definitive", "assertion_preliminary",
        "course_improved", "course_resolved", "causality_explicit",
    ]] = Field(default_factory=list, alias="explicitClaims")
    independent_clinical_validation: Literal[False] = Field(
        default=False, alias="independentClinicalValidation"
    )

    @property
    def ref(self) -> str:
        """Internal compatibility alias; external schemas use evidenceId."""
        return self.evidence_id

    @model_validator(mode="after")
    def coherent_semantics(self) -> EvidenceAnnotation:
        if not _REASON_PATTERN.fullmatch(self.reason_code):
            raise ValueError("reasonCode must be a lowercase snake_case label")
        if len(self.explicit_claims) != len(set(self.explicit_claims)):
            raise ValueError("explicitClaims must be unique")
        if self.requiredness in {"not_needed", "harmful"}:
            if self.use_role != "none" or self.causal_strength != "none":
                raise ValueError("Excluded evidence cannot be assigned a use or causal role")
            if self.safety_critical:
                raise ValueError("Excluded evidence cannot be safety critical")
        if self.use_role == "none" and self.requiredness in {"must_use", "helpful"}:
            raise ValueError("Positive evidence must have a use role")
        if self.use_role == "status_check" and self.causal_strength != "none":
            raise ValueError("Status checks cannot establish causality")
        if self.safety_critical and self.requiredness != "must_use":
            raise ValueError("Safety-critical evidence must be must_use")
        return self


class FollowupPlan(StrictModel):
    needed: bool
    questions: list[str] = Field(max_length=2)

    @model_validator(mode="after")
    def questions_match_decision(self) -> FollowupPlan:
        cleaned = [question.strip() for question in self.questions]
        if any(not question for question in cleaned):
            raise ValueError("Follow-up questions must not be blank")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Follow-up questions must be unique")
        if self.needed != bool(cleaned):
            raise ValueError("needed must agree with the presence of follow-up questions")
        self.questions = cleaned
        return self


class AnswerPolicy(StrictModel):
    personalized_answer_expected: bool = Field(alias="personalizedAnswerExpected")
    causal_claim_allowed: bool = Field(alias="causalClaimAllowed")
    uncertainty_disclosure_required: bool = Field(alias="uncertaintyDisclosureRequired")
    direct_answer_first: bool = Field(default=True, alias="directAnswerFirst")
    unspecified_state_policy: Literal["remain_unspecified"] = Field(
        default="remain_unspecified", alias="unspecifiedStatePolicy"
    )
    source_text_clinically_validated: Literal[False] = Field(
        default=False, alias="sourceTextClinicallyValidated"
    )
    legacy_fact_can_define_gold: Literal[False] = Field(
        default=False, alias="legacyFactCanDefineGold"
    )


class Provenance(StrictModel):
    source_type: Literal["handcrafted", "synthetic", "deidentified_real"] = Field(
        alias="sourceType"
    )
    review_status: Literal["unreviewed", "reviewed"] = Field(alias="reviewStatus")
    generator: str = Field(min_length=1, max_length=120)
    parent_example_id: str | None = Field(default=None, alias="parentExampleId")
    parser_version: str = Field(alias="parserVersion", min_length=1, max_length=80)


class Corruption(StrictModel):
    kind: Literal[
        "wrong_date", "unsupported_causality", "medication_state", "assertion_status",
        "experiencer", "critical_omission", "numeric_value", "unit", "reference_range",
        "sync_time",
    ]
    source_example_id: str = Field(alias="sourceExampleId", pattern=_ID_PATTERN)


class TrainingExample(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = Field(alias="schemaVersion")
    example_id: str = Field(alias="exampleId", pattern=_ID_PATTERN)
    root_id: str = Field(alias="rootId", pattern=_ID_PATTERN)
    cco_file: str = Field(alias="ccoFile", min_length=1, max_length=1024)
    snapshot_id: str = Field(alias="snapshotId", pattern=r"^[0-9a-fA-F-]{36}$")
    content_hash: str = Field(alias="contentHash", pattern=_SHA256_PATTERN)
    split: Literal["train", "dev", "test"]
    variant: Literal[
        "base", "relevant_counterfactual", "irrelevant_counterfactual", "order_paraphrase",
        "controlled_corruption", "human_gold",
    ]
    question: QuestionSpec
    personalization_decision: Literal["required", "helpful", "not_supported"] = Field(
        alias="personalizationDecision"
    )
    evidence_annotations: list[EvidenceAnnotation] = Field(
        default_factory=list, alias="evidenceAnnotations", max_length=500
    )
    alternative_evidence_sets: list[list[str]] = Field(
        default_factory=list, alias="alternativeEvidenceSets", max_length=20
    )
    critical_evidence_bundles: list[list[str]] = Field(
        default_factory=list, alias="criticalEvidenceBundles", max_length=20
    )
    followup_plan: FollowupPlan = Field(alias="followupPlan")
    answer_policy: AnswerPolicy = Field(alias="answerPolicy")
    trainable: bool
    candidate_answer: str | None = Field(
        default=None, alias="candidateAnswer", max_length=8000
    )
    gold_answer: str | None = Field(default=None, alias="goldAnswer", max_length=8000)
    provenance: Provenance
    corruption: Corruption | None = None

    @model_validator(mode="after")
    def coherent_target(self) -> TrainingExample:
        refs = [item.ref for item in self.evidence_annotations]
        if len(refs) != len(set(refs)):
            raise ValueError("Evidence annotations must have unique refs")
        by_ref = {item.ref: item for item in self.evidence_annotations}
        positive = [
            item for item in self.evidence_annotations
            if item.requiredness in {"must_use", "helpful"}
        ]
        must_use = [item for item in positive if item.requiredness == "must_use"]

        if self.personalization_decision == "required" and not must_use:
            raise ValueError("required personalization needs at least one must_use reference")
        if self.personalization_decision == "helpful" and (not positive or must_use):
            raise ValueError("helpful personalization needs helpful but no must_use evidence")
        if self.personalization_decision == "not_supported" and positive:
            raise ValueError("not_supported personalization cannot contain positive evidence")
        expected_personalization = self.personalization_decision != "not_supported"
        if self.answer_policy.personalized_answer_expected != expected_personalization:
            raise ValueError("Answer policy conflicts with personalizationDecision")

        for group_name, groups in (
            ("alternativeEvidenceSets", self.alternative_evidence_sets),
            ("criticalEvidenceBundles", self.critical_evidence_bundles),
        ):
            canonical = []
            for group in groups:
                if not group or len(group) != len(set(group)):
                    raise ValueError(f"{group_name} entries must be nonempty unique ref lists")
                if any(ref not in by_ref for ref in group):
                    raise ValueError(f"{group_name} contains an unannotated ref")
                canonical.append(tuple(sorted(group)))
            if len(canonical) != len(set(canonical)):
                raise ValueError(f"{group_name} contains duplicate groups")

        critical_refs = {ref for group in self.critical_evidence_bundles for ref in group}
        if any(not by_ref[ref].safety_critical for ref in critical_refs):
            raise ValueError("Critical bundles may contain only safety-critical evidence")
        if (any(item.causal_strength == "supported" for item in positive)
                and not self.answer_policy.causal_claim_allowed):
            raise ValueError("Supported causal evidence conflicts with answer policy")
        if self.variant == "controlled_corruption" and self.corruption is None:
            raise ValueError("Controlled corruption examples require corruption metadata")
        if self.variant != "controlled_corruption" and self.corruption is not None:
            raise ValueError("Only controlled corruption examples may carry corruption metadata")
        if self.provenance.review_status == "unreviewed":
            if self.trainable:
                raise ValueError("Unreviewed examples cannot be trainable")
            if self.gold_answer is not None:
                raise ValueError("Unreviewed examples cannot carry a gold answer")
        if self.trainable and (
            self.provenance.review_status != "reviewed" or self.gold_answer is None
        ):
            raise ValueError("Trainable examples require review and a gold answer")
        return self

    def ranker_target(self) -> dict[str, Any]:
        """Return only fields the ranker is expected to generate."""
        return {
            "personalizationDecision": self.personalization_decision,
            "evidence": [
                annotation.model_dump(by_alias=True) for annotation in self.evidence_annotations
            ],
            "alternativeEvidenceSets": self.alternative_evidence_sets,
            "criticalEvidenceBundles": self.critical_evidence_bundles,
            "followupPlan": self.followup_plan.model_dump(by_alias=True),
        }


class RankerPrediction(StrictModel):
    """The intentionally small output contract expected from the ranker."""

    personalization_decision: Literal["required", "helpful", "not_supported"] = Field(
        alias="personalizationDecision"
    )
    evidence: list[EvidenceAnnotation] = Field(default_factory=list, max_length=500)
    alternative_evidence_sets: list[list[str]] = Field(
        default_factory=list, alias="alternativeEvidenceSets", max_length=20
    )
    critical_evidence_bundles: list[list[str]] = Field(
        default_factory=list, alias="criticalEvidenceBundles", max_length=20
    )
    followup_plan: FollowupPlan = Field(alias="followupPlan")

    @model_validator(mode="after")
    def coherent_prediction(self) -> RankerPrediction:
        refs = [item.ref for item in self.evidence]
        if len(refs) != len(set(refs)):
            raise ValueError("Ranker evidence must have unique refs")
        selected = set(refs)
        for groups in (self.alternative_evidence_sets, self.critical_evidence_bundles):
            canonical: list[tuple[str, ...]] = []
            for group in groups:
                if not group or len(group) != len(set(group)):
                    raise ValueError("Ranker evidence groups must be nonempty and unique")
                if any(ref not in selected for ref in group):
                    raise ValueError("Ranker evidence groups must reference predicted evidence")
                canonical.append(tuple(sorted(group)))
            if len(canonical) != len(set(canonical)):
                raise ValueError("Ranker evidence groups must be unique")
        return self


def _positive_refs(items: list[EvidenceAnnotation]) -> set[str]:
    return {
        item.ref for item in items if item.requiredness in {"must_use", "helpful"}
    }


def _required_evidence_units(gold: TrainingExample) -> list[list[set[str]]]:
    """Build recall units; alternative sets count as one interchangeable unit."""
    must_use = {
        item.ref
        for item in gold.evidence_annotations
        if item.requiredness == "must_use"
    }
    alternatives = [set(group) for group in gold.alternative_evidence_sets]
    covered_by_alternatives = set().union(*alternatives) if alternatives else set()
    units: list[list[set[str]]] = [
        [{ref}] for ref in sorted(must_use - covered_by_alternatives)
    ]
    if alternatives:
        units.append(alternatives)
    return units


def score_ranker_prediction(
    gold: TrainingExample,
    prediction: RankerPrediction | dict[str, Any],
    available_refs: set[str],
) -> dict[str, Any]:
    """Score source-grounded ranker behavior without judging clinical prose."""
    predicted = (
        prediction
        if isinstance(prediction, RankerPrediction)
        else RankerPrediction.model_validate(prediction)
    )
    gold_by_ref = {item.ref: item for item in gold.evidence_annotations}
    predicted_by_ref = {item.ref: item for item in predicted.evidence}
    selected = _positive_refs(predicted.evidence)
    invalid_refs = sorted(selected - available_refs)
    valid_selected = selected & available_refs
    adjudicated_selected = valid_selected & set(gold_by_ref)
    gold_positive = _positive_refs(gold.evidence_annotations)
    true_positive = adjudicated_selected & gold_positive
    false_positive = sorted(adjudicated_selected - gold_positive)
    unadjudicated = sorted(valid_selected - set(gold_by_ref))
    precision_denominator = len(true_positive) + len(false_positive)
    precision = len(true_positive) / precision_denominator if precision_denominator else 1.0

    required_units = _required_evidence_units(gold)
    satisfied_units = sum(
        any(option.issubset(valid_selected) for option in alternatives)
        for alternatives in required_units
    )
    required_recall = satisfied_units / len(required_units) if required_units else 1.0
    alternative_satisfied = not gold.alternative_evidence_sets or any(
        set(group).issubset(valid_selected) for group in gold.alternative_evidence_sets
    )

    critical_refs = {
        ref for bundle in gold.critical_evidence_bundles for ref in bundle
    }
    critical_recall = (
        len(critical_refs & valid_selected) / len(critical_refs)
        if critical_refs
        else 1.0
    )
    status_role_misses = sum(
        1
        for ref, annotation in gold_by_ref.items()
        if annotation.requiredness in {"must_use", "helpful"}
        and annotation.use_role == "status_check"
        and (
            ref not in predicted_by_ref
            or predicted_by_ref[ref].requiredness not in {"must_use", "helpful"}
            or predicted_by_ref[ref].use_role != "status_check"
        )
    )

    causal_rank = {"none": 0, "possible": 1, "supported": 2}
    causal_overreach = 0
    for ref in valid_selected:
        predicted_item = predicted_by_ref[ref]
        gold_item = gold_by_ref.get(ref)
        allowed = causal_rank[gold_item.causal_strength] if gold_item is not None else 0
        if causal_rank[predicted_item.causal_strength] > allowed:
            causal_overreach += 1

    return {
        "personalizationDecisionMatch": (
            predicted.personalization_decision == gold.personalization_decision
        ),
        "requiredEvidenceUnits": len(required_units),
        "requiredEvidenceRecall": round(required_recall, 6),
        "alternativeEvidenceSatisfied": alternative_satisfied,
        "adjudicatedSelectionPrecision": round(precision, 6),
        "criticalEvidenceRecall": round(critical_recall, 6),
        "causalOverreachCount": causal_overreach,
        "statusRoleMisses": status_role_misses,
        "falsePositiveRefs": false_positive,
        "unadjudicatedRefs": unadjudicated,
        "invalidRefs": invalid_refs,
    }


class DatasetPolicy(StrictModel):
    personalization_priority: Literal[True] = Field(alias="personalizationPriority")
    primary_source: Literal["text_cco"] = Field(default="text_cco", alias="primarySource")
    legacy_fact_role: Literal["legacy_audit_only"] = Field(
        default="legacy_audit_only", alias="legacyFactRole"
    )
    max_not_supported_fraction: float = Field(
        default=0.10, alias="maxNotSupportedFraction", ge=0, le=1
    )
    min_reviewed_fraction: float = Field(default=0, alias="minReviewedFraction", ge=0, le=1)


class DatasetManifest(StrictModel):
    schema_version: Literal[MANIFEST_VERSION] = Field(alias="schemaVersion")
    dataset_id: str = Field(alias="datasetId", pattern=_ID_PATTERN)
    purpose: Literal["schema_smoke_test", "gold_pilot", "ranker_sft", "answer_sft", "dpo"]
    policy: DatasetPolicy


class DatasetValidationError(ValueError):
    """A dataset error that intentionally omits clinical text from its message."""


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON property")
        result[key] = value
    return result


def _safe_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_pairs)


def _resolve_cco(name: str, project_root: Path) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
        raise DatasetValidationError("CCO path must be relative to the project data directory")
    data_root = (project_root / "data").resolve()
    candidate = (data_root / relative).resolve()
    if not candidate.is_relative_to(data_root) or not candidate.is_file():
        raise DatasetValidationError("CCO source is missing or outside the data directory")
    return candidate


def read_training_examples(path: Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line, object_pairs_hook=_unique_pairs)
                example = TrainingExample.model_validate(raw)
                if example.example_id in seen:
                    raise ValueError("Duplicate example ID")
            except (ValueError, TypeError) as exc:
                raise DatasetValidationError(
                    f"Invalid training annotation on line {line_number}"
                ) from exc
            seen.add(example.example_id)
            examples.append(example)
    if not examples:
        raise DatasetValidationError("Training dataset is empty")
    return examples


def validate_training_dataset(
    dataset_path: Path,
    manifest_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Validate source grounding and split policy without exposing patient content."""
    try:
        manifest = DatasetManifest.model_validate(_safe_json(manifest_path))
    except (OSError, ValueError, TypeError) as exc:
        raise DatasetValidationError("Invalid training dataset manifest") from exc
    examples = read_training_examples(dataset_path)
    loaded_ccos: dict[Path, tuple[Any, dict[str, Any]]] = {}
    roots: dict[str, set[str]] = defaultdict(set)
    decisions: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    reviews: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    trainable: Counter[str] = Counter()
    legacy_fact_ids_present = 0

    for index, example in enumerate(examples, 1):
        try:
            cco_path = _resolve_cco(example.cco_file, project_root)
            if cco_path not in loaded_ccos:
                record = parse_cco_text(_safe_json(cco_path))
                loaded_ccos[cco_path] = (record, evidence_span_index(record))
            record, available_evidence = loaded_ccos[cco_path]
            if example.snapshot_id != record.id:
                raise ValueError("snapshot mismatch")
            if example.content_hash != record.content_hash:
                raise ValueError("content hash mismatch")
            for annotation in example.evidence_annotations:
                if annotation.fact_id is not None:
                    legacy_fact_ids_present += 1
                span = available_evidence.get(annotation.evidence_id)
                if span is None:
                    raise ValueError("unknown evidence reference")
                if annotation.source_span.model_dump(by_alias=True) != span.location():
                    raise ValueError("evidence source span mismatch")
                explicit = extract_explicit_claims(span.section_heading, span.text)
                if not set(annotation.explicit_claims).issubset(explicit):
                    raise ValueError("unsupported explicit source claim")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise DatasetValidationError(
                f"Source grounding failed for example number {index}"
            ) from exc
        roots[example.root_id].add(example.split)
        decisions[example.personalization_decision] += 1
        splits[example.split] += 1
        reviews[example.provenance.review_status] += 1
        variants[example.variant] += 1
        trainable["true" if example.trainable else "false"] += 1

    leaking_roots = sum(len(root_splits) > 1 for root_splits in roots.values())
    if leaking_roots:
        raise DatasetValidationError("Scenario roots cross dataset splits")
    not_supported_fraction = decisions["not_supported"] / len(examples)
    if not_supported_fraction > manifest.policy.max_not_supported_fraction + 1e-12:
        raise DatasetValidationError("not_supported examples exceed the manifest policy")
    reviewed_fraction = reviews["reviewed"] / len(examples)
    if reviewed_fraction < manifest.policy.min_reviewed_fraction:
        raise DatasetValidationError("Reviewed examples are below the manifest policy")

    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    return {
        "valid": True,
        "datasetId": manifest.dataset_id,
        "purpose": manifest.purpose,
        "schemaVersion": SCHEMA_VERSION,
        "examples": len(examples),
        "uniqueRoots": len(roots),
        "uniqueCcos": len(loaded_ccos),
        "splits": dict(sorted(splits.items())),
        "personalizationDecisions": dict(sorted(decisions.items())),
        "notSupportedFraction": round(not_supported_fraction, 6),
        "reviewStatus": dict(sorted(reviews.items())),
        "variants": dict(sorted(variants.items())),
        "trainable": dict(sorted(trainable.items())),
        "readyForTraining": trainable["true"] == len(examples),
        "sha256": digest,
        "primarySource": manifest.policy.primary_source,
        "legacyFactRole": manifest.policy.legacy_fact_role,
        "legacyFactIdsPresent": legacy_fact_ids_present,
        "factIdsUsedAsPrimaryEvidence": False,
        "clinicalCorrectnessEvaluated": False,
        "answerQualityEvaluated": False,
    }

