from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class CcoImportRequest(ApiModel):
    cco: dict[str, Any]
    file_name: str | None = Field(
        default=None, alias="fileName", pattern=r"^[A-Za-z0-9._-]+\.json$"
    )


class CcoSummary(ApiModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    snapshot_id: str = Field(alias="snapshotId")
    subject_id: str = Field(alias="subjectId")
    schema_version: str = Field(alias="schemaVersion")
    as_of: str = Field(alias="asOf")
    fact_count: int = Field(alias="factCount", ge=0)
    source_file: str = Field(alias="sourceFile")
    data_class: Literal["real", "synthetic"] = Field(alias="dataClass")
    content_hash: str = Field(alias="contentHash")


class HistoryMessage(ApiModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class SelectionRequest(ApiModel):
    snapshot_id: str = Field(alias="snapshotId", min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)
    token_budget: int = Field(default=8000, alias="tokenBudget", ge=512, le=16000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=6)

    @field_validator("question", "snapshot_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must not be blank")
        return value.strip()


class DocumentSelectionRequest(ApiModel):
    cco: dict[str, Any]
    question: str = Field(min_length=1, max_length=4000)
    token_budget: int = Field(default=8000, alias="tokenBudget", ge=512, le=16000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must not be blank")
        return value.strip()


class SearchPlan(ApiModel):
    queries: list[str] = Field(min_length=1, max_length=5)
    concepts: list[str] = Field(default_factory=list, max_length=12)
    time_intent: Literal["current", "history", "compare", "overview"] = Field(alias="timeIntent")
    date_from: str | None = Field(default=None, alias="dateFrom")
    date_to: str | None = Field(default=None, alias="dateTo")
    needs_clarification: bool = Field(default=False, alias="needsClarification")
    clarification_question: str | None = Field(default=None, alias="clarificationQuestion")


class ValueAssertion(ApiModel):
    ref: str
    field: Literal[
        "value.amount", "value.unit", "value.flag", "value.code", "value.display",
        "occurredFrom", "occurredTo", "producedAt",
    ]
    value: str | int | float | bool | None


class AnswerStatement(ApiModel):
    text: str = Field(min_length=1, max_length=2000)
    evidence_refs: list[str] = Field(alias="evidenceRefs", min_length=1, max_length=20)
    values: list[ValueAssertion] = Field(default_factory=list, max_length=40)


class GroundedAnswer(ApiModel):
    statements: list[AnswerStatement] = Field(min_length=1, max_length=12)


class ContextPacket(ApiModel):
    # A packet is not a modified CCO document.
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    packet_id: str = Field(alias="packetId")
    snapshot_id: str = Field(alias="snapshotId")
    can_answer: bool = Field(alias="canAnswer")
    evidence: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    coverage: list[dict[str, Any]]
    warnings: list[Any]
    model_context: dict[str, Any] = Field(alias="modelContext")


class TextAnswerRequest(ApiModel):
    record_id: str = Field(alias="recordId", min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=4000)
    stream: bool = False
    history: list[HistoryMessage] = Field(default_factory=list, max_length=6)
    validation: bool | None = None

    @field_validator("question", "record_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must not be blank")
        return value.strip()


class ChatResponse(ApiModel):
    status: Literal[
        "answered", "answered_unvalidated", "insufficient_evidence", "blocked",
        "model_unavailable", "validation_failed", "clarification_required",
    ]
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    context_packet: dict[str, Any] = Field(alias="contextPacket")
    metrics: dict[str, Any] = Field(default_factory=dict)
    model: str
    provider: str | None = None
    validation: dict[str, Any] = Field(default_factory=dict)

