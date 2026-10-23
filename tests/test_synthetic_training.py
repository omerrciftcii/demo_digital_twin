import json
from pathlib import Path

from medical_ai_selective_context.synthetic_training import (
    SyntheticEvent,
    SyntheticScenario,
    build_synthetic_example,
    generate_synthetic_pilot,
    prepare_gold_review_queue,
    render_event_graph,
)
from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.training_dataset import validate_training_dataset


def _scenario() -> SyntheticScenario:
    return SyntheticScenario(
        scenarioId="scenario-preliminary-001",
        snapshotId="11111111-1111-4111-8111-111111111111",
        subjectId="22222222-2222-4222-8222-222222222222",
        asOf="2026-10-21T12:00:00+03:00",
        age=42,
        sex="kadın",
        question="Kayıtlarda migren tanım var mı?",
        answer="Kayıtta migren ön tanı olarak geçiyor; kesin tanı olarak sunamam.",
        intents=["lookup"],
        subjects=["diagnosis"],
        personalizationDecision="required",
        events=[
            SyntheticEvent(
                eventId="event-migraine",
                kind="diagnosis",
                display="Migren",
                occurredOn="2026-09-03",
                assertion="preliminary",
                diagnosisGroup="other",
                requiredness="must_use",
                useRole="direct_fact",
                reasonCode="explicit_preliminary_diagnosis",
            ),
            SyntheticEvent(
                eventId="event-old-infection",
                kind="diagnosis",
                display="Eski üst solunum yolu enfeksiyonu",
                occurredOn="2021-01-02",
                assertion="definitive",
                diagnosisGroup="acute",
                requiredness="not_needed",
                useRole="none",
                reasonCode="unrelated_old_acute_record",
            ),
        ],
    )


def test_event_graph_renders_text_cco_and_binds_gold_to_rendered_spans():
    scenario = _scenario()

    rendered = render_event_graph(scenario)
    example = build_synthetic_example(scenario, rendered, split="train")
    record = parse_cco_text(rendered.cco)

    assert set(rendered.cco) == {"id", "subjectId", "asOf", "text"}
    assert record.id == scenario.snapshot_id
    assert set(rendered.event_evidence) == {"event-migraine", "event-old-infection"}
    assert len(example["evidenceAnnotations"]) == 2
    assert example["contentHash"] == record.content_hash
    assert all(item["factId"] is None for item in example["evidenceAnnotations"])
    assert example["trainable"] is False
    assert example["goldAnswer"] is None
    assert example["candidateAnswer"] == scenario.answer
    assert example["evidenceAnnotations"][0]["explicitClaims"] == [
        "assertion_preliminary"
    ]


def test_unspecified_state_is_not_invented_during_rendering():
    scenario = _scenario()
    scenario.events = [
        SyntheticEvent(
            eventId="event-prescription",
            kind="medication_prescription",
            display="Örnek ilaç",
            occurredOn="2024-02-03",
            requiredness="must_use",
            useRole="status_check",
            safetyCritical=True,
            reasonCode="old_prescription_not_current_use",
        )
    ]

    rendered = render_event_graph(scenario)

    assert "reçete" in rendered.cco["text"].casefold()
    assert "aktif" not in rendered.cco["text"].casefold()
    assert "kullanıyor" not in rendered.cco["text"].casefold()
    assert "iyileş" not in rendered.cco["text"].casefold()


def test_generator_creates_requested_count_and_validates(tmp_path: Path):
    package = generate_synthetic_pilot(tmp_path, count=12, seed=7)

    annotations = [
        json.loads(line)
        for line in package.annotations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    graphs = [
        json.loads(line)
        for line in package.event_graphs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = validate_training_dataset(
        package.annotations,
        package.manifest,
        package.project_root,
    )

    assert len(annotations) == 12
    assert len(graphs) == 12
    assert report["valid"] is True
    assert report["examples"] == 12
    assert report["factIdsUsedAsPrimaryEvidence"] is False
    assert package.event_graphs.read_text(encoding="utf-8").count("scenarioId") == 12


def test_gold_queue_keeps_real_text_local_and_starts_without_gold_labels(tmp_path: Path):
    cco_dir = tmp_path / "private-cco"
    cco_dir.mkdir()
    payload = {
        "id": "55555555-5555-4555-8555-555555555555",
        "subjectId": "66666666-6666-4666-8666-666666666666",
        "asOf": "2026-10-21T12:00:00+03:00",
        "text": "42 yaşında kadın.\n\n## Tanılar\n- Çok gizli klinik ifade.",
    }
    (cco_dir / "source.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({
            "benchmark_id": "BENCH_TEST",
            "cco_id": payload["id"],
            "subject_id": payload["subjectId"],
            "category": "temporal_trap",
            "question": "Bugünkü şikâyetim eski kayıtla ilişkili mi?",
            "expected_criteria": ["Eski kaydı güncel durum gibi sunmamalı."],
            "failure_modes": ["Eski kaydı aktif saymak"],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    package = prepare_gold_review_queue(
        cco_dir,
        tmp_path / "gold-review",
        count=1,
        questions_path=questions,
    )

    rendered = package.tasks.read_text(encoding="utf-8")
    task = json.loads(rendered)
    assert task["reviewStatus"] == "pending"
    assert task["trainable"] is False
    assert task["goldLabels"] is None
    assert task["question"] == "Bugünkü şikâyetim eski kayıtla ilişkili mi?"
    assert task["reviewHints"]["source"] == "unreviewed_benchmark_seed"
    assert task["legacyFactRole"] == "legacy_audit_only"
    assert "Çok gizli" not in rendered
    assert payload["subjectId"] not in rendered
