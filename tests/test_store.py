import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from evidence_support import synthetic_cco

from medical_ai_selective_context.store import CcoStore, unwrap_and_validate_cco


def create_store(tmp_path: Path) -> CcoStore:
    return CcoStore(tmp_path / "cco", tmp_path / "runtime" / "metadata.sqlite3")


def test_actual_complete_synthetic_sample_matches_producer_schema() -> None:
    path = Path(__file__).parents[1] / "data" / "synthetic" / "cco-complete.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert unwrap_and_validate_cco({"Document": document}) == document
    assert sum(len(section["facts"]) for section in document["sections"].values()) == 162


def test_document_is_not_repaired_or_mutated() -> None:
    document = synthetic_cco()
    result = unwrap_and_validate_cco({"document": document})
    assert result == document
    result["sections"].clear()
    assert document["sections"]
    del document["subjectId"]
    with pytest.raises(ValueError, match="required"):
        unwrap_and_validate_cco({"document": document, "subjectId": "0" * 36})


@pytest.mark.parametrize("invalid", ["0.4.0", "0.5.0", "0.6.0", "9.9.9"])
def test_only_current_schema_supported(invalid: str) -> None:
    document = synthetic_cco()
    document["schemaVersion"] = invalid
    with pytest.raises(ValueError, match="Unsupported CCO"):
        unwrap_and_validate_cco(document)


@pytest.mark.parametrize("field,value", [("id", "not-a-uuid"), ("asOf", "yesterday"),
                                         ("additional", "secret-patient-value")])
def test_strict_formats_properties_and_private_errors(field: str, value: str) -> None:
    document = synthetic_cco()
    document[field] = value
    with pytest.raises(ValueError) as error:
        unwrap_and_validate_cco(document)
    assert value not in str(error.value)


def test_scalar_flag_rejected_boolean_flag_not_interpreted() -> None:
    document = synthetic_cco()
    value = document["sections"]["lab"]["facts"][0]["value"]
    value["flag"] = True
    with pytest.raises(ValueError):
        unwrap_and_validate_cco(document)
    value.pop("amount")
    value["type"] = "boolean"
    assert unwrap_and_validate_cco(document)["sections"]["lab"]["facts"][0]["value"] == value


def test_import_deduplicates_content_not_patient_and_never_falls_back(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    document = synthetic_cco()
    first = store.import_document(document)
    assert first == store.import_document({"document": document})
    document["asOf"] = "2026-10-12T00:00:00Z"
    second = store.import_document(document)
    assert first["snapshotId"] != second["snapshotId"]
    assert store.count == 2
    assert store.get(document["subjectId"]) is None
    assert store.get(first["snapshotId"])["asOf"] == "2026-10-11T00:00:00Z"
    returned = store.get(first["snapshotId"])
    returned["sections"].clear()
    assert store.get(first["snapshotId"])["sections"]
    store.close()


def test_filename_collision_does_not_overwrite_existing_snapshot(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    first = store.import_document(synthetic_cco(), "capture.json")
    changed = synthetic_cco()
    changed["asOf"] = "2026-10-12T00:00:00Z"
    with pytest.raises(ValueError, match="refusing overwrite"):
        store.import_document(changed, "capture.json")
    assert store.count == 1
    assert json.loads((store.data_dir / "capture.json").read_text())["asOf"] == first["asOf"]
    for path in ("../escape.json", r"..\escape.json", "CON.json", "sub/file.json"):
        with pytest.raises(ValueError):
            store.import_document(changed, path)
    store.close()


def test_archives_survive_manual_source_replacement_and_restart(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    original = synthetic_cco()
    old = store.import_document(original, "capture.json")
    changed = synthetic_cco()
    changed["asOf"] = "2026-10-12T00:00:00Z"
    (store.data_dir / "capture.json").write_text(json.dumps(changed), encoding="utf-8")
    assert store.reload()["loaded"] == 2
    store.close()
    reopened = create_store(tmp_path)
    assert reopened.reload()["loaded"] == 2
    assert reopened.get(old["snapshotId"]) == original
    reopened.close()


def test_classification_requires_trusted_argument_and_cannot_promote_real(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    document = synthetic_cco()
    first = store.import_document(document, data_class="synthetic")
    assert store.reload()["loaded"] == 1
    assert store.summary(first["snapshotId"])["dataClass"] == "synthetic"
    assert store.import_document(document)["dataClass"] == "real"
    assert store.import_document(document, data_class="synthetic")["dataClass"] == "real"
    store.close()


def test_threadpool_safe_reads(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    snapshot_id = store.import_document(synthetic_cco())["snapshotId"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(store.get, [snapshot_id] * 8))
    assert all(item == synthetic_cco() for item in results)
    store.close()


def test_rejects_nonfinite_numeric_values() -> None:
    document = synthetic_cco()
    document["sections"]["lab"]["facts"][0]["value"]["amount"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        unwrap_and_validate_cco(document)
