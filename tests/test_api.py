import json
from pathlib import Path

import httpx
import pytest
import yaml
from evidence_support import synthetic_cco
from fastapi.testclient import TestClient

from medical_ai_selective_context.evidence import build_evidence
from medical_ai_selective_context.llm import ModelClient, ModelSettings
from medical_ai_selective_context.main import app, create_app

ROOT = Path(__file__).parents[1]


class OfflineGraph:
    available = False

    async def initialize(self, _):
        return self.status()

    def status(self):
        return {"status": "unavailable", "phiPolicy": "terminology-and-fact-metadata-only"}

    async def close(self):
        pass


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(
        tmp_path / "cco", graph=OfflineGraph(), embedding_model="",
        load_sample=False, model_client=ModelClient(ModelSettings()),
    )) as instance:
        yield instance


def test_contract_file_is_valid_openapi():
    contract = yaml.safe_load((ROOT / "openapi.yaml").read_text(encoding="utf-8"))
    assert contract["openapi"].startswith("3.1")
    assert set(contract["paths"]) == set(app.openapi()["paths"])
    assert "snapshotId" in contract["components"]["schemas"]["SelectionRequest"]["properties"]


def test_import_select_and_source_without_cco_or_model_services(client):
    document = synthetic_cco()
    original = json.dumps(document, sort_keys=True)
    imported = client.post("/api/v1/ccos", json={"cco": document})
    assert imported.status_code == 201, imported.text
    snapshot = imported.json()["snapshotId"]
    assert imported.json()["dataClass"] == "real"
    assert client.get("/api/v1/ccos").json()[0]["snapshotId"] == snapshot
    selected = client.post("/api/v1/context/select", json={
        "snapshotId": snapshot, "question": "Kreatinin geçmişi nasıl?",
    })
    assert selected.status_code == 200, selected.text
    packet = selected.json()
    assert packet["snapshotId"] == snapshot
    assert packet["canAnswer"] is True
    assert len(packet["evidence"]) >= 4
    assert packet["indexStatus"]["semantic"] == "disabled"
    assert json.dumps(client.get(f"/api/v1/ccos/{snapshot}").json(), sort_keys=True) == original
    assert "selectiveContext" not in client.get(f"/api/v1/ccos/{snapshot}").json()
    assert selected.headers["cache-control"] == "no-store"


def test_real_data_chat_blocked_but_returns_local_packet(client):
    snapshot = client.post("/api/v1/ccos", json={"cco": synthetic_cco()}).json()["snapshotId"]
    result = client.post("/api/v1/chat", json={
        "snapshotId": snapshot, "question": "Kreatinin geçmişim?",
    }).json()
    assert result["status"] == "blocked"
    assert result["validation"]["error"] == "external_real_data_blocked"
    assert result["contextPacket"]["evidence"]


def test_clients_cannot_label_arbitrary_cco_synthetic(client):
    response = client.post("/api/v1/ccos", json={
        "cco": synthetic_cco(), "dataClass": "synthetic",
    })
    assert response.status_code == 422
    assert "input" not in response.text


def test_selection_requires_snapshot_and_rejects_subject_only(client):
    response = client.post("/api/v1/context/select", json={
        "subjectId": synthetic_cco()["subjectId"], "question": "Kreatinin?",
    })
    assert response.status_code == 422


def test_raw_document_selection_is_ephemeral(client):
    response = client.post("/api/v1/context/select-document", json={
        "cco": synthetic_cco(), "question": "Kreatinin ölçümleri",
    })
    assert response.status_code == 200
    assert client.get("/api/v1/ccos").json() == []


def test_missing_snapshot_and_invalid_schema_fail_closed(client):
    assert client.get("/api/v1/ccos/missing").status_code == 404
    document = synthetic_cco()
    document["sections"]["lab"]["facts"][0]["value"]["flag"] = "high"
    result = client.post("/api/v1/ccos", json={"cco": document})
    assert result.status_code == 422
    assert "mg/dL" not in result.text


def test_local_ui_and_cross_origin_write_guard(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/static/app.js").status_code == 200
    blocked = client.post("/api/v1/ccos/reload", headers={"Origin": "https://malicious.example"})
    assert blocked.status_code == 403


@pytest.mark.parametrize("question", [" ", "\t\n"])
def test_blank_questions_fail_validation_without_input_echo(client, question):
    result = client.post("/api/v1/context/select", json={
        "snapshotId": "missing", "question": question,
    })
    assert result.status_code == 422
    assert "input" not in result.text
    result = client.post("/api/v1/context/select-document", json={
        "cco": synthetic_cco(), "question": question,
    })
    assert result.status_code == 422


def test_synthetic_end_to_end_has_two_model_calls_and_exact_citations(tmp_path):
    calls = []
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": [
                {"supported_parameters": ["structured_outputs"]},
            ]}})
        body = json.loads(request.content)
        calls.append(body)
        payload = json.loads(body["messages"][1]["content"])
        if body["response_format"]["json_schema"]["name"] == "retrieval_plan":
            content = {
                "queries": ["kreatinin"], "concepts": [], "timeIntent": "history",
                "dateFrom": None, "dateTo": None, "needsClarification": False,
                "clarificationQuestion": None,
            }
        else:
            row = next(entry for entry in payload["context"]["evidence"]
                       if (entry.get("value") or {}).get("amount") == 0.9)
            content = {"statements": [{
                "text": "Kaynakta kreatinin 0,9 mg/dL.",
                "evidenceRefs": [row["ref"]],
                "values": [
                    {"ref": row["ref"], "field": "value.amount", "value": 0.9},
                    {"ref": row["ref"], "field": "value.unit", "value": "mg/dL"},
                ],
            }]}
        return httpx.Response(200, json={
            "model": body["model"], "provider": "test-provider",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        })
    model = ModelClient(ModelSettings(api_key="test-key"), httpx.MockTransport(handler))
    with TestClient(create_app(
        tmp_path / "cco", model_client=model, graph=OfflineGraph(),
        embedding_model="", load_sample=False,
    )) as client:
        svc = client.app.state.services
        doc = synthetic_cco()
        snapshot = svc.store.import_document(doc, data_class="synthetic")["snapshotId"]
        svc.index.index_snapshot(snapshot, build_evidence(doc, snapshot))
        result = client.post("/api/v1/chat", json={
            "snapshotId": snapshot, "question": "Kreatinin geçmişim?",
        })
        assert result.status_code == 200, result.text
        output = result.json()
        assert output["status"] == "answered", output
        assert len(calls) == 2
        selected_refs = {item["ref"] for item in output["contextPacket"]["evidence"]}
        assert all(item["ref"] in selected_refs for item in output["citations"])
        assert output["validation"]["clinicalMeaningVerified"] is False

