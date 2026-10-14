from fastapi.testclient import TestClient

from demo.patient_chat.app import create_app


def test_demo_serves_local_ui_and_patient_aliases():
    async def complete(_messages):
        return {"answer": "Yanıt", "model": "test", "latencyMs": 4}

    with TestClient(create_app(records={"P01": "HAM METİN"}, completion=complete)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "MedicalAI" in page.text
        assert "script-src 'self'" in page.headers["content-security-policy"]
        assert client.get("/api/patients").json() == [{"id": "P01", "label": "Hasta 1"}]
        assert client.get("/api/health").json()["status"] == "ready"


def test_demo_sends_unchanged_record_and_returns_raw_markdown():
    calls = []

    async def complete(messages):
        calls.append(messages)
        return {
            "answer": "**Son kayıtlarına göre**\n\nA) Kaşıntı\nB) Batma",
            "model": "gemma-test",
            "latencyMs": 15,
        }

    raw_text = "İlk satır\n\n- Değer: 42 mg/dL"
    with TestClient(create_app(records={"P01": raw_text}, completion=complete)) as client:
        response = client.post(
            "/api/chat",
            json={"patientId": "P01", "question": "Gözüm neden kızardı?", "history": []},
        )
    assert response.status_code == 200
    assert response.json()["answer"].startswith("**Son kayıtlarına göre**")
    assert calls[0][1] == {"role": "user", "content": raw_text}
    assert "Gözüm neden kızardı?" in calls[0][-1]["content"]
    assert len(calls[0]) == 3


def test_demo_rejects_unknown_patient_and_cross_origin_write():
    async def complete(_messages):
        raise AssertionError("model must not be called")

    with TestClient(create_app(records={"P01": "HAM METİN"}, completion=complete)) as client:
        missing = client.post(
            "/api/chat", json={"patientId": "P99", "question": "Merhaba", "history": []}
        )
        blocked = client.post(
            "/api/chat",
            headers={"Origin": "https://example.invalid"},
            json={"patientId": "P01", "question": "Merhaba", "history": []},
        )
    assert missing.status_code == 404
    assert blocked.status_code == 403
