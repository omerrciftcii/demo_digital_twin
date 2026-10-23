import datetime as dt

from fastapi.testclient import TestClient

from demo.patient_chat.app import SYSTEM_PROMPT, create_app
from demo.patient_chat.preprocess_s import S_CONTEXT_NOTE, preprocessed_record

RECORD = """45 yaşında kadın.

## Tanılar (hastane kaydı)
Diğer:
- Örnek tanı A (A10) — kesin tanı; son kayıt 01.02.2020, Dahiliye.
- Örnek tanı A (A10.1) — kesin tanı; son kayıt 01.03.2022, Dahiliye.
- Örnek tanı B (B20) — kesin tanı; son kayıt 05.10.2026, Dahiliye.

## Hastane başvuruları
- 05.10.2026, Dahiliye, genel muayene.
"""


def _client(calls):
    async def complete(messages):
        calls.append(messages)
        system = messages[0]["content"]
        if system.startswith("Görevin yalnızca etiketleme"):
            return {"answer": '{"etiketler": []}', "model": "test", "latencyMs": 1}
        if system.startswith("Görevin yalnızca seçim"):
            return {"answer": '{"ilgili": [{"id": "k1", "neden": "kronik_durum"}]}', "model": "test", "latencyMs": 1}
        return {"answer": "Yanıt", "model": "test", "latencyMs": 1}

    return TestClient(create_app(records={"P01": RECORD}, completion=complete))


def test_raw_mode_is_default_and_unchanged():
    calls = []
    with _client(calls) as client:
        response = client.post("/api/chat", json={"patientId": "P01", "question": "Nasılım?"})
    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][0]["content"] == SYSTEM_PROMPT
    assert calls[0][1]["content"] == RECORD.strip()
    assert response.json()["preprocessMode"] == "raw"


def test_s6b_mode_uses_preprocessed_context():
    calls = []
    with _client(calls) as client:
        response = client.post(
            "/api/chat",
            json={"patientId": "P01", "question": "Örnek tanı A tekrar mı etti?", "preprocessMode": "s6b"},
        )
    assert response.status_code == 200
    final = calls[-1]
    assert final[0]["content"].endswith(S_CONTEXT_NOTE)
    assert "## Zaman özeti" in final[1]["content"]
    assert "Geçmiş kayıtlar (bugünkü durumu yansıtmaz, ilgili olabilir)" in final[1]["content"]
    assert response.json()["answer"] == "Yanıt"
    assert response.json()["preprocessInfo"]["mode"] == "s6b"


def test_s5_mode_and_invalid_mode():
    calls = []
    with _client(calls) as client:
        ok = client.post("/api/chat", json={"patientId": "P01", "question": "Nasılım?", "preprocessMode": "s5"})
        bad = client.post("/api/chat", json={"patientId": "P01", "question": "Nasılım?", "preprocessMode": "x"})
    assert ok.status_code == 200 and ok.json()["preprocessMode"] == "s5"
    assert bad.status_code == 422


def test_preprocessed_record_marks_ages():
    text = preprocessed_record(RECORD.strip(), dt.date(2026, 10, 22))
    assert "Bugün: 22.10.2026" in text
    assert "[17 gün önce]" in text
