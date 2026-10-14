from fastapi.testclient import TestClient

from demo.mobile_chat_api.app import create_mobile_app

OFFICE_CLIENT = ("10.34.14.50", 50100)
OUTSIDE_CLIENT = ("192.168.1.25", 50101)
TOKEN = "demo-token"


def make_client(*, client=OFFICE_CLIENT, completion=None):
    async def default_completion(_messages):
        return {"answer": "Yanıt", "model": "gemma-test", "latencyMs": 12}

    app = create_mobile_app(
        records={"P01": "HAM CCO METNİ"},
        completion=completion or default_completion,
        access_token=TOKEN,
        allowed_cidrs=("10.34.14.0/23", "127.0.0.0/8"),
        allowed_hosts=("testserver",),
    )
    return TestClient(app, client=client)


def test_mobile_api_requires_office_network_and_token():
    with make_client() as client:
        missing_token = client.get("/api/health")
        accepted = client.get("/api/health", headers={"X-Demo-Token": TOKEN})
    with make_client(client=OUTSIDE_CLIENT) as client:
        outside = client.get("/api/health", headers={"X-Demo-Token": TOKEN})

    assert missing_token.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json() == {"status": "ready", "modelReady": True, "patientCount": 1}
    assert outside.status_code == 403


def test_mobile_api_returns_aliases_without_patient_text():
    with make_client() as client:
        response = client.get("/api/patients", headers={"X-Demo-Token": TOKEN})

    assert response.status_code == 200
    assert response.json() == [{"id": "P01", "label": "Hasta 1"}]
    assert "HAM CCO METNİ" not in response.text
    assert response.headers["cache-control"] == "no-store"


def test_mobile_api_sends_unchanged_record_and_history_to_model():
    calls = []

    async def complete(messages):
        calls.append(messages)
        return {
            "answer": "**Son kayıtlarına göre**\n\nA) Kaşıntı\nB) Batma",
            "model": "gemma-test",
            "latencyMs": 25,
        }

    with make_client(completion=complete) as client:
        response = client.post(
            "/api/chat",
            headers={"X-Demo-Token": TOKEN},
            json={
                "patientId": "P01",
                "question": "Gözüm neden kızardı?",
                "history": [{"role": "user", "content": "Dün başladı."}],
            },
        )

    assert response.status_code == 200
    assert response.json()["latencyMs"] == 25
    assert calls[0][1] == {"role": "user", "content": "HAM CCO METNİ"}
    assert calls[0][2] == {"role": "user", "content": "Dün başladı."}
    assert "Gözüm neden kızardı?" in calls[0][-1]["content"]


def test_mobile_api_rejects_unknown_patient():
    with make_client() as client:
        response = client.post(
            "/api/chat",
            headers={"X-Demo-Token": TOKEN},
            json={"patientId": "P99", "question": "Merhaba", "history": []},
        )

    assert response.status_code == 404


def test_mobile_api_uses_compact_contextual_prompt_for_quick_reply():
    calls = []

    async def complete(messages):
        calls.append(messages)
        return {"answer": "Kısa takip yanıtı.", "model": "gemma-test", "latencyMs": 18}

    with make_client(completion=complete) as client:
        response = client.post(
            "/api/chat",
            headers={"X-Demo-Token": TOKEN},
            json={
                "patientId": "P01",
                "question": "Kaşıntı veya sulanma var",
                "quickReply": True,
                "history": [
                    {"role": "user", "content": "Gözüm biraz kızarık."},
                    {
                        "role": "assistant",
                        "content": "Hangisi var? A) Kaşıntı veya sulanma var",
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert calls[0][1] == {"role": "user", "content": "HAM CCO METNİ"}
    assert calls[0][2]["content"] == "Gözüm biraz kızarık."
    assert "önceki takip sorusuna seçtiği yanıt" in calls[0][-1]["content"]
    assert "2-3 kısa, doğal cümlede" in calls[0][-1]["content"]
    assert "Yanıtı şu sırayla" not in calls[0][-1]["content"]
