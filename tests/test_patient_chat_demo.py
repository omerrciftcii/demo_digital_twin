from fastapi.testclient import TestClient

from demo.patient_chat.app import create_app


def test_demo_serves_local_ui_and_patient_aliases():
    async def complete(_messages):
        return {"answer": "Yanıt", "model": "test", "latencyMs": 4}

    with TestClient(create_app(records={"P01": "HAM METİN"}, completion=complete)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "MedicalAI" in page.text
        assert "Adım adım" in page.text
        assert "Çoklu seçim" in page.text
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


def test_demo_uses_short_continuation_prompt_for_clicked_option():
    calls = []

    async def complete(messages):
        calls.append(messages)
        return {"answer": "Kısa takip yanıtı.", "model": "gemma-test", "latencyMs": 9}

    with TestClient(create_app(records={"P01": "HAM METİN"}, completion=complete)) as client:
        response = client.post(
            "/api/chat",
            json={
                "patientId": "P01",
                "question": "Yanma veya batma var",
                "quickReply": True,
                "history": [
                    {"role": "user", "content": "Gözüm kızardı."},
                    {"role": "assistant", "content": "Hangisi var?\n\nA) Yanma var"},
                ],
            },
        )

    assert response.status_code == 200
    assert calls[0][1] == {"role": "user", "content": "HAM METİN"}
    assert "önceki takip sorusuna seçtiği yanıt" in calls[0][-1]["content"]
    assert "2-3 kısa, doğal cümlede" in calls[0][-1]["content"]
    assert "Yanıtı şu sırayla" not in calls[0][-1]["content"]


def test_demo_requests_patient_specific_multi_select_options():
    calls = []

    async def complete(messages):
        calls.append(messages)
        return {"answer": "A) Yeni başladı\nB) Ağrı var", "model": "gemma-test", "latencyMs": 7}

    with TestClient(create_app(records={"P01": "HAM METİN"}, completion=complete)) as client:
        response = client.post(
            "/api/chat",
            json={
                "patientId": "P01",
                "question": "Gözüm kızardı",
                "interactionMode": "multi_select",
            },
        )

    assert response.status_code == 200
    prompt = calls[0][-1]["content"]
    assert "Gözüm kızardı" in prompt
    assert "tam 5 veya 6 kısa seçenek" in prompt
    assert "sorusuna ve verilen hasta kaydındaki" in prompt


def test_demo_evaluates_multiple_selected_options_together():
    calls = []

    async def complete(messages):
        calls.append(messages)
        return {"answer": "Birleşik değerlendirme.", "model": "gemma-test", "latencyMs": 8}

    with TestClient(create_app(records={"P01": "HAM METİN"}, completion=complete)) as client:
        response = client.post(
            "/api/chat",
            json={
                "patientId": "P01",
                "question": "Seçtiklerim",
                "quickReply": True,
                "interactionMode": "multi_select",
                "selectedOptions": ["Yanma var", "Dün başladı"],
                "history": [
                    {"role": "user", "content": "Gözüm kızardı"},
                    {"role": "assistant", "content": "A) Yanma var\nB) Dün başladı"},
                ],
            },
        )

    assert response.status_code == 200
    prompt = calls[0][-1]["content"]
    assert "- Yanma var" in prompt
    assert "- Dün başladı" in prompt
    assert "birlikte işaretlediği" in prompt
    assert 'İlk cümleyi "Seçimlerine göre, ... birlikte öne çıkıyor."' in prompt
    assert "Bu ilk cümlede tanı koyma" in prompt
    assert "Seçimleri yalnızca tekrarlama" in prompt
    assert "önceki seçenekleri tekrarlama" in prompt
    assert "A)-F) seçenek satırları üretme" in prompt
    assert "Son kayıtlarına göre" not in prompt


def test_demo_rejects_more_than_six_selected_options():
    async def complete(_messages):
        raise AssertionError("model must not be called")

    with TestClient(create_app(records={"P01": "HAM METİN"}, completion=complete)) as client:
        response = client.post(
            "/api/chat",
            json={
                "patientId": "P01",
                "question": "Seçtiklerim",
                "interactionMode": "multi_select",
                "selectedOptions": [f"Seçenek {index}" for index in range(7)],
            },
        )

    assert response.status_code == 422
