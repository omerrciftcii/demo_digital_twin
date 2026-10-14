"""Full Context on the verified path. The chart and facts are made up, not fixtures."""

import asyncio
import json

import httpx

from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_evidence import build_evidence
from medical_ai_selective_context.text_llm import TextModelClient, TextModelSettings
from medical_ai_selective_context.text_qa import TextAnswerer, build_full_context

AS_OF = "2026-10-01T00:00:00+00:00"
CHART = (
    "Giriş.\n\n## Laboratuvar\n"
    "- Hemoglobin 11 g/dL (06.06.2024), düşük (aralık 12-16).\n"
    "- TSH 2 mIU/L (06.06.2024), normal (aralık 0,4-4).\n\n"
    "## Uyku\n"
    "- Ortalama uyku 6 sa (01.06.2024).\n"
)


def _record():
    return parse_cco_text({
        "format": "cco-text/1",
        "id": "00000000-0000-4000-8000-0000000000d1",
        "subjectId": "00000000-0000-4000-8000-0000000000d2",
        "asOf": AS_OF, "language": "tr", "text": CHART,
    })


def _row(fact_id, concept, display, value, unit, line, section):
    return {
        "fact_id": fact_id, "type": "lab", "concept": concept, "display_name": display,
        "value": value, "unit": unit, "date": "2024-06-06", "date_precision": "day",
        "source_section": section, "source_text": CHART.splitlines()[line - 1],
        "source_line": line, "source_type": "derived_from_cco_text", "confidence": "high",
        "requires_human_review": False,
    }


def _facts():
    return {
        "format": "cco-derived-facts/1",
        "source": {"authoritative": False, "derived_from_cco_text": True},
        "facts": [
            _row("f-hb", "hemoglobin", "Hemoglobin", 11, "g/dL", 4, "Laboratuvar"),
            _row("f-tsh", "tsh", "TSH", 2, "mIU/L", 5, "Laboratuvar"),
            _row("f-sleep", "sleep_duration", "Ortalama uyku", 6, "sa", 8, "Uyku"),
        ],
    }


def _id(fact_id):
    package = build_evidence(_record(), CHART, structured=_facts())
    return next(item.id for item in package if item.fact_id == fact_id)


def _sse(text):
    frames = [
        "data: " + json.dumps({
            "model": "local-model",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }, ensure_ascii=False) + "\n\n",
        "data: " + json.dumps({
            "model": "local-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 10},
        }) + "\n\n",
        "data: [DONE]\n\n",
    ]
    return "".join(frames).encode("utf-8")


def _answer(evidence_id):
    return json.dumps({
        "conclusion": {
            "status": "supported", "text": "Hemoglobin 11 g/dL.",
            "claim_type": "fact", "evidence_ids": [evidence_id],
        },
        "sentences": [{
            "text": "Hemoglobin değerin 11 g/dL.", "claim_type": "fact",
            "evidence_ids": [evidence_id],
        }],
        "missing_information": [],
        "requires_clinician_review": False,
    }, ensure_ascii=False)


def _server(answers, tokenize=None):
    state = {"chat": 0, "tokenize": [], "bodies": []}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            state["tokenize"].append(body)
            if tokenize is None:
                return httpx.Response(404)
            return httpx.Response(200, json=tokenize(body))
        state["bodies"].append(body)
        text = answers[min(state["chat"], len(answers) - 1)]
        state["chat"] += 1
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(text),
        )

    return handler, state


def _client(handler, base_url="http://127.0.0.1:8000/v1"):
    return TextModelClient(TextModelSettings(
        api_key="k", base_url=base_url, model="local-model",
        require_zdr=False, local_inference=True,
    ), httpx.MockTransport(handler))


def _enable(monkeypatch, full=True):
    monkeypatch.setenv("DETERMINISTIC_EVIDENCE_VERIFICATION_ENABLED", "true")
    if full:
        monkeypatch.setenv("QA_FULL_CONTEXT_ENABLED", "true")
    else:
        monkeypatch.delenv("QA_FULL_CONTEXT_ENABLED", raising=False)


def _ask(handler, question="Hemoglobinim ne çıktı?", history=None, **kwargs):
    answerer = TextAnswerer(_client(handler))
    return asyncio.run(answerer.answer(
        _record(), question, history, structured_facts=_facts(), **kwargs,
    ))


def test_full_context_sends_the_whole_chart_and_every_fact_once(monkeypatch):
    _enable(monkeypatch)
    handler, state = _server([_answer(_id("f-hb"))])
    result = _ask(handler)
    messages = state["bodies"][0]["messages"]
    assert [item["role"] for item in messages] == ["system", "user", "assistant", "user"]
    assert CHART in messages[1]["content"]
    assert all(CHART not in item["content"] for item in messages[2:])
    for fact_id in ("f-hb", "f-tsh", "f-sleep"):
        assert f"{_id(fact_id)}\t" in messages[1]["content"]
    assert "Öne çıkan" not in messages[-1]["content"]
    assert "DURUM:" not in messages[-1]["content"]
    context = result["validation"]["context"]
    assert context["mode"] == "full"
    assert context["ccoComplete"] is True
    assert context["sentCcoSha256"] == context["ccoSha256"]
    assert context["catalogFactCount"] == context["factCount"] == 3
    assert context["trimmed"] is False
    assert context["tokenCountSource"] == "estimate"
    assert result["validation"]["trace"]["dropped_for_budget_ids"] == []
    assert result["validation"]["trace"]["cited_fact_ids"] == ["f-hb"]
    assert result["validation"]["claims"] == {"total": 2, "rejected": 0}


def test_prefix_does_not_depend_on_the_question_or_the_history():
    first = build_full_context(_record(), "Hemoglobinim ne?", [], _facts())
    second = build_full_context(
        _record(), "Uykum nasıl?",
        [{"role": "user", "content": "Önceki soru"}, {"role": "assistant", "content": "Yanıt"}],
        _facts(),
    )
    assert first.messages[:3] == second.messages[:3]
    assert first.report["prefixSha256"] == second.report["prefixSha256"]
    assert second.messages[3]["content"] == "Önceki soru"
    assert second.messages[-1]["content"].endswith("Uykum nasıl?")


def test_history_cuts_are_reported_not_silent():
    history = [{"role": "user", "content": f"soru {index}"} for index in range(7)]
    history.append({"role": "assistant", "content": "x" * 2500})
    report = build_full_context(_record(), "Soru?", history, _facts()).report["history"]
    assert report == {"received": 8, "sent": 6, "dropped": 2, "truncated": 1}


def test_overflow_does_not_call_the_model_and_does_not_trim(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setenv("QA_CONTEXT_TOKENS", "600")
    handler, state = _server([_answer(_id("f-hb"))])
    result = _ask(handler)
    assert state["chat"] == 0
    assert result["status"] == "context_overflow"
    assert result["metrics"]["modelCalls"] == 0
    assert result["validation"]["errorCodes"] == ["context_overflow"]
    context = result["validation"]["context"]
    assert context["fits"] is False
    assert context["ccoComplete"] is True
    assert context["requiredTokens"] > context["contextLimit"] == 600


def test_served_tokenizer_count_and_limit_win_over_the_estimate(monkeypatch):
    _enable(monkeypatch)
    handler, state = _server(
        [_answer(_id("f-hb"))],
        tokenize=lambda body: {"count": 3000, "max_model_len": 4096, "tokens": []},
    )
    result = _ask(handler)
    assert state["chat"] == 0
    sent = state["tokenize"][0]
    assert sent["model"] == "local-model"
    assert sent["add_generation_prompt"] is True
    assert sent["messages"][1]["content"].count(CHART) == 1
    context = result["validation"]["context"]
    assert context["tokenCountSource"] == "vllm_tokenize"
    assert context["promptTokens"] == 3000
    assert context["serverMaxModelLen"] == 4096
    assert context["contextLimit"] == 4096
    assert result["status"] == "context_overflow"


def test_retry_that_would_overflow_is_skipped_and_reported(monkeypatch):
    _enable(monkeypatch)

    def count(body):
        return {"count": 20000 if len(body["messages"]) > 4 else 500, "max_model_len": 16384}

    handler, state = _server(["hâlâ json değil"], tokenize=count)
    result = _ask(handler)
    assert state["chat"] == 1
    assert result["metrics"]["retry"] is False
    assert result["validation"]["trace"]["retry_skipped"] == "context_overflow"
    assert result["validation"]["context"]["retryFits"] is False


def test_count_tokens_never_sends_text_to_an_untrusted_host():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    for base_url in ("https://openrouter.ai/api/v1", "http://8.8.8.8:8000/v1"):
        client = _client(refuse, base_url)
        assert asyncio.run(client.count_tokens([{"role": "user", "content": "kayıt"}])) is None


def test_oracle_sends_only_gold_lines_and_facts():
    plan = build_full_context(_record(), "TSH?", [], _facts(), oracle_fact_ids={"f-tsh"})
    content = plan.messages[1]["content"]
    assert "## Laboratuvar" in content
    assert "- TSH 2 mIU/L" in content
    assert "Hemoglobin" not in content
    assert "Ortalama uyku" not in content
    assert [item.fact_id for item in plan.prompt_package] == ["f-tsh"]
    assert len(plan.package) == 3
    assert plan.report["mode"] == "oracle"
    assert plan.report["oracleFactCount"] == 1
    assert plan.report["ccoComplete"] is False
    assert plan.report["sectionsSent"] == 1
    empty = build_full_context(_record(), "Göz?", [], _facts(), oracle_fact_ids=[])
    assert empty.report["catalogFactCount"] == 0
    assert "Hemoglobin" not in empty.messages[1]["content"]


def test_oracle_keeps_every_fact_on_a_sent_line_bindable():
    facts = _facts()
    facts["facts"].append(
        _row("f-hb-range", "hemoglobin_range", "Hemoglobin aralığı", 12, "g/dL", 4, "Laboratuvar"),
    )
    plan = build_full_context(_record(), "Hemoglobin?", [], facts, oracle_fact_ids={"f-hb"})
    assert {item.fact_id for item in plan.prompt_package} == {"f-hb", "f-hb-range"}
    assert plan.report["oracleFactCount"] == 1
    assert plan.report["catalogFactCount"] == 2
    assert "TSH" not in plan.messages[1]["content"]


def test_default_verified_path_is_unchanged(monkeypatch):
    _enable(monkeypatch, full=False)
    handler, state = _server([_answer(_id("f-hb"))], tokenize=lambda body: {"count": 1})
    result = _ask(handler)
    assert state["tokenize"] == []
    assert "context" not in result["validation"]
    assert result["validation"]["claims"]["total"] == 2
