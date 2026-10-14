import asyncio

import httpx
import pytest

from scripts.run_patient_centric_gpu_eval import AuditedClient, coverage
from scripts.run_patient_centric_retrieval_eval import _satisfied, expand_cases


def test_group_metrics_do_not_confuse_any_and_all():
    groups = [
        {"group_id": "alternative", "any_of": ["a", "b"], "critical": False},
        {"group_id": "trend", "all_of": ["a", "b"], "critical": True},
    ]
    assert coverage(groups, {"a"}) == {
        "groupRecall": 0.5,
        "criticalMiss": True,
        "missingGroups": ["trend"],
    }
    assert coverage([], set())["criticalMiss"] is None
    with pytest.raises(ValueError):
        _satisfied({"any_of": ["a"], "all_of": ["b"]}, {"a", "b"})


def test_history_never_crosses_patient_or_includes_future_turn():
    cases = expand_cases(
        {
            "single_turn_cases": [],
            "conversations": [
                {
                    "conversation_id": p,
                    "patient_id": p,
                    "turns": [
                        {"question": p + " first"},
                        {"question": p + " next"},
                    ],
                }
                for p in ("p1", "p2")
            ],
        }
    )
    assert cases[0]["user_history"] == cases[2]["user_history"] == []
    assert cases[1]["user_history"] == ["p1 first"]
    assert cases[3]["user_history"] == ["p2 first"]


def test_every_completion_refuses_above_16k_even_on_32k_server():
    client = AuditedClient()
    seen = []

    def handle(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"count": 15000, "max_model_len": 32768})

    client._transport = httpx.MockTransport(handle)

    async def call():
        async for _ in client.complete([{"role": "user", "content": "synthetic"}]):
            pass

    with pytest.raises(Exception, match="context_overflow_eval_guard"):
        asyncio.run(call())
    assert seen == ["/tokenize"]


def test_tokenizer_failure_is_not_replaced_by_estimate():
    client = AuditedClient()
    client._transport = httpx.MockTransport(lambda _: httpx.Response(503))
    with pytest.raises(Exception, match="tokenizer_unavailable_eval_closed"):
        asyncio.run(client.count_tokens([{"role": "user", "content": "synthetic"}]))
