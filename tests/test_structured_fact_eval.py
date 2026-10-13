from scripts.run_structured_fact_eval import _summarize


def test_retry_success_rate_uses_only_retried_rows_as_denominator():
    rows = [
        {"metrics": {"retry": False}, "validation": {"passed": True}, "answer": "a"},
        {"metrics": {"retry": True}, "validation": {"passed": True}, "answer": "b"},
        {"metrics": {"retry": True}, "validation": {"passed": False}, "answer": "c"},
    ]
    summary = _summarize(rows)
    assert summary["retryCount"] == 2
    assert summary["retrySuccessCount"] == 1
    assert summary["retrySuccessRate"] == 0.5
