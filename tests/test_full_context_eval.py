from pathlib import Path

from scripts.run_full_context_eval import (
    EVAL_ROOT,
    _gold,
    _inside_eval_root,
    grade_errors,
    regrade,
    summarize,
)


def _row(key, *, called, passed=None, codes=(), first=None, final=None, critical_miss=False,
         retry=False, fallback=False):
    return {
        "key": key, "status": "answered" if called else "dry_run",
        "conversationId": None, "modelCalls": 1 if called else 0, "passed": passed,
        "attemptCodes": list(codes), "attemptClaims": [first] if first else [],
        "claims": final, "retry": retry, "retrySkipped": None, "fallback": fallback,
        "ttftMs": 100.0 if called else None, "totalMs": 900.0 if called else None,
        "promptTokenCheck": None, "context": {"ccoComplete": True},
        "gold": {"recall": 0.5, "hasCritical": True, "criticalMiss": critical_miss},
    }


def test_rates_use_only_rows_where_the_model_answered():
    rows = [
        _row("full:a", called=True, passed=True, codes=["value_not_in_evidence"],
             first={"total": 4, "rejected": 1}, final={"total": 3, "rejected": 0},
             critical_miss=True),
        _row("full:b", called=True, passed=False, codes=["date_mismatch"],
             first={"total": 2, "rejected": 1}, final={"total": 2, "rejected": 1},
             retry=True, fallback=True),
        _row("full:c", called=False, critical_miss=True),
    ]
    summary = summarize(rows, {"full:a": {"clinical": "dogru"}})
    verifier = summary["verifier"]
    assert verifier["modelCalledCount"] == 2
    assert verifier["passRate"] == 0.5
    assert verifier["unsupportedClaimRateFirst"] == round(2 / 6, 4)
    assert verifier["unsupportedClaimRateFinal"] == 0.2
    assert verifier["numericErrorRows"] == 1
    assert verifier["temporalErrorRows"] == 1
    assert summary["retry"] == {"count": 1, "successCount": 0, "skippedForContext": 0}
    assert summary["fallbackCount"] == 1
    assert summary["gold"]["criticalQuestionCount"] == 2
    assert summary["gold"]["criticalMissRate"] == 0.5
    assert summary["manualGrades"]["clinical"] == {"dogru": 1, "ungraded": 2}


def test_gold_uses_cited_fact_ids():
    groups = [
        {"group_id": "a", "role": "direct", "critical": True, "any_of": ["f1", "f2"]},
        {"group_id": "b", "role": "missingness", "critical": True, "all_of": ["f3", "f4"]},
    ]
    gold = _gold(groups, {"f2", "f3"})
    assert gold["recall"] == 0.5
    assert gold["criticalMiss"] is True


def test_regrade_uses_the_saved_rows_and_rejects_off_scale_grades():
    rows = [_row("full:a", called=True, passed=True), _row("full:b", called=True, passed=False)]
    for row in rows:
        row["arm"] = "full"
    payload = {"arms": ["full"], "rows": rows, "summary": {}}
    grades = {"full:a": {"clinical": "dogru", "criticalLoss": "var"}}
    assert grade_errors(grades, rows) == []
    summary = regrade(payload, grades)["summary"]["full"]["manualGrades"]
    assert summary["clinical"] == {"dogru": 1, "ungraded": 1}
    assert summary["criticalLoss"] == {"var": 1, "ungraded": 1}
    errors = grade_errors({"full:a": {"clinical": "D"}, "full:z": {}}, rows)
    assert errors == ["full:z: not in this run", "full:a.clinical: 'D' not in dogru/kismen/yanlis"]


def test_outputs_must_stay_under_data_evaluation():
    assert _inside_eval_root(EVAL_ROOT / "run")
    assert not _inside_eval_root(Path("docs") / "run")
