from scripts.run_patient_centric_retrieval_eval import _aggregate_slice, _recall


def _row(*, critical: bool, miss: bool, recall: float | None) -> dict:
    metrics = {
        str(k): {
            "recall": recall,
            "criticalMiss": miss,
            "directRecall": None,
            "indirectRecall": None,
            "medicationRecall": None,
            "missingnessRecall": None,
        }
        for k in (1, 3, 5, 10, 20)
    }
    return {
        "hasCriticalGroups": critical,
        "coldStart": False,
        "latencyMs": 1.0,
        "metrics": metrics,
    }


def test_empty_gold_is_unscored() -> None:
    assert _recall([], {"fact"}) is None


def test_critical_miss_rate_uses_only_questions_with_critical_groups() -> None:
    """Questions with no critical group must not dilute the miss rate.

    Counting them as successes turned 23/25 into 23/33 in the first local run.
    """
    rows = [
        _row(critical=True, miss=True, recall=0.0),
        _row(critical=True, miss=False, recall=1.0),
        _row(critical=False, miss=False, recall=None),
    ]
    slot = _aggregate_slice(rows)["recallAtK"]["10"]
    assert slot["criticalEvidenceQuestionCount"] == 2
    assert slot["criticalEvidenceMissRate"] == 0.5
    assert slot["macroRecall"] == 0.5
