"""Small, explicitly synthetic CCO 0.7 contract fixtures."""

from copy import deepcopy


def synthetic_cco() -> dict:
    def fact(number, amount, date):
        return {"factId": f"00000000-0000-4000-8000-{number:012d}",
                "definition": "lab.creatinine.series|1.1.0", "evaluationKey": f"sample:{number}",
                "evaluationStatus": "success", "occurredFrom": date,
                "occurredTo": date, "recordedAt": date,
                "value": {"type": "quantity", "amount": amount, "unit": "mg/dL"},
                "comparisons": [], "evidence": [{"sourceType": "lab", "sourceId": str(number),
                                                  "role": "source", "occurredAt": date}]}

    old = fact(1, 2.1, "2021-01-01T00:00:00Z")
    latest = fact(2, 0.9, "2026-01-01T00:00:00Z")
    frozen = {key: deepcopy(old[key]) for key in
              ("factId", "definition", "evaluationKey", "evaluationStatus", "value")}
    frozen["value"]["amount"] = 2.2
    return {
        "schemaVersion": "0.7.0", "id": "00000000-0000-4000-8000-000000000100",
        "subjectId": "00000000-0000-4000-8000-000000000200", "purpose": "user-chat",
        "asOf": "2026-10-11T00:00:00Z", "generatedAt": "2026-10-11T00:00:01Z",
        "culture": "tr-TR", "consent": {"withheldFamilies": []},
        "sections": {"lab": {"facts": [old, latest], "insights": [{
            "insightId": "00000000-0000-4000-8000-000000000300",
            "definition": "lab.creatinine.review|1.0.0", "severity": "critical",
            "producedAt": "2026-02-01T00:00:00Z", "facts": [frozen],
        }], "missingData": []}},
        "contentReferences": [], "safety": {"requiresHumanReview": True,
                                              "disclaimerPolicy": "standard"},
        "limits": {"maxFactsPerSection": 100, "maxPayloadBytes": 200000, "truncated": False},
        "release": {"mode": "preview", "unreviewedDefinitions": ["lab.creatinine.review|1.0.0"]},
        "glossary": {
            "families": {"lab": {"label": "Laboratuvar", "description": "Tarihli ölçümler"}},
            "facts": {"lab.creatinine.series": {"label": "Kreatinin ölçümleri",
                                                 "description": "Numunelerin tarihli serisi"}},
            "rules": {"lab.creatinine.review": {"label": "Klinisyen incelemesi",
                                                 "description": "Mevcut kaynak yorumu"}},
        },
    }
