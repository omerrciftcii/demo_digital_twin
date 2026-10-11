import asyncio
import json

import pytest

from medical_ai_selective_context.graph import Neo4jTerminologyGraph, TerminologyGraphUnavailable
from medical_ai_selective_context.terminology import catalog_from_glossaries, normalize


def _catalog():
    return catalog_from_glossaries([{
        "subjectId": "must-never-reach-graph",
        "sections": {"lab": {"facts": [{"factId": "never", "value": 999}]}},
        "glossary": {
            "families": {"lab": {"label": "Laboratuvar", "description": "Kaynak tahliller."}},
            "facts": {"lab.creatinine.latest": {"label": "Kreatinin", "description": "Son değer."}},
            "rules": {},
        },
    }])


class FakeDriver:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.closed = False

    async def verify_connectivity(self):
        if self.fail:
            raise OSError("connection unavailable")

    async def execute_query(self, query, **params):
        self.calls.append((query, params))
        return [], None, None

    async def close(self):
        self.closed = True


def test_catalog_uses_glossary_only_and_contains_no_disease_alias_pack():
    catalog = _catalog()
    serialized = json.dumps(catalog)
    assert "must-never-reach-graph" not in serialized
    assert "factId" not in serialized
    assert "999" not in serialized
    assert catalog["definitions"][0]["family"] == "lab"
    assert set(catalog) == {"key", "families", "definitions", "rules"}
    assert catalog == _catalog()


def test_conflicting_trusted_metadata_requires_review():
    one = {"glossary": {"families": {"lab": {"label": "Bir", "description": "Bir."}}}}
    two = {"glossary": {"families": {"lab": {"label": "İki", "description": "İki."}}}}
    with pytest.raises(ValueError, match="conflicting"):
        catalog_from_glossaries([one, two])


def test_initialization_writes_allowlisted_metadata_and_no_patient_fields():
    async def scenario():
        graph = Neo4jTerminologyGraph()
        driver = FakeDriver()
        graph._driver = driver
        catalog = _catalog()
        catalog["definitions"][0]["subjectId"] = "discard-extra-key"
        status = await graph.initialize(catalog)
        assert status["status"] == "available"
        assert status["metadataCount"] == 2
        writes = json.dumps(driver.calls)
        assert "discard-extra-key" not in writes
        assert "TerminologyConcept" not in writes
        assert "CcoMetadata" in writes
        assert "query" not in driver.calls[2][1]
        await graph.close()
        assert driver.closed
    asyncio.run(scenario())


def test_outage_returns_degraded_status_and_resolve_fails_safely():
    async def scenario():
        graph = Neo4jTerminologyGraph()
        graph._driver = FakeDriver(fail=True)
        status = await graph.initialize(_catalog())
        assert status["status"] == "unavailable"
        assert status["lastError"] == "OSError"
        with pytest.raises(TerminologyGraphUnavailable):
            await graph.resolve("Kreatinin")
    asyncio.run(scenario())


def test_query_is_read_only_parameterized_and_not_stored():
    async def scenario():
        graph = Neo4jTerminologyGraph()
        driver = FakeDriver()
        graph._driver = driver
        await graph.initialize(_catalog())
        before = len(driver.calls)
        await graph.resolve("Kreatinin kişisel soru")
        assert len(driver.calls) == before + 1
        query, params = driver.calls[-1]
        assert "SET " not in query and "MERGE " not in query
        assert "kişisel" not in query
        assert params["catalog"] == graph.active_release
        assert "$search" in query
    asyncio.run(scenario())


def test_normalization_is_language_generic_not_disease_rules():
    assert normalize("İŞLEM, çĞıöşü") == "islem cgiosu"


def test_graph_cannot_reintroduce_stopword_only_lexical_matches():
    async def scenario():
        graph = Neo4jTerminologyGraph()
        driver = FakeDriver()
        graph._driver = driver
        await graph.initialize(_catalog())
        before = len(driver.calls)
        assert await graph.resolve("Son günlerde benim için nasıl") == []
        assert len(driver.calls) == before
        await graph.resolve("Son kreatinin değeri")
        assert "son" not in driver.calls[-1][1]["terms"]
        assert "kreatinin" in driver.calls[-1][1]["terms"]
    asyncio.run(scenario())
