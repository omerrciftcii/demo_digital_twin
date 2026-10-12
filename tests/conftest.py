"""Keep the suite off a real encoder.

Retrieval tests assume lexical scores. A cached local encoder changes which
sections join the prompt and makes those tests depend on weights. Tests that
need a vector stub the loader themselves.
"""

import pytest


@pytest.fixture(autouse=True)
def _deterministic_encoder(monkeypatch):
    import medical_ai_selective_context.text_embed as embed

    monkeypatch.setattr(embed, "_load_model", lambda: None)
    embed._SECTION_CACHE.clear()
    embed._LINE_CACHE.clear()
    embed._PASSAGE_CACHE.clear()
    embed._QUERY_CACHE.clear()
