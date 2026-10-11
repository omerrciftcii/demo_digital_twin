import json
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from evidence_support import synthetic_cco

from medical_ai_selective_context.evidence import build_evidence
from medical_ai_selective_context.index import (
    EvidenceIndex,
    lexical_terms,
    normalize_text,
    token_chunks,
)


def test_hybrid_exact_lexical_and_snapshot_isolation(tmp_path: Path) -> None:
    index = EvidenceIndex(tmp_path / "index.sqlite3", embedding_model="")
    document = synthetic_cco()
    first = build_evidence(document, "a")
    second = build_evidence(document, "b")
    index.index_snapshot("a", first)
    index.index_snapshot("b", second)
    results = index.search("a", ["KREATİNİN", "lab.creatinine.series"])
    assert results
    assert {result["ref"] for result in results} <= {entry["ref"] for entry in first}
    assert {channel for result in results for channel in result["channels"]} >= {"exact", "lexical"}
    assert all("lexicalScore" in result for result in results if "lexical" in result["channels"])
    assert index.search("not-there", ["kreatinin"]) == []
    assert index.search("a", ["xxxxnonexistent"]) == []
    assert index.entries("a") == first
    status = index.status()
    assert status["semantic"] == "disabled"
    assert status["entryCount"] == 8
    assert status["modelDownloadEnabled"] is False
    index.index_snapshot("a", first)
    assert index.status()["entryCount"] == 8
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(lambda _: bool(index.search("a", ["kreatinin"])), range(8)))
    index.close()


def test_code_search_not_a_disease_keyword_map(tmp_path: Path) -> None:
    index = EvidenceIndex(tmp_path / "index.sqlite3", embedding_model="")
    document = synthetic_cco()
    document["sections"]["lab"]["facts"][0]["value"] = {
        "type": "coded", "system": "test", "code": "QX-123.9",
        "display": "Synthetic coded item", "unit": "code", "attributes": {},
    }
    entries = build_evidence(document, "a")
    index.index_snapshot("a", entries)
    result = index.search("a", ["QX-123.9"])[0]
    assert result["ref"] == entries[0]["ref"]
    assert "exact" in result["channels"]
    assert index.search("a", ['" OR "* --']) == []
    index.close()


def test_stopword_only_query_does_not_create_lexical_hits(tmp_path: Path) -> None:
    index = EvidenceIndex(tmp_path / "index.sqlite3", embedding_model="")
    document = synthetic_cco()
    entry = build_evidence(document, "a")[0]
    entry["text"] += " son gunlerde ne nasil benim last what about my "
    index.index_snapshot("a", [entry])
    assert index.search("a", ["Son günlerde ne nasıl benim? last what about my"]) == []
    assert lexical_terms("Son kreatinin ve HbA1c sonuçları?") == ["kreatinin", "hba1c", "sonuclari"]
    assert "stopwords_v1" in index.status()["lexicalPolicy"]
    index.close()


def test_common_temporal_words_do_not_turn_unrelated_or_implicit_query_into_lexical_hit(
    tmp_path: Path, monkeypatch,
) -> None:
    sample = Path(__file__).parents[1] / "data" / "synthetic" / "cco-complete.json"
    document = json.loads(sample.read_text(encoding="utf-8"))
    entries = build_evidence(document, "a")
    index = EvidenceIndex(tmp_path / "index.sqlite3", embedding_model="")
    index.index_snapshot("a", entries)
    implicit = "Son günlerde geceleri yeterince dinlenebiliyor muyum?"
    unrelated = "Mars gezegeninin yüzölçümü nedir son günlerde?"
    assert index.search("a", [implicit]) == []
    assert index.search("a", [unrelated]) == []
    assert index.search("a", ["Son kreatinin sonuçlarım nasıl?"])

    # Semantic candidate generation still receives the untouched original
    # sentence, including temporal context; no synonym/clinical list is used.
    sleep = next(entry for entry in entries if entry["section"] == "sleep")
    queried = []

    def semantic(snapshot_id, query, limit):
        queried.append(query)
        return [(sleep["ref"], 0.84)]

    monkeypatch.setattr(index, "_load_semantic", lambda: True)
    monkeypatch.setattr(index, "_dense_search", semantic)
    result = index.search("a", [implicit])[0]
    assert queried == [implicit]
    assert result["channels"] == ["semantic"]
    index.close()


def test_rejects_cross_snapshot_index_batch(tmp_path: Path) -> None:
    index = EvidenceIndex(tmp_path / "index.sqlite3", embedding_model="")
    with pytest.raises(ValueError, match="snapshot mismatch"):
        index.index_snapshot("b", build_evidence(synthetic_cco(), "a"))
    assert index.status()["entryCount"] == 0
    index.close()


def test_ref_collision_cannot_replace_another_snapshots_evidence(tmp_path: Path) -> None:
    index = EvidenceIndex(tmp_path / "index.sqlite3", embedding_model="")
    entries = build_evidence(synthetic_cco(), "a")
    index.index_snapshot("a", entries)
    attempted = [{**entry, "snapshotId": "b"} for entry in entries]
    with pytest.raises(ValueError, match="another snapshot"):
        index.index_snapshot("b", attempted)
    assert index.entries("a") == entries
    assert index.entries("b") == []
    index.close()


def test_uncached_model_never_triggers_network_download(tmp_path: Path, monkeypatch) -> None:
    called = []

    def cache_only(**kwargs):
        called.append(kwargs)
        raise OSError("not in local cache")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=cache_only))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=None))
    index = EvidenceIndex(tmp_path / "index.sqlite3")
    index.index_snapshot("a", build_evidence(synthetic_cco(), "a"))
    assert called == [{"repo_id": "intfloat/multilingual-e5-small", "filename": "config.json",
                       "local_files_only": True}]
    assert index.status()["semantic"] == "local_model_unavailable"
    assert index.search("a", ["Kreatinin"])
    index.close()


def test_partial_local_model_snapshot_loads_and_batches_evidence(tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    configured = []
    encoded = []
    expected_dir = tmp_path / "public-cache" / "snapshots" / "revision"

    def local_config(**kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["filename"] == "config.json"
        return str(expected_dir / "config.json")

    class LocalModel:
        tokenizer = CharacterTokenizer()

        def __init__(self, path, **kwargs):
            configured.append((path, kwargs))

        def encode(self, chunks, **kwargs):
            encoded.append(chunks)
            return np.ones((len(chunks), 3), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=local_config))
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=LocalModel))
    index = EvidenceIndex(tmp_path / "index.sqlite3")
    entries = build_evidence(synthetic_cco(), "a")
    index.index_snapshot("a", entries)
    assert configured == [(str(expected_dir), {"device": "cpu", "local_files_only": True,
                                               "trust_remote_code": False})]
    assert index.status()["semantic"] == "available"
    assert len(encoded) == 1
    assert index.status()["vectorChunkCount"] >= len(entries)
    index.index_snapshot("a", entries)
    assert len(encoded) == 1
    index.close()


def test_semantic_raw_cosine_and_original_turkish_query_exposed(tmp_path: Path, monkeypatch) -> None:
    index = EvidenceIndex(tmp_path / "index.sqlite3", embedding_model="")
    entry = build_evidence(synthetic_cco(), "a")[0]
    index.index_snapshot("a", [entry])
    queried = []

    def search(snapshot_id, query, limit):
        queried.append(query)
        return [(entry["ref"], 0.8123)]

    monkeypatch.setattr(index, "_load_semantic", lambda: True)
    monkeypatch.setattr(index, "_dense_search", search)
    results = index.search("a", ["İlgili ÖLÇÜM"])
    assert queried == ["İlgili ÖLÇÜM"]
    assert results[0]["semanticScore"] == 0.8123
    assert results[0]["score"] != 0.8123
    assert "semantic" in results[0]["channels"]
    index.close()


def test_explicit_download_requests_only_one_required_weight_format(monkeypatch) -> None:
    downloaded = []

    def public_files(**kwargs):
        return ["model.safetensors", "pytorch_model.bin", "onnx/model.onnx",
                "openvino/model.bin", "config.json", "tokenizer.json"]

    def download(**kwargs):
        downloaded.append(kwargs)
        return "/synthetic-model-cache"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        list_repo_files=public_files, snapshot_download=download,
    ))
    script = Path(__file__).parents[1] / "scripts" / "prepare_embeddings.py"
    monkeypatch.setattr(sys, "argv", [str(script)])
    runpy.run_path(str(script), run_name="__main__")
    patterns = downloaded[0]["allow_patterns"]
    assert "model.safetensors" in patterns
    assert "pytorch_model.bin" not in patterns
    assert not any("onnx" in pattern or "openvino" in pattern for pattern in patterns)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True, truncation=False):
        assert truncation is False
        ids = [ord(character) for character in text]
        return [0, *ids, 0] if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value) for value in ids if value)


def test_long_embedding_windows_preserve_tail_and_respect_token_limit() -> None:
    tokenizer = CharacterTokenizer()
    text = "başlangıç" + "x" * 1800 + "SON-KANIT"
    chunks = token_chunks(text, tokenizer, overlap=0)
    assert len(chunks) > 1
    assert "".join(chunk.removeprefix("passage: ") for chunk in chunks) == text
    assert all(len(tokenizer.encode(chunk)) <= 512 for chunk in chunks)
    assert chunks[-1].endswith("SON-KANIT")
    assert normalize_text("İLAÇ IŞIK") == "ilac isik"
