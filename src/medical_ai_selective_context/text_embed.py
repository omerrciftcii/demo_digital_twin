"""Local section embeddings. No download and no outbound patient text.

The default model is a small multilingual encoder. Weights are read from the
local Hugging Face cache only. If the library or the weights are missing,
callers get an empty score map and stay on the lexical index.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .text_document import TextRecord

_CACHE_RECORDS = 32
_LOCK = threading.Lock()
_MODEL: Any = None
_MODEL_NAME = ""
_ATTEMPTED = False
_SECTION_CACHE: OrderedDict[tuple[str, str], Any] = OrderedDict()
_QUERY_CACHE: OrderedDict[tuple[str, str], Any] = OrderedDict()


def embedding_model_name() -> str:
    """TEXT_EMBEDDING_MODEL selects the local encoder. Empty disables it."""
    disabled = os.getenv("TEXT_EMBEDDING_DISABLE", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return ""
    return os.getenv("TEXT_EMBEDDING_MODEL", "intfloat/multilingual-e5-small").strip()


def _device() -> str:
    choice = os.getenv("TEXT_EMBEDDING_DEVICE", "auto").strip().lower()
    if choice in {"cpu", "cuda"}:
        return choice
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model() -> Any:
    global _MODEL, _MODEL_NAME, _ATTEMPTED
    name = embedding_model_name()
    if _ATTEMPTED and _MODEL_NAME == name:
        return _MODEL
    with _LOCK:
        if _ATTEMPTED and _MODEL_NAME == name:
            return _MODEL
        _ATTEMPTED = True
        _MODEL_NAME = name
        _MODEL = None
        _SECTION_CACHE.clear()
        _QUERY_CACHE.clear()
        if not name:
            return None
        try:
            from huggingface_hub import hf_hub_download
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        try:
            config = hf_hub_download(repo_id=name, filename="config.json", local_files_only=True)
            cached = str(Path(config).parent)
            model = SentenceTransformer(
                cached, device=_device(), local_files_only=True, trust_remote_code=False,
            )
            model.max_seq_length = 256
            _MODEL = model
        except (OSError, RuntimeError, ValueError, ImportError):
            _MODEL = None
        return _MODEL


def _remember(cache: OrderedDict, key: tuple[str, str], value: Any) -> Any:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_RECORDS:
        cache.popitem(last=False)
    return value


def _encode(model: Any, texts: list[str]) -> Any:
    return model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def section_matrix(record: TextRecord) -> Any:
    """One normalized vector per section, cached for the record text."""
    model = _load_model()
    name = embedding_model_name()
    if model is None or not record.sections:
        return None
    key = (record.content_hash, name)
    with _LOCK:
        cached = _SECTION_CACHE.get(key)
        if cached is not None:
            _SECTION_CACHE.move_to_end(key)
            return cached
    texts = [f"passage: {section.text[:4000]}" for section in record.sections]
    matrix = _encode(model, texts)
    with _LOCK:
        return _remember(_SECTION_CACHE, key, matrix)


def query_vector(question: str) -> Any:
    model = _load_model()
    name = embedding_model_name()
    if model is None or not question.strip():
        return None
    key = (question.strip(), name)
    with _LOCK:
        cached = _QUERY_CACHE.get(key)
        if cached is not None:
            _QUERY_CACHE.move_to_end(key)
            return cached
    vector = _encode(model, [f"query: {question.strip()[:2000]}"])[0]
    with _LOCK:
        return _remember(_QUERY_CACHE, key, vector)


def semantic_section_scores(record: TextRecord, question: str) -> dict[int, float]:
    """Cosine of each section with the question. Empty when no local model."""
    matrix = section_matrix(record)
    vector = query_vector(question)
    if matrix is None or vector is None:
        return {}
    scores = matrix @ vector
    return {index: float(score) for index, score in enumerate(scores)}
