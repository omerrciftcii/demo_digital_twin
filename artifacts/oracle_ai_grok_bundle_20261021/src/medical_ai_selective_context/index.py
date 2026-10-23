"""Snapshot-isolated hybrid retrieval. No model download or remote embedding calls."""

import hashlib
import json
import re
import sqlite3
import threading
import unicodedata
from pathlib import Path
from typing import Any

from .store import canonical_json


def normalize_text(text: str) -> str:
    folded = text.casefold().replace("ı", "i")
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", folded)
        if not unicodedata.combining(character)
    )


# Search-language function words, not clinical terminology or family routing.
# Temporal words are consumed by the planner and remain in the original E5
# query; on their own they must not count as lexical clinical evidence.
LEXICAL_STOPWORDS = frozenset(
    normalize_text(word)
    for word in [
        "acaba",
        "ama",
        "artık",
        "asıl",
        "aynı",
        "bazı",
        "bazen",
        "bana",
        "beni",
        "benim",
        "ben",
        "bize",
        "bizi",
        "bizim",
        "biz",
        "bir",
        "birkaç",
        "biri",
        "bu",
        "buna",
        "bunu",
        "bunun",
        "burada",
        "bütün",
        "çok",
        "çünkü",
        "da",
        "daha",
        "de",
        "diye",
        "en",
        "gibi",
        "hangi",
        "hangisi",
        "hakkında",
        "hatta",
        "hem",
        "her",
        "hiç",
        "için",
        "ile",
        "ise",
        "işte",
        "kadar",
        "ki",
        "kim",
        "mı",
        "mi",
        "mu",
        "mü",
        "nasıl",
        "ne",
        "neden",
        "nedir",
        "nerede",
        "neye",
        "niçin",
        "o",
        "olan",
        "olarak",
        "olur",
        "olsun",
        "onun",
        "onu",
        "öyle",
        "sana",
        "seni",
        "senin",
        "sen",
        "siz",
        "sizin",
        "size",
        "sizi",
        "şey",
        "şu",
        "şunu",
        "ve",
        "veya",
        "ya",
        "yani",
        "yoksa",
        "sadece",
        "göre",
        "üzerine",
        "var",
        "mıydı",
        "midir",
        "mıdır",
        "mudur",
        "müdür",
        "son",
        "önce",
        "sonra",
        "şimdi",
        "şuanda",
        "bugün",
        "dün",
        "zaman",
        "zamanı",
        "zamanda",
        "tarih",
        "tarihi",
        "gün",
        "günler",
        "günlerde",
        "günün",
        "günün",
        "hafta",
        "haftalar",
        "haftalarda",
        "ay",
        "aylar",
        "aylarda",
        "yıl",
        "yıllar",
        "yıllarda",
        "geçen",
        "önceki",
        "sonraki",
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "is",
        "are",
        "am",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "for",
        "from",
        "in",
        "on",
        "at",
        "by",
        "with",
        "without",
        "as",
        "about",
        "into",
        "over",
        "under",
        "than",
        "then",
        "that",
        "this",
        "these",
        "those",
        "i",
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        "it",
        "its",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "where",
        "when",
        "can",
        "could",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "should",
        "may",
        "might",
        "all",
        "any",
        "some",
        "each",
        "every",
        "only",
        "just",
        "please",
        "tell",
        "explain",
        "show",
        "last",
        "latest",
        "recent",
        "recently",
        "now",
        "today",
        "yesterday",
        "tomorrow",
        "before",
        "after",
        "during",
        "since",
        "time",
        "times",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
    ]
)


def lexical_terms(query: str) -> list[str]:
    """Keep query content words; exact codes and semantic text are unaffected."""
    return list(
        dict.fromkeys(
            word
            for word in re.findall(r"\w+", normalize_text(query))
            if word not in LEXICAL_STOPWORDS
        )
    )


def token_chunks(
    text: str, tokenizer: Any, max_tokens: int = 512, prefix: str = "passage: ", overlap: int = 32
) -> list[str]:
    """Token-aware windows preserve the end of long narratives, never truncate.

    The prefix and special tokens share the same 512-token model budget. Final
    windows are re-tokenised as a guard against token decode/re-encode changes.
    Only search chunks are decoded; source evidence always remains untouched.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    overhead = len(tokenizer.encode(prefix, add_special_tokens=True, truncation=False))
    capacity = max_tokens - overhead - 8
    if capacity <= 0:
        raise ValueError("Embedding token budget is too small")
    if not tokens:
        return [prefix]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + capacity, len(tokens))
        while end > start:
            part = prefix + tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            if len(tokenizer.encode(part, add_special_tokens=True, truncation=False)) <= max_tokens:
                break
            end -= 1
        if end == start:
            raise ValueError("Cannot fit source token in embedding chunk")
        chunks.append(part)
        if end == len(tokens):
            break
        start = max(start + 1, end - min(overlap, max(0, end - start - 1)))
    return chunks


class EvidenceIndex:
    def __init__(
        self,
        db_path: Path,
        embedding_model: str = "intfloat/multilingual-e5-small",
        semantic_min_score: float | None = None,
    ):
        self.db_path = Path(db_path)
        self.embedding_model = embedding_model
        if semantic_min_score is not None and not -1 <= semantic_min_score <= 1:
            raise ValueError("semantic_min_score must be between -1 and 1")
        self.semantic_min_score = semantic_min_score
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS evidence_entries (
                ref TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, payload TEXT NOT NULL,
                definition_id TEXT NOT NULL, code TEXT NOT NULL, system TEXT NOT NULL,
                fingerprint TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS evidence_snapshot ON evidence_entries(snapshot_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
                ref UNINDEXED, snapshot_id UNINDEXED, text, tokenize='unicode61'
            );
            CREATE TABLE IF NOT EXISTS evidence_vectors (
                ref TEXT NOT NULL, snapshot_id TEXT NOT NULL, model TEXT NOT NULL,
                chunk INTEGER NOT NULL, vector BLOB NOT NULL,
                PRIMARY KEY(ref, model, chunk)
            );
            CREATE INDEX IF NOT EXISTS vectors_snapshot ON evidence_vectors(snapshot_id, model);
        """)
        self._db.commit()
        self._lock = threading.RLock()
        self._model: Any = None
        self._semantic_attempted = False
        self._semantic_state = "not_loaded"

    def _load_semantic(self) -> bool:
        if self._semantic_attempted:
            return self._model is not None
        self._semantic_attempted = True
        if not self.embedding_model:
            self._semantic_state = "disabled"
            return False
        try:
            from huggingface_hub import hf_hub_download
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._semantic_state = "optional_dependencies_missing"
            return False
        try:
            # Resolve the public model from cache only. No patient text is ever
            # passed to a network client and startup never downloads weights.
            # Resolve one required file rather than snapshot_download: newer
            # hub clients reject partial snapshots missing optional exports.
            config = hf_hub_download(
                repo_id=self.embedding_model, filename="config.json", local_files_only=True
            )
            cached = str(Path(config).parent)
            self._model = SentenceTransformer(
                cached, device="cpu", local_files_only=True, trust_remote_code=False
            )
            self._model.max_seq_length = 512
            self._semantic_state = "available"
            return True
        except (OSError, RuntimeError, ValueError, ImportError):
            self._model = None
            self._semantic_state = "local_model_unavailable"
            return False

    def _persist_vectors(self, snapshot_id: str, entries: list[dict[str, Any]]) -> None:
        if self._model is None or not entries:
            return
        chunks = []
        owners = []
        for entry in entries:
            entry_chunks = token_chunks(entry["text"], self._model.tokenizer)
            chunks.extend(entry_chunks)
            owners.extend((entry["ref"], number) for number in range(len(entry_chunks)))
        vectors = self._model.encode(
            chunks,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if len(vectors) != len(owners):
            raise ValueError("Embedding batch size mismatch")
        for (ref, number), vector in zip(owners, vectors, strict=True):
            self._db.execute(
                "INSERT OR REPLACE INTO evidence_vectors VALUES (?, ?, ?, ?, ?)",
                (
                    ref,
                    snapshot_id,
                    self.embedding_model,
                    number,
                    vector.astype("float32").tobytes(),
                ),
            )

    def index_snapshot(self, snapshot_id: str, entries: list[dict[str, Any]]) -> None:
        with self._lock:
            if any(entry.get("snapshotId") != snapshot_id for entry in entries):
                raise ValueError("Evidence snapshot mismatch")
            if len({entry["ref"] for entry in entries}) != len(entries):
                raise ValueError("Duplicate evidence reference")
            for entry in entries:
                owner = self._db.execute(
                    "SELECT snapshot_id FROM evidence_entries WHERE ref=?", (entry["ref"],)
                ).fetchone()
                if owner is not None and owner["snapshot_id"] != snapshot_id:
                    raise ValueError("Evidence reference belongs to another snapshot")
            semantic = self._load_semantic()
            existing = {
                row["ref"]: row["fingerprint"]
                for row in self._db.execute(
                    "SELECT ref, fingerprint FROM evidence_entries WHERE snapshot_id=?",
                    (snapshot_id,),
                )
            }
            current_refs = {entry["ref"] for entry in entries}
            pending_vectors = []
            with self._db:
                for stale in set(existing) - current_refs:
                    self._db.execute("DELETE FROM evidence_entries WHERE ref=?", (stale,))
                    self._db.execute("DELETE FROM evidence_fts WHERE ref=?", (stale,))
                    self._db.execute("DELETE FROM evidence_vectors WHERE ref=?", (stale,))
                for entry in entries:
                    payload = canonical_json(entry)
                    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
                    changed = existing.get(entry["ref"]) != fingerprint
                    if changed:
                        value = entry.get("value") or {}
                        self._db.execute(
                            "INSERT OR REPLACE INTO evidence_entries VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                entry["ref"],
                                snapshot_id,
                                payload,
                                normalize_text(entry["definitionId"]),
                                normalize_text(str(value.get("code", ""))),
                                normalize_text(str(value.get("system", ""))),
                                fingerprint,
                            ),
                        )
                        self._db.execute("DELETE FROM evidence_fts WHERE ref=?", (entry["ref"],))
                        self._db.execute(
                            "INSERT INTO evidence_fts VALUES (?, ?, ?)",
                            (entry["ref"], snapshot_id, normalize_text(entry["text"])),
                        )
                        self._db.execute(
                            "DELETE FROM evidence_vectors WHERE ref=?", (entry["ref"],)
                        )
                    vector_exists = self._db.execute(
                        "SELECT 1 FROM evidence_vectors WHERE ref=? AND model=? LIMIT 1",
                        (entry["ref"], self.embedding_model),
                    ).fetchone()
                    if semantic and not vector_exists:
                        pending_vectors.append(entry)
                if pending_vectors:
                    try:
                        self._persist_vectors(snapshot_id, pending_vectors)
                    except (OSError, RuntimeError, ValueError, TypeError, ImportError):
                        # Lexical/exact remain usable and this state is
                        # surfaced; never pretend an embedding was made.
                        self._semantic_state = "encoding_failed"
                        for entry in pending_vectors:
                            self._db.execute(
                                "DELETE FROM evidence_vectors WHERE ref=?", (entry["ref"],)
                            )

    def entries(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                json.loads(row["payload"])
                for row in self._db.execute(
                    "SELECT payload FROM evidence_entries WHERE snapshot_id=? ORDER BY rowid",
                    (snapshot_id,),
                )
            ]

    def search(self, snapshot_id: str, queries: list[str], limit: int = 60) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        with self._lock:
            results: dict[str, dict[str, Any]] = {}

            def add(
                ranking: list[str], channel: str, raw_scores: dict[str, float] | None = None
            ) -> None:
                for rank, ref in enumerate(dict.fromkeys(ranking), 1):
                    item = results.setdefault(ref, {"ref": ref, "score": 0.0, "channels": []})
                    item["score"] += 1.0 / (60 + rank)
                    if channel not in item["channels"]:
                        item["channels"].append(channel)
                    if raw_scores is not None:
                        field = "semanticScore" if channel == "semantic" else "lexicalScore"
                        score = raw_scores[ref]
                        previous = item.get(field)
                        if previous is None:
                            item[field] = score
                        else:
                            item[field] = (
                                max(previous, score)
                                if channel == "semantic"
                                else min(previous, score)
                            )

            # Normalisation is lexical only; retain Turkish orthography for E5.
            unique_queries = {normalize_text(q.strip()): q.strip() for q in queries if q.strip()}
            for query, original_query in unique_queries.items():
                tokens = re.findall(r"[\w.-]+", query, flags=re.UNICODE)
                exact_candidates = list(dict.fromkeys([query, *tokens]))
                if exact_candidates:
                    placeholders = ",".join("?" for _ in exact_candidates)
                    rows = self._db.execute(
                        f"SELECT ref FROM evidence_entries WHERE snapshot_id=? AND "
                        f"(definition_id IN ({placeholders}) OR code IN ({placeholders})) "
                        "ORDER BY ref LIMIT ?",
                        [snapshot_id, *exact_candidates, *exact_candidates, limit],
                    ).fetchall()
                    add([row["ref"] for row in rows], "exact")
                words = lexical_terms(query)
                if words:
                    # All user terms are quoted; FTS operators/SQL are not executable.
                    match = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
                    rows = self._db.execute(
                        "SELECT ref, bm25(evidence_fts) AS lexical_score FROM evidence_fts "
                        "WHERE evidence_fts MATCH ? "
                        "AND snapshot_id=? ORDER BY bm25(evidence_fts), ref LIMIT ?",
                        (match, snapshot_id, limit),
                    ).fetchall()
                    add(
                        [row["ref"] for row in rows],
                        "lexical",
                        {row["ref"]: row["lexical_score"] for row in rows},
                    )
                if self._load_semantic():
                    try:
                        semantic = self._dense_search(snapshot_id, original_query, limit)
                        add([ref for ref, _ in semantic], "semantic", dict(semantic))
                    except (OSError, RuntimeError, ValueError, TypeError, ImportError):
                        self._semantic_state = "query_encoding_failed"
            return sorted(results.values(), key=lambda item: (-item["score"], item["ref"]))[:limit]

    def _dense_search(self, snapshot_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        import numpy as np

        rows = self._db.execute(
            "SELECT ref, vector FROM evidence_vectors WHERE snapshot_id=? AND model=?",
            (snapshot_id, self.embedding_model),
        ).fetchall()
        if not rows:
            return []
        queries = token_chunks(query, self._model.tokenizer, prefix="query: ")
        vectors = self._model.encode(
            queries, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        scores: dict[str, float] = {}
        for row in rows:
            embedding = np.frombuffer(row["vector"], dtype=np.float32)
            score = float(np.max(vectors @ embedding))
            scores[row["ref"]] = max(scores.get(row["ref"], float("-inf")), score)
        ranked = sorted(scores, key=lambda ref: (-scores[ref], ref))
        return [
            (ref, scores[ref])
            for ref in ranked
            if self.semantic_min_score is None or scores[ref] >= self.semantic_min_score
        ][:limit]

    def status(self) -> dict[str, Any]:
        with self._lock:
            count = self._db.execute("SELECT count(*) FROM evidence_entries").fetchone()[0]
            vectors = self._db.execute(
                "SELECT count(*) FROM evidence_vectors WHERE model=?", (self.embedding_model,)
            ).fetchone()[0]
            return {
                "status": "ready" if self._semantic_state == "available" else "degraded",
                "lexical": "fts5",
                "exact": True,
                "lexicalPolicy": "unicode_normalized_tr_en_stopwords_v1; no clinical keyword map",
                "semantic": self._semantic_state,
                "embeddingModel": self.embedding_model,
                "semanticMinScore": self.semantic_min_score,
                "scoreMeaning": "RRF ranking; semantic cosine and lexical BM25 are not confidence",
                "modelDownloadEnabled": False,
                "entryCount": count,
                "vectorChunkCount": vectors,
            }

    def close(self) -> None:
        with self._lock:
            self._db.close()
