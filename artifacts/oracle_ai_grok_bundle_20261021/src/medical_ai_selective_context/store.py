"""Strict CCO 0.7 ingestion and immutable, content-addressed local snapshots."""

import hashlib
import json
import re
import sqlite3
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

SUPPORTED_SCHEMA_VERSIONS = {"0.7.0"}
SCHEMA_PATH = Path(__file__).parent / "schema" / "cco-0.7.schema.json"
_VALIDATOR = Draft202012Validator(
    json.loads(SCHEMA_PATH.read_text(encoding="utf-8")), format_checker=FormatChecker()
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON property is not allowed")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_keys)


def unwrap_and_validate_cco(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept a CCO or CcoDto envelope without repairing the original document.

    Validation errors expose paths and constraint names, not clinical values.
    Metadata from an envelope is deliberately never copied into a document.
    """
    if not isinstance(payload, dict):
        raise ValueError("CCO JSON must be an object")
    candidate = payload
    for key in ("document", "Document"):
        if key in payload:
            candidate = payload[key]
            break
    if not isinstance(candidate, dict):
        raise ValueError("CCO envelope document must be an object")
    if candidate.get("schemaVersion") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("Unsupported CCO schemaVersion; only 0.7.0 is accepted")
    try:
        canonical_json(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError("CCO must contain finite, JSON-compatible values") from exc
    error = next(_VALIDATOR.iter_errors(candidate), None)
    if error is not None:
        pointer = "/" + "/".join(str(item).replace("~", "~0").replace("/", "~1")
                                  for item in error.absolute_path)
        raise ValueError(f"Invalid CCO at {pointer}: {error.validator} constraint failed")
    return deepcopy(candidate)


def fact_count(document: dict[str, Any]) -> int:
    return sum(len(section["facts"]) for section in document["sections"].values())


class CcoStore:
    def __init__(self, data_dir: Path, db_path: Path | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.archive_dir = self.data_dir / "snapshots"
        self.db_path = Path(db_path) if db_path else self.data_dir.parent / "runtime" / (
            self.data_dir.name + "-snapshots.sqlite3"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS snapshots (snapshot_id TEXT PRIMARY KEY, "
            "content_hash TEXT NOT NULL, source_file TEXT NOT NULL, "
            "data_class TEXT NOT NULL CHECK(data_class IN ('real','synthetic')))"
        )
        self._db.commit()
        self._lock = threading.RLock()
        self._documents: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _file_name(file_name: str) -> str:
        if (not file_name.lower().endswith(".json") or len(file_name) > 180
                or re.search(r'[\\/:*?"<>|\x00-\x1f]', file_name)
                or file_name in {".json", "..json"}
                or Path(file_name).name != file_name):
            raise ValueError("fileName must be a plain .json file name")
        if file_name.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *[f"COM{x}" for x in range(1, 10)],
            *[f"LPT{x}" for x in range(1, 10)],
        }:
            raise ValueError("fileName is reserved")
        return file_name

    @staticmethod
    def _write_once(path: Path, document: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as output:
                output.write(canonical_json(document) + "\n")
        except FileExistsError:
            try:
                existing = unwrap_and_validate_cco(_read_json(path))
            except (ValueError, OSError) as exc:
                raise ValueError("Existing snapshot file is invalid; refusing overwrite") from exc
            if content_hash(existing) != content_hash(document):
                raise ValueError("File name already contains a different snapshot; refusing overwrite")

    def _register(self, document: dict[str, Any], source_file: str,
                  data_class: str | None = None) -> dict[str, Any]:
        digest = content_hash(document)
        snapshot_id = "cco_" + digest
        previous = self._db.execute(
            "SELECT source_file, data_class FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        # Only trusted caller code can mark synthetic. Real content may never
        # become synthetic later; explicit real reimports revoke eligibility.
        if previous:
            source_file = previous[0]
            classification = "real" if data_class == "real" else previous[1]
        else:
            classification = data_class or "real"
        self._write_once(self.archive_dir / (snapshot_id + ".json"), document)
        self._db.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?) ON CONFLICT(snapshot_id) "
            "DO UPDATE SET data_class=excluded.data_class",
            (snapshot_id, digest, source_file, classification),
        )
        self._db.commit()
        self._documents[snapshot_id] = deepcopy(document)
        self._metadata[snapshot_id] = {
            "snapshotId": snapshot_id, "subjectId": document["subjectId"],
            "asOf": document["asOf"], "schemaVersion": document["schemaVersion"],
            "purpose": document["purpose"], "sourceFile": source_file,
            "dataClass": classification, "contentHash": digest,
            "factCount": fact_count(document),
        }
        return self.summary(snapshot_id)

    def reload(self) -> dict[str, Any]:
        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._documents.clear()
            self._metadata.clear()
            rejected = []
            # Archive copies survive manual source-file replacement or deletion.
            paths = sorted(self.data_dir.glob("*.json")) + sorted(self.archive_dir.glob("*.json"))
            for path in paths:
                try:
                    document = unwrap_and_validate_cco(_read_json(path))
                    self._register(document, path.name)
                except (OSError, ValueError, sqlite3.Error) as exc:
                    reason = str(exc) if isinstance(exc, ValueError) else "Source could not be read"
                    rejected.append({"file": path.name, "reason": reason})
            return {"loaded": self.count, "rejected": rejected}

    def import_document(self, payload: dict[str, Any], file_name: str | None = None,
                        data_class: str = "real") -> dict[str, Any]:
        if data_class not in {"real", "synthetic"}:
            raise ValueError("data_class must be real or synthetic")
        document = unwrap_and_validate_cco(payload)
        digest = content_hash(document)
        snapshot_id = "cco_" + digest
        target_name = self._file_name(file_name) if file_name else snapshot_id + ".json"
        with self._lock:
            if file_name:
                self._write_once(self.data_dir / target_name, document)
            return self._register(document, target_name, data_class)

    def get(self, snapshot_id: str) -> dict[str, Any] | None:
        """Return a deep copy of the raw, schema-compliant CCO; never merge."""
        with self._lock:
            document = self._documents.get(snapshot_id)
            return deepcopy(document) if document is not None else None

    def summary(self, snapshot_id: str) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._metadata[snapshot_id])

    def list_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.summary(key) for key in sorted(self._metadata)]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._documents)

    def close(self) -> None:
        with self._lock:
            self._db.close()
