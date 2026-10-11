"""Generic normalization and trusted patient-free metadata, not clinical rules.

Callers must only supply vetted glossaries. Filtering fields is not a
de-identification procedure for arbitrary free text.
"""

import hashlib
import json
import re
import unicodedata
from typing import Any


def normalize(value: str) -> str:
    translated = value.casefold().translate(str.maketrans("çğıöşü", "cgiosu"))
    decomposed = unicodedata.normalize("NFKD", translated)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9.-]+", " ", plain).strip()


def definition_base(value: Any) -> str:
    return value.split("|", 1)[0] if isinstance(value, str) else ""


def catalog_from_glossaries(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract IDs/text only, never fact instances or subjects.

    Family membership comes only from exact canonical ID prefixes, not guessed
    clinical relationships. Conflicting metadata requires review.
    """
    groups: dict[str, dict[str, dict[str, str]]] = {
        "families": {}, "definitions": {}, "rules": {}
    }
    for document in documents:
        glossary = document.get("glossary", {})
        if not isinstance(glossary, dict):
            raise ValueError("glossary must be an object")
        for source, destination in (
            ("families", "families"), ("facts", "definitions"), ("rules", "rules")
        ):
            rows = glossary.get(source, {})
            if not isinstance(rows, dict):
                raise ValueError("glossary groups must be objects")
            for key, value in rows.items():
                if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9.-]{0,199}", key):
                    raise ValueError("invalid glossary identifier")
                if not isinstance(value, dict):
                    raise ValueError("glossary entries must be objects")
                label, description = value.get("label"), value.get("description")
                if not isinstance(label, str) or not 1 <= len(label) <= 120:
                    raise ValueError("invalid glossary label")
                if not isinstance(description, str) or not 1 <= len(description) <= 600:
                    raise ValueError("invalid glossary description")
                row = {"id": key, "label": label, "description": description}
                previous = groups[destination].get(key)
                if previous is not None and previous != row:
                    raise ValueError(f"conflicting trusted glossary metadata: {key}")
                groups[destination][key] = row
    for group in ("definitions", "rules"):
        for key, row in groups[group].items():
            families = [family for family in groups["families"] if key.startswith(family + ".")]
            if families:
                row["family"] = max(families, key=len)
    catalog = {name: [rows[key] for key in sorted(rows)] for name, rows in groups.items()}
    digest = hashlib.sha256(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return {"key": f"glossary-{digest}", **catalog}
