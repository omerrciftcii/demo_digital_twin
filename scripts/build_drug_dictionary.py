"""Build an optional drug dictionary from a full public Turkish product list.

Source: Tip-Atlasi-Projesi/ilaclardb (CC0), file data/cvs/ilac.csv.zip.
Each row is one licensed product: id, barcode, ATC code, ingredient, product
name. This script groups products that share an ATC code and keeps the brand
token plus ingredient tokens. It does not list any drug by hand.

WHO ATC/DDD is an INN index and has no Turkish brand column, so it is not
the source here.

The file is written to data/drug_classes.json, outside src/. A full product
list contains names that the source scan forbids inside src. When the file
is absent, retrieval does not expand brands; the model reads the chart.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_ai_selective_context.index import normalize_text

SOURCE = (
    "https://raw.githubusercontent.com/Tip-Atlasi-Projesi/ilaclardb/"
    "HEAD/data/cvs/ilac.csv.zip"
)
DEFAULT_OUT = ROOT / "data" / "drug_classes.json"
_BRAND = re.compile(r"[a-z]{4,}")
_HEAD = re.compile(r"[a-z]{5,}")
_WORD = re.compile(r"[a-z]{4,}")
_ATC = re.compile(r"[A-Z]\d{2}[A-Z]{2}\d{2}")
# Words in at least this share of product names are packaging, not identity.
_PACKAGING_RATE = 0.01
_SPLIT = re.compile(r"\s*(?:,|;|/|\+|\bve\b|\band\b)\s*")
# This dataset writes this sentence when the ingredient cell is empty.
_MISSING_INGREDIENT = "etken maddesi bilgisi"


def _brand(product: str) -> str:
    match = _BRAND.search(normalize_text(product))
    return match.group(0) if match else ""


def _ingredients(field: str) -> list[str]:
    """First word of each ingredient. Later words are salts and strengths."""
    folded = normalize_text(field)
    if folded.startswith(_MISSING_INGREDIENT):
        return []
    names = []
    for part in _SPLIT.split(folded):
        match = _HEAD.match(part.strip())
        if match:
            names.append(match.group(0))
    return list(dict.fromkeys(names))


def entries_from(table: str) -> list[dict]:
    parsed: list[tuple[str, set[str]]] = []
    packaging: Counter[str] = Counter()
    products = 0
    for row in csv.reader(io.StringIO(table)):
        if len(row) < 5:
            continue
        products += 1
        folded_product = normalize_text(row[4])
        packaging.update(set(_WORD.findall(folded_product)))
        atc = row[2].strip().upper()
        names = {name for name in (_brand(row[4]), *_ingredients(row[3])) if name}
        parsed.append((atc, names))
    banned = {
        token
        for token, count in packaging.items()
        if products and count / products >= _PACKAGING_RATE
    }
    grouped: dict[str, set[str]] = {}
    for atc, names in parsed:
        names = {name for name in names if name not in banned}
        if not names:
            continue
        if _ATC.fullmatch(atc):
            key = atc
        else:
            key = "missing:" + "|".join(sorted(names))
        grouped.setdefault(key, set()).update(names)
    rows = []
    for key in sorted(grouped):
        atc = "" if key.startswith("missing:") else key
        names = sorted(name for name in grouped[key] if name not in banned)
        if names:
            rows.append({"group": atc or "missing", "atc": atc, "names": names})
    return rows


def main() -> None:
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    with urllib.request.urlopen(SOURCE, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(name for name in archive.namelist() if name.endswith("ilac.csv"))
        table = archive.read(member).decode("utf-8")
    rows = entries_from(table)
    if len(rows) < 1000:
        raise SystemExit(f"dictionary looks truncated ({len(rows)} entries)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "source": SOURCE,
                "description": (
                    "Generated from the public Turkish licensed-product list. "
                    "One entry is one ATC code. Do not edit names by hand."
                ),
                "entries": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    names = {name for row in rows for name in row["names"]}
    print(f"wrote {len(rows)} entries, {len(names)} names to {destination}")


if __name__ == "__main__":
    main()
