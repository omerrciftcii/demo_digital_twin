"""PDF extraction checks and contact sheets for local visual inspection only."""

import argparse
import json
import re
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    base = args.private_root
    rows = [
        json.loads(s)
        for s in (base / "gpu_16k_v1_1" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cases = {r["case"]["case_id"] for r in rows}
    for kind, filename in (
        ("qa", "MedicalAI_Soru_Yanit_2026-10-13.pdf"),
        ("eval", "MedicalAI_Degerlendirme_2026-10-13.pdf"),
    ):
        with pdfplumber.open(base / "output" / "pdf" / filename) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            assert "\ufffd" not in text and "\u25a0" not in text
            assert not re.search(r"[a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}", text)
            if kind == "qa":
                for case in cases:
                    assert case in text, case
                for label in ("Oracle Evidence", "Full CCO", "Selective", "Hybrid PoC"):
                    assert text.count(label) == 33, (label, text.count(label))
            bad = []
            for n, page in enumerate(pdf.pages, 1):
                # Footer is below 800pt; content still must remain inside the page.
                for char in page.chars:
                    if (
                        char["x0"] < 35
                        or char["x1"] > page.width - 35
                        or char["top"] < 25
                        or char["bottom"] > page.height - 20
                    ):
                        bad.append(n)
                        break
            assert not bad, f"text outside margins pages {bad}"
            print(kind, "pages", len(pdf.pages), "text_chars", len(text), "checks_passed")
        images = sorted((base / "tmp" / "pdfs").glob(kind + "-*.png"))
        for offset in range(0, len(images), 12):
            selected = images[offset : offset + 12]
            canvas = Image.new("RGB", (1200, ((len(selected) + 3) // 4) * 450), "#D9E2E7")
            draw = ImageDraw.Draw(canvas)
            for i, path in enumerate(selected):
                with Image.open(path) as page:
                    page.thumbnail((288, 420))
                    x, y = (i % 4) * 300 + 6, (i // 4) * 450 + 25
                    canvas.paste(page, (x, y))
                    draw.text((x, y - 20), path.stem, fill="black")
            canvas.save(base / "tmp" / "pdfs" / f"contact_{kind}_{offset // 12}.png")


if __name__ == "__main__":
    main()
