"""Private two-PDF report. Requires finalized rows/summary and reviewed notes.

No network calls, no chart export, no patient identifier filenames. Raw answer
text remains verbatim except XML escaping for PDF layout. User-authorized Drive
upload is a separate step. Outputs must stay in the ignored evaluation folder.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "oracle": "Oracle Evidence",
    "full": "Full CCO",
    "selective": "Selective",
    "hybrid": "Hybrid PoC",
}
INK, TEAL, LIGHT = (
    colors.HexColor("#142F41"),
    colors.HexColor("#167D8D"),
    colors.HexColor("#EDF5F7"),
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return "N/A" if value is None else f"%{100 * value:.1f}".replace(".", ",")


def number(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}".replace(".", ",")


def styles(font_dir):
    pdfmetrics.registerFont(TTFont("Report", str(font_dir / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("ReportBold", str(font_dir / "arialbd.ttf")))
    pdfmetrics.registerFontFamily(
        "Report", normal="Report", bold="ReportBold", italic="Report", boldItalic="ReportBold"
    )
    result = getSampleStyleSheet()
    for style in result.byName.values():
        style.fontName = "Report"
    result.add(
        ParagraphStyle(
            "Body",
            fontName="Report",
            fontSize=9.4,
            leading=13.5,
            textColor=INK,
            spaceAfter=6,
            alignment=TA_LEFT,
        )
    )
    result.add(ParagraphStyle("SmallText", parent=result["Body"], fontSize=8, leading=11))
    result.add(
        ParagraphStyle(
            "Cover", fontName="ReportBold", fontSize=28, leading=34, textColor=INK, spaceAfter=18
        )
    )
    result.add(
        ParagraphStyle(
            "Section",
            fontName="ReportBold",
            fontSize=16,
            leading=21,
            textColor=INK,
            spaceBefore=12,
            spaceAfter=10,
            keepWithNext=True,
        )
    )
    result.add(
        ParagraphStyle(
            "Arm",
            fontName="ReportBold",
            fontSize=11,
            leading=15,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        )
    )
    return result


def para(text, sty, name="Body"):
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), sty[name])


def footer(canvas, doc):
    canvas.setStrokeColor(TEAL)
    canvas.line(19 * mm, 16 * mm, 191 * mm, 16 * mm)
    canvas.setFont("Report", 8)
    canvas.setFillColor(INK)
    canvas.drawString(19 * mm, 11 * mm, "MedicalAI | Özel değerlendirme - kimlikler kodlanmıştır")
    canvas.drawRightString(191 * mm, 11 * mm, str(doc.page))


def build(path, story):
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=19 * mm,
        bottomMargin=22 * mm,
        title=path.stem.replace("_", " "),
        author="MedicalAI Evaluation",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def table(headers, rows, sty, widths=None):
    data = [[para(v, sty, "SmallText") for v in headers]]
    data.extend([[para(v, sty, "SmallText") for v in row] for row in rows])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                ("LINEBELOW", (0, 0), (-1, 0), 1, TEAL),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#D8E2E7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return t


def cover(title, subtitle, sty):
    return [
        Spacer(1, 25 * mm),
        para("MEDICAL AI / DENEY RAPORU", sty, "Arm"),
        para(title, sty, "Cover"),
        para(subtitle, sty),
        Spacer(1, 8 * mm),
        HRFlowable(width="100%", thickness=2, color=TEAL),
        Spacer(1, 7 * mm),
        para("13 Ekim 2026 | Gemma 4 26B-A4B AWQ-4bit", sty),
        para(
            "16.384 token istemci bütçesi; sunucu üst sınırı 32.768 token. "
            "Production davranışı değiştirilmedi. Klinik kullanım onayı değildir.",
            sty,
        ),
        para(
            "P01, P02 ve P03 kimliği kodlanmış hasta etiketleridir. Bu dosya yine de "
            "hassas sağlık içeriği barındırır; herkese açık paylaşım için değildir.",
            sty,
        ),
    ]


def qa_pdf(rows, out, sty):
    story = cover(
        "Sorular ve yanıtlar",
        "Hasta odaklı konuşma benchmark'ı: "
        "15 tek turlu soru + 6 konuşmada 18 tur = 33 vaka, 4 koşul, 132 sonuç satırı.",
        sty,
    )
    story += [
        para("Okuma kılavuzu", sty, "Section"),
        para(
            "Her vakada aynı soru dört koşulda gösterilir. Yanıtlar verifier ve varsa retry "
            "sonrasında oluşan deney çıktılarıdır; klinik olarak onaylanmış yanıtlar değildir. "
            "Güvenlik yönlendiricisi bu reasoning deneyinde uygulanmadı, kararları ayrıca ölçüldü. "
            "Dolayısıyla bu belge production API davranışını temsil etmez.",
            sty,
        ),
        para(
            "Konuşma turları aynı koşulun önceki gerçek yanıtını kullanır. Önceki kullanıcı "
            "soruları her turda gösterilir; önceki asistan yanıtı ilgili önceki vakada bulunur. "
            "Kaynak CCO metinleri ve kimlikler bu belgeye eklenmemiştir. Sayısal değerlendirme, "
            "kanıt kapsamı ve hata incelemesi ikinci PDF'dedir.",
            sty,
        ),
    ]
    by_case = OrderedDict()
    for row in rows:
        by_case.setdefault(row["case"]["case_id"], {})[row["arm"]] = row
    story.append(PageBreak())
    story.append(para("Vaka dizini", sty, "Section"))
    story.append(
        table(
            ["No", "Hasta / vaka", "Soru türü"],
            [
                [i, cid, next(iter(arms.values()))["case"]["symptom"]]
                for i, (cid, arms) in enumerate(by_case.items(), 1)
            ],
            sty,
            [12 * mm, 110 * mm, 50 * mm],
        )
    )
    for i, (cid, arms) in enumerate(by_case.items(), 1):
        case = next(iter(arms.values()))["case"]
        story.extend(
            [
                PageBreak(),
                para(f"{i:02d} / {case['patient_id']}", sty, "Arm"),
                para(cid, sty, "Section"),
                para("Soru: " + case["question"], sty),
            ]
        )
        if case.get("user_history"):
            story.append(
                para(
                    "Önceki kullanıcı soruları: " + " / ".join(case["user_history"]),
                    sty,
                    "SmallText",
                )
            )
        for arm, label in LABELS.items():
            row = arms[arm]
            answer = row["result"].get("answer") or "Yanıt üretilmedi."
            if re.search(r"[a-fA-F0-9]{8}-(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}", answer):
                raise ValueError("identifier_in_answer_manual_redaction_required")
            blocks = [p.strip() for p in answer.split("\n") if p.strip()]
            # Keep the short answer with its disclaimer, never a disclaimer-only page.
            story.append(KeepTogether([para(label, sty, "Arm"), *[para(p, sty) for p in blocks]]))
    build(out, story)


def evaluation_pdf(summary, retrieval, emergency, manifest, notes, benchmark, out, sty):
    story = cover(
        "Karşılaştırmalı değerlendirme",
        "Retrieval, reasoning, context composition "
        "ve verifier etkilerini ayıran teknik değerlendirme ve nitel hata incelemesi.",
        sty,
    )
    for text in notes["executive_summary"]:
        story.append(para(text, sty))
    story.extend([PageBreak(), para("1. Deney kapsamı ve karşılaştırma sınırları", sty, "Section")])
    story.extend(para(p, sty) for p in notes["design"])
    story.append(
        table(
            ["Koşul", "Modelin aldığı bağlam"],
            [
                [
                    "Oracle",
                    "Gold grupların tüm alternatif fact'leri + aynı kaynak satırındaki diğer fact'ler; satırlar aynen.",
                ],
                [
                    "Full CCO",
                    "Tam orijinal CCO + tüm fact kataloğu; taşarsa kısaltma yok, model çağrısı yok.",
                ],
                [
                    "Selective",
                    "Mevcut build_messages + _fit_evidence; gerçek bütçe/prompt davranışı korunur.",
                ],
                [
                    "Hybrid PoC",
                    "Mevcut ranking + CPU E5, eşit RRF60; ilk 20 fact'in kaynak satırları, aynı satır fact'leri de eklenir.",
                ],
            ],
            sty,
            [31 * mm, 141 * mm],
        )
    )
    story.append(para("2. Retrieval sonuçları", sty, "Section"))
    rr = retrieval["summary"]["approaches"]
    story.append(
        table(
            ["Ranking", "R@5", "R@10", "R@20", "Kritik kaçırma @20"],
            [
                [
                    LABELS[a],
                    *[pct(rr[a]["all"]["recallAtK"][str(k)]["macroRecall"]) for k in (5, 10, 20)],
                    pct(rr[a]["all"]["recallAtK"]["20"]["criticalEvidenceMissRate"]),
                ]
                for a in ("selective", "hybrid")
            ],
            sty,
            [38 * mm, 26 * mm, 26 * mm, 26 * mm, 56 * mm],
        )
    )
    story.extend(para(p, sty) for p in notes["retrieval"])
    story.append(
        table(
            ["Ranking", "Tek tur R@20", "Konuşma R@20", "Sıcak ort./p95 ms"],
            [
                [
                    LABELS[a],
                    pct(rr[a]["singleTurn"]["recallAtK"]["20"]["macroRecall"]),
                    pct(rr[a]["conversationTurns"]["recallAtK"]["20"]["macroRecall"]),
                    number(rr[a]["all"]["latencyMs"]["warm"]["mean"])
                    + " / "
                    + number(rr[a]["all"]["latencyMs"]["warm"]["p95"]),
                ]
                for a in ("selective", "hybrid")
            ],
            sty,
            [38 * mm, 40 * mm, 40 * mm, 54 * mm],
        )
    )
    story.append(para("3. Model ve verifier - bütün vaka kapsamı", sty, "Section"))
    all_results = summary["all"]
    story.append(
        table(
            ["Koşul", "Çağrılan vaka", "Verifier geçen", "Retry", "Fallback", "Ort./p95 sn"],
            [
                [
                    LABELS[a],
                    f"{s['calledRows']}/{s['rows']}",
                    f"{s['verifierPassCount']}/{s['calledRows']}",
                    s["retryCount"],
                    s["fallbackCount"],
                    number(s["latencyMs"]["engineMean"] / 1000)
                    + " / "
                    + number(s["latencyMs"]["engineP95"] / 1000),
                ]
                for a, s in all_results.items()
            ],
            sty,
            [31 * mm, 25 * mm, 28 * mm, 18 * mm, 21 * mm, 49 * mm],
        )
    )
    story.extend(para(p, sty) for p in notes["verifier"])
    story.append(
        para(
            f"Eşleştirilmiş karşılaştırma: dört kolun da çağrıldığı {summary['matchedCaseCount']} vaka",
            sty,
            "Arm",
        )
    )
    story.append(
        table(
            ["Koşul", "Verifier geçen", "Katalog recall", "Metin recall", "Atıf recall"],
            [
                [
                    LABELS[a],
                    f"{s['verifierPassCount']}/{s['calledRows']}",
                    pct(s["catalogSent"]["groupRecall"]),
                    pct(s["rawTextSent"]["groupRecall"]),
                    pct(s["citation"]["groupRecall"]),
                ]
                for a, s in summary["matched"].items()
            ],
            sty,
            [34 * mm, 31 * mm, 35 * mm, 36 * mm, 36 * mm],
        )
    )
    story.append(para("4. Context ve provenance", sty, "Section"))
    story.append(
        table(
            [
                "Koşul",
                "Gönderilen katalog recall",
                "Gönderilen ham metin recall",
                "En büyük prompt",
            ],
            [
                [
                    LABELS[a],
                    pct(s["catalogSent"]["groupRecall"]),
                    pct(s["rawTextSent"]["groupRecall"]),
                    s["promptTokens"]["max"],
                ]
                for a, s in all_results.items()
            ],
            sty,
            [34 * mm, 47 * mm, 51 * mm, 40 * mm],
        )
    )
    story.extend(para(p, sty) for p in notes["provenance"])
    story.append(para("5. Klinik açıdan incelenebilir hata örnekleri", sty, "Section"))
    story.append(
        para(
            "Aşağıdaki bulgular ajan tarafından yapılan ön incelemedir; bağımsız klinisyen "
            "adjudikasyonu değildir. Bunlardan tüm benchmark için klinik doğruluk yüzdesi çıkarılmaz.",
            sty,
        )
    )
    for item in notes["observations"]:
        story.append(para(item["case"] + " / " + item["layer"], sty, "Arm"))
        story.append(para(item["finding"], sty))
        if item.get("quote"):
            story.append(para("Çıktıdan: " + item["quote"], sty, "SmallText"))
    story.append(para("6. Çok turlu ve çapraz hasta değerlendirmesi", sty, "Section"))
    story.extend(para(p, sty) for p in notes["conversation"])
    story.append(para("7. Emergency classifier - bağımsız sentetik set", sty, "Section"))
    story.extend(para(p, sty) for p in notes["emergency"])
    story.append(
        table(
            ["Eşik", "Test semantik TP/TN/FP/FN", "Test regex+semantik TP/TN/FP/FN"],
            [
                [
                    threshold,
                    "/".join(str(scores["semanticOnly"][k]) for k in ("tp", "tn", "fp", "fn")),
                    "/".join(str(scores["regexOrSemantic"][k]) for k in ("tp", "tn", "fp", "fn")),
                ]
                for threshold, scores in emergency["sweep"]["test"].items()
            ],
            sty,
            [24 * mm, 72 * mm, 76 * mm],
        )
    )
    story.append(para("8. Ölçüm sözlüğü ve açık klinik değerlendirme", sty, "Section"))
    story.extend(para(p, sty) for p in notes["metrics"])
    story.append(para("9. İzolasyon, test durumu ve kalan işler", sty, "Section"))
    story.extend(para(p, sty) for p in notes["limitations"])
    story.append(para("Benchmark SHA256: " + manifest["benchmarkHash"], sty, "SmallText"))
    story.append(
        para(
            "Kaynak dosyaların ayrı hash manifest'i özel deney klasöründe tutulur; "
            "branch cursor/deterministic-evidence-verification-dbef, HEAD fe8342c. "
            "Mevcut dirty çalışma ağacının sabit kopyası kullanılmıştır.",
            sty,
            "SmallText",
        )
    )
    story.append(para("Kaynaklar (yalnız sentetik triyaj tasarımı)", sty, "Arm"))
    story.append(para("NHS: https://www.nhs.uk/symptoms/heart-palpitations/", sty, "SmallText"))
    story.append(para("NHS: https://www.nhs.uk/symptoms/red-eye/", sty, "SmallText"))
    story.append(
        para("NHS: https://www.nhs.uk/symptoms/headaches/ (erişim: 13.10.2026)", sty, "SmallText")
    )
    story.extend([PageBreak(), para("Ek A. Bağımsız evidence taslağı", sty, "Section")])
    story.append(
        para(
            "Bu liste model çıktısından çıkarılmadı; benchmark v1.1 evidence profilleridir. "
            "Klinisyen onayı bekler. Her sorunun profile_refs alanı uygulanacak grupları belirler. "
            "Tek tur baş ağrısı/göz/yorgunluk/uykusuzluk kendi profilini, çarpıntı hem palpitations "
            "hem insomnia profilini kullanır. Yorgunluk konuşmasının ilk turu fatigue; ikinci ve "
            "üçüncü turu fatigue+insomnia kullanır. Göz konuşmasının üç turu eye_redness kullanır. "
            "ID'ler yalnız ilgili hastanın değişmemiş fact paketinde çözülmelidir.",
            sty,
        )
    )
    for profile_id, profile in benchmark["evidence_profiles"].items():
        story.append(para(profile_id, sty, "Arm"))
        found = False
        for role in ("direct", "indirect", "medication", "missingness"):
            for group in profile.get(role, []):
                found = True
                operator = "any_of" if group.get("any_of") else "all_of"
                story.append(
                    para(
                        f"{group['group_id']} | {role} | kritik={bool(group.get('critical'))} | {operator}\n"
                        + ", ".join(group[operator]),
                        sty,
                        "SmallText",
                    )
                )
        if not found:
            story.append(
                para(
                    "Pozitif eşleşen fact yok; Recall N/A. Global yokluk doğrulanmış sayılmaz.", sty
                )
            )
    story.extend([PageBreak(), para("Ek B. Sentetik triyaj soruları", sty, "Section")])
    story.append(
        para(
            "Etiketler klinisyen onayı bekler. Bunlar gerçek hastaların acil durum beyanları değildir.",
            sty,
        )
    )
    for row in emergency["rows"]:
        story.append(
            KeepTogether(
                [
                    para(
                        row["id"] + (" | acil" if row["emergency"] else " | acil değil"), sty, "Arm"
                    ),
                    para(row["question"], sty),
                    para(
                        "Cosine: " + number(row["cosine"], 4) + " | regex: " + str(row["regex"]),
                        sty,
                        "SmallText",
                    ),
                ]
            )
        )
    build(out, story)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--emergency", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()
    if not args.out_dir.resolve().is_relative_to(ROOT / "data" / "evaluation"):
        parser.error("PHI output must stay under ignored data/evaluation")
    rows = [
        json.loads(s)
        for s in (args.run_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary, manifest = read(args.run_dir / "summary.json"), read(args.run_dir / "manifest.json")
    if len(rows) != 132 or not summary["complete"]:
        raise ValueError("incomplete experiment")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sty = styles(args.font_dir)
    qa_pdf(rows, args.out_dir / "MedicalAI_Soru_Yanit_2026-10-13.pdf", sty)
    evaluation_pdf(
        summary,
        read(args.retrieval),
        read(args.emergency),
        manifest,
        read(args.notes),
        read(args.benchmark),
        args.out_dir / "MedicalAI_Degerlendirme_2026-10-13.pdf",
        sty,
    )
    print("Created two private PDFs")


if __name__ == "__main__":
    main()
