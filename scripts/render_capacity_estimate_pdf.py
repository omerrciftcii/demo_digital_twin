from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "pdf"
    / "MedicalAI_Tek_GPU_Kapasite_Tahmini_2026-10-14.pdf"
)


def register_fonts() -> None:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("Report", str(regular)))
        pdfmetrics.registerFont(TTFont("Report-Bold", str(bold)))
        return
    pdfmetrics.registerFont(TTFont("Report", "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("Report-Bold", "DejaVuSans-Bold.ttf"))


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    dark = colors.HexColor("#18352F")
    green = colors.HexColor("#176B59")
    muted = colors.HexColor("#667A73")
    return {
        "eyebrow": ParagraphStyle(
            "Eyebrow",
            fontName="Report",
            fontSize=9,
            leading=12,
            textColor=muted,
            spaceAfter=2 * mm,
        ),
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Report-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_LEFT,
            textColor=dark,
            spaceAfter=3 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName="Report",
            fontSize=9.3,
            leading=13,
            textColor=muted,
            spaceAfter=5 * mm,
        ),
        "highlight": ParagraphStyle(
            "Highlight",
            fontName="Report-Bold",
            fontSize=11,
            leading=15,
            textColor=dark,
            backColor=colors.HexColor("#EEF6F2"),
            borderColor=colors.HexColor("#D4E7DE"),
            borderWidth=0.6,
            borderPadding=8,
            spaceAfter=5 * mm,
        ),
        "section": ParagraphStyle(
            "Section",
            fontName="Report-Bold",
            fontSize=12.5,
            leading=15,
            textColor=dark,
            spaceBefore=2 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName="Report",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#263E37"),
            spaceAfter=2 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            fontName="Report",
            fontSize=7.4,
            leading=10,
            textColor=muted,
        ),
        "metric": ParagraphStyle(
            "Metric",
            fontName="Report-Bold",
            fontSize=14,
            leading=17,
            alignment=TA_CENTER,
            textColor=green,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            fontName="Report",
            fontSize=7.2,
            leading=9,
            alignment=TA_CENTER,
            textColor=muted,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            fontName="Report-Bold",
            fontSize=7.2,
            leading=9,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            fontName="Report",
            fontSize=7.2,
            leading=9,
            textColor=colors.HexColor("#263E37"),
            alignment=TA_CENTER,
        ),
        "table_cell_left": ParagraphStyle(
            "TableCellLeft",
            fontName="Report-Bold",
            fontSize=7.2,
            leading=9,
            textColor=dark,
            alignment=TA_LEFT,
        ),
    }


def footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#DDE8E2"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont("Report", 7.2)
    canvas.setFillColor(colors.HexColor("#7A8C85"))
    canvas.drawString(18 * mm, height - 10.5 * mm, "MedicalAI • Tek GPU Kapasite Tahmini")
    canvas.drawString(18 * mm, 10 * mm, "Ölçülen servis metriklerine dayalı tahmini kapasite")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Sayfa {doc.page}")
    canvas.restoreState()


def metric_cell(value: str, label: str, styles: dict[str, ParagraphStyle]) -> Table:
    table = Table(
        [[Paragraph(value, styles["metric"])], [Paragraph(label, styles["metric_label"])]],
        colWidths=[48 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F8F6")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D5E5DE")),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 0),
                ("TOPPADDING", (0, 1), (-1, 1), 1),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
            ]
        )
    )
    return table


def styled_table(data: list[list[Paragraph]], widths: list[float]) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B59")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#D5E5DE")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8F6")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build() -> None:
    register_fonts()
    styles = make_styles()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=16 * mm,
        title="MedicalAI - Tek GPU Kapasite Tahmini",
        author="MedicalAI",
        subject="Mevcut Gemma 4 ve vLLM servisi için kapasite tahmini",
    )

    story: list[object] = [
        Spacer(1, 5 * mm),
        Paragraph("MedicalAI", styles["eyebrow"]),
        Paragraph("Tek GPU Kapasite Tahmini", styles["title"]),
        Paragraph(
            "Mevcut Gemma 4 26B-A4B AWQ 4-bit ve vLLM servisinin ölçülen çalışma değerleri üzerinden 1,2 milyon kayıtlı kullanıcı için yaklaşık kapasite.",
            styles["subtitle"],
        ),
        Paragraph(
            "Planlama değeri: tek GPU için yaklaşık 100.000 soru/gün. Kesin kapasite, kontrollü eşzamanlı yük testiyle belirlenmelidir.",
            styles["highlight"],
        ),
    ]

    metrics = Table(
        [[
            metric_cell("24 GB", "NVIDIA L40S-24Q", styles),
            metric_cell("3,8-4,3 sn", "gözlenen yanıt süresi", styles),
            metric_cell("~100 bin", "planlanan soru/gün", styles),
        ]],
        colWidths=[52 * mm, 52 * mm, 52 * mm],
        hAlign="LEFT",
    )
    metrics.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend([metrics, Spacer(1, 4 * mm)])

    story.extend(
        [
            Paragraph("Ölçülen çalışma koşulu", styles["section"]),
            Paragraph(
                "Servis 32.768 token bağlam, en fazla 16 aktif dizi ve prefix cache ile çalışıyor. 554 tamamlanmış istekte ortalama giriş 8.855 token, ortalama çıktı 319 token ve model katmanı ortalama gecikmesi 2,98 saniye ölçüldü. Uzun ve birbirinden farklı hasta kayıtlarında gerçek eşzamanlılık, bellek nedeniyle yapılandırmadaki 16 değerinin altında kalabilir.",
                styles["body"],
            ),
            Paragraph("Tek GPU için tahmini kapasite", styles["section"]),
        ]
    )

    headers = ["Senaryo", "Soru/sn", "Günlük", "Aylık", "Kullanıcı başına/ay"]
    rows = [
        ["En kötü ihtimal", "0,5-0,7", "40-60 bin", "1,2-1,8 milyon", "1-1,5 soru"],
        ["Beklenen ihtimal", "1-1,4", "90-120 bin", "2,7-3,6 milyon", "2-3 soru"],
        ["En iyi ihtimal", "2-3", "180-250 bin", "5,4-7,5 milyon", "4,5-6 soru"],
    ]
    capacity_data: list[list[Paragraph]] = [
        [Paragraph(value, styles["table_head"]) for value in headers]
    ]
    for row in rows:
        capacity_data.append(
            [Paragraph(row[0], styles["table_cell_left"])]
            + [Paragraph(value, styles["table_cell"]) for value in row[1:]]
        )
    story.extend(
        [
            styled_table(capacity_data, [35 * mm, 24 * mm, 28 * mm, 35 * mm, 38 * mm]),
            Spacer(1, 4 * mm),
            Paragraph("1,2 milyon kullanıcının her gün bir soru sorması", styles["section"]),
        ]
    )

    daily_headers = ["Senaryo", "Tahmini GPU", "Açıklama"]
    daily_rows = [
        ["En kötü ihtimal", "20-30", "Uzun benzersiz bağlam, düşük cache ve yoğun kuyruk"],
        ["Beklenen ihtimal", "12-16", "Yoğun saat ve arıza payı dahil planlama"],
        ["En iyi ihtimal", "6-8", "Yüksek cache verimi ve dengeli trafik"],
    ]
    daily_data: list[list[Paragraph]] = [
        [Paragraph(value, styles["table_head"]) for value in daily_headers]
    ]
    for row in daily_rows:
        daily_data.append(
            [Paragraph(row[0], styles["table_cell_left"])]
            + [Paragraph(row[1], styles["table_cell"]), Paragraph(row[2], styles["table_cell"])]
        )
    story.extend(
        [
            styled_table(daily_data, [39 * mm, 31 * mm, 90 * mm]),
            Spacer(1, 4 * mm),
            Paragraph("Beklenen kullanım örneği", styles["section"]),
            Paragraph(
                "Kullanıcıların %10'u günlük aktif olur ve kişi başına bir soru sorarsa yük yaklaşık 120.000 soru/gündür. Ortalama yük için 2 GPU yeterli olabilir; yoğun saat, bakım ve arıza yedekliliğiyle 3-4 GPU daha güvenli başlangıç kapasitesidir.",
                styles["body"],
            ),
            Paragraph(
                "Sonuç: 1,2 milyon kayıtlı kullanıcı, kişi başına ayda birkaç soru düzeyinde mevcut mimari ölçeklenerek karşılanabilir. Tüm kullanıcıların her gün aktif olduğu kullanımda çoklu GPU kümesi gerekir. Rakamlar model servisi içindir; API, kuyruk, kimlik doğrulama, yük dengeleme ve PHI güvenliği ayrıca boyutlandırılmalıdır.",
                styles["highlight"],
            ),
            Paragraph(
                "Not: Bu çalışma canlı servise ek yük uygulanmadan, mevcut yapılandırma ve ölçülen istek metrikleri üzerinden hazırlanmış kapasite tahminidir.",
                styles["small"],
            ),
        ]
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    build()
