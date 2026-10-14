from __future__ import annotations

import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patient_centric_private"
    / "raw_cco_results"
    / "raw_cco_30_question_results_latest.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "pdf"
    / "MedicalAI_Ilk_Klinik_Soru_Cevap_Ciktilari_2026-10-14.pdf"
)

SELECTED = {
    "P01": [1, 2, 3, 4, 5],
    "P02": [1, 2, 3, 5, 8],
    "P03": [1, 2, 3, 4, 5],
}

DISPLAY_QUESTIONS = {
    1: "Geceleri iyi uyuyamıyorum. Neden olabilir?",
    2: "Son zamanlarda kendimi çok yorgun ve halsiz hissediyorum. Neden olabilir?",
    3: "Gözüm biraz kızardı ve batıyor. Neden olabilir?",
    4: "Başım ağrıyor. Bunun sebebi ne olabilir?",
    5: "Ara sıra çarpıntı hissediyorum. Bu normal mi?",
    8: "Genel sağlık durumumda dikkat etmem gereken bir şey var mı?",
}

QUALITY = {
    ("P01", 1): (
        "Kısmen başarılı",
        "Sayısal veriler doğru; eksik geceleri düzensizlik kanıtı sayması ve dengeleyici uyku göstergelerini geri planda bırakması yorumu gereğinden olumsuzlaştırıyor.",
    ),
    ("P01", 2): (
        "Başarılı",
        "Ön tanılarla aynı gün normal çıkan laboratuvarları birlikte değerlendiriyor; güncel kilo, hareket ve uyku verilerini kesin neden ilan etmeden kullanıyor.",
    ),
    ("P01", 3): (
        "Başarılı",
        "Kişisel kayıtta gözle ilgili kanıt olmadığını doğru yönetiyor ve uydurma kişisel bağlantı kurmuyor.",
    ),
    ("P01", 4): (
        "Zayıf",
        "Doğrudan kanıt yokken düşük adım ve tek gecelik uyku verisini baş ağrısına bağlıyor; 110-115 mg/dL ev ölçümlerini kaynakta olmayan bir normal etiketiyle sunuyor.",
    ),
    ("P01", 5): (
        "Başarılı",
        "Nabız değerleri ve trend doğru; çarpıntı anına ait ritim kaydı bulunmadığı daha açık belirtilmeliydi.",
    ),
    ("P02", 1): (
        "Kısmen başarılı",
        "Dört gecelik süreler ve 50 dakikalık uykuya dalma doğru; iki eksik geceyi uyku bozukluğu göstergesi gibi çerçevelemesi hatalı.",
    ),
    ("P02", 2): (
        "Kısmen başarılı",
        "Anemi değerleri doğru; Temmuz 2024 laboratuvarının bugün için eski olduğu ana yorumda yeterince görünür değil.",
    ),
    ("P02", 3): (
        "Başarılı",
        "Göz şikâyetine ilişkin kişisel kanıt olmadığını açıkça söylüyor; genel olasılıkları kayıtla karıştırmıyor.",
    ),
    ("P02", 5): (
        "Kısmen başarılı",
        "Nabız sayıları doğru; kaynakta referans değerlendirmesi bulunmadığı halde düşük-normal etiketi üretiyor.",
    ),
    ("P02", 8): (
        "Başarılı",
        "Son klinik kayıtları doğru tarih ve branşlarla ayırıyor; eski anemi kaydının güncel durumu göstermediğini belirtiyor.",
    ),
    ("P03", 1): (
        "Başarılı",
        "Beş aylık wearable veri boşluğunu doğru tanıyor ve bugünkü uyku hakkında sonuç üretmiyor.",
    ),
    ("P03", 2): (
        "Zayıf",
        "Normal aralıktaki hemoglobin ve hematokrit düşüşünü sorun gibi öne çıkarıyor; 2018 anemi öyküsünü bugünkü yorgunluğa gereğinden fazla bağlıyor.",
    ),
    ("P03", 3): (
        "Kısmen başarılı",
        "Göz geçmişini doğru buluyor; ancak yaklaşık 13 aylık kaydın bugünkü şikâyeti açıklamadığı daha açık söylenmeliydi.",
    ),
    ("P03", 4): (
        "Başarılı",
        "2022 baş ağrısı kaydını buluyor fakat bugünü yansıtmayacağını net biçimde ayırıyor.",
    ),
    ("P03", 5): (
        "Başarılı",
        "Güncel nabız ve ritim verisi olmadığını doğru saptıyor; eski wearable kaydından sonuç üretmiyor.",
    ),
}


def register_fonts() -> None:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("Report", str(regular)))
        pdfmetrics.registerFont(TTFont("Report-Bold", str(bold)))
        return
    pdfmetrics.registerFont(TTFont("Report", "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("Report-Bold", "DejaVuSans-Bold.ttf"))


def inline_markup(value: str) -> str:
    safe = escape(value)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)


def answer_flowables(answer: str, styles: dict[str, ParagraphStyle]) -> list[object]:
    output: list[object] = []
    paragraph_lines: list[str] = []

    def flush() -> None:
        if paragraph_lines:
            output.append(
                Paragraph(inline_markup(" ".join(paragraph_lines)), styles["answer"])
            )
            output.append(Spacer(1, 2.4 * mm))
            paragraph_lines.clear()

    for raw_line in answer.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        heading = re.fullmatch(r"\*\*(.+)\*\*", line)
        if heading:
            flush()
            output.append(Paragraph(escape(heading.group(1)), styles["answer_heading"]))
            output.append(Spacer(1, 1.2 * mm))
            continue
        bullet = re.match(r"^[*-]\s+(.+)$", line)
        if bullet:
            flush()
            output.append(
                Paragraph(f"• {inline_markup(bullet.group(1))}", styles["bullet"])
            )
            continue
        option = re.match(r"^([A-E])\)\s+(.+)$", line)
        if option:
            flush()
            output.append(
                Paragraph(
                    f"<b>{option.group(1)})</b> {inline_markup(option.group(2))}",
                    styles["option"],
                )
            )
            continue
        paragraph_lines.append(line)
    flush()
    return output


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    green = colors.HexColor("#176B59")
    dark = colors.HexColor("#18352F")
    muted = colors.HexColor("#667A73")
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Report-Bold",
            fontSize=25,
            leading=30,
            textColor=dark,
            alignment=TA_LEFT,
            spaceAfter=7 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName="Report",
            fontSize=11,
            leading=16,
            textColor=muted,
            spaceAfter=4 * mm,
        ),
        "patient": ParagraphStyle(
            "Patient",
            fontName="Report-Bold",
            fontSize=20,
            leading=24,
            textColor=dark,
            spaceAfter=6 * mm,
        ),
        "question_no": ParagraphStyle(
            "QuestionNo",
            fontName="Report-Bold",
            fontSize=10,
            leading=13,
            textColor=green,
            spaceBefore=2 * mm,
            spaceAfter=2 * mm,
        ),
        "question": ParagraphStyle(
            "Question",
            fontName="Report-Bold",
            fontSize=12.3,
            leading=17,
            textColor=dark,
            backColor=colors.HexColor("#EEF6F2"),
            borderColor=colors.HexColor("#D4E7DE"),
            borderWidth=0.6,
            borderPadding=9,
            spaceAfter=4 * mm,
        ),
        "answer_label": ParagraphStyle(
            "AnswerLabel",
            fontName="Report-Bold",
            fontSize=9,
            leading=11,
            textColor=muted,
            spaceAfter=2 * mm,
        ),
        "answer": ParagraphStyle(
            "Answer",
            fontName="Report",
            fontSize=9.2,
            leading=13.6,
            textColor=colors.HexColor("#263E37"),
        ),
        "answer_heading": ParagraphStyle(
            "AnswerHeading",
            fontName="Report-Bold",
            fontSize=9.6,
            leading=13,
            textColor=dark,
            spaceBefore=1.5 * mm,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            fontName="Report",
            fontSize=9.1,
            leading=13.2,
            leftIndent=4 * mm,
            firstLineIndent=-3.2 * mm,
            textColor=colors.HexColor("#263E37"),
            spaceAfter=1.2 * mm,
        ),
        "option": ParagraphStyle(
            "Option",
            fontName="Report",
            fontSize=8.8,
            leading=12.2,
            leftIndent=4 * mm,
            textColor=colors.HexColor("#3E5D54"),
            spaceAfter=0.8 * mm,
        ),
        "quality": ParagraphStyle(
            "Quality",
            fontName="Report",
            fontSize=8.8,
            leading=13,
            textColor=colors.HexColor("#493E2D"),
            backColor=colors.HexColor("#FBF7EE"),
            borderColor=colors.HexColor("#E9DDBF"),
            borderWidth=0.5,
            borderPadding=8,
            spaceBefore=3 * mm,
            spaceAfter=6 * mm,
        ),
        "section": ParagraphStyle(
            "Section",
            fontName="Report-Bold",
            fontSize=16,
            leading=20,
            textColor=dark,
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
        "body": ParagraphStyle(
            "Body",
            fontName="Report",
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#263E37"),
            spaceAfter=3 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            fontName="Report",
            fontSize=8.5,
            leading=12.5,
            textColor=muted,
            spaceAfter=2 * mm,
        ),
        "metric": ParagraphStyle(
            "Metric",
            fontName="Report-Bold",
            fontSize=15,
            leading=18,
            alignment=TA_CENTER,
            textColor=green,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            fontName="Report",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=muted,
        ),
    }


def footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#DDE8E2"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont("Report", 7.4)
    canvas.setFillColor(colors.HexColor("#7A8C85"))
    canvas.drawString(18 * mm, height - 10.5 * mm, "MedicalAI • İlk Klinik Soru-Cevap Çıktıları")
    canvas.drawString(
        18 * mm,
        10 * mm,
        "Sentetik olmayan özel sağlık verisi içerir • Sınırlı paylaşım",
    )
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
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
                ("TOPPADDING", (0, 1), (-1, 1), 1),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
            ]
        )
    )
    return table


def build() -> None:
    register_fonts()
    styles = make_styles()
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    indexed = {
        (item["patient_id"], item["question_number"]): item for item in data["results"]
    }
    selected_items = [
        indexed[(patient_id, number)]
        for patient_id, numbers in SELECTED.items()
        for number in numbers
    ]
    latencies = [item["latency_ms"] for item in selected_items]
    verdicts = [QUALITY[(item["patient_id"], item["question_number"])][0] for item in selected_items]
    successful = verdicts.count("Başarılı")
    partial = verdicts.count("Kısmen başarılı")
    weak = verdicts.count("Zayıf")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=17 * mm,
        title="MedicalAI - İlk Klinik Soru-Cevap Çıktıları",
        author="MedicalAI",
        subject="Gemma 4 ham CCO soru-cevap çıktıları ve kısa kalite analizi",
    )
    story: list[object] = [
        Spacer(1, 22 * mm),
        Paragraph("MedicalAI", styles["subtitle"]),
        Paragraph("İlk Klinik Soru-Cevap Çıktıları", styles["title"]),
        Paragraph(
            "Gemma 4 • Ham CCO bağlamı • 3 hasta • 15 gerçek soru-cevap",
            styles["subtitle"],
        ),
        Spacer(1, 6 * mm),
        Paragraph(
            "İlk çıktı • Retrieval denemeleri devam ediyor",
            styles["question"],
        ),
        Spacer(1, 8 * mm),
    ]
    metrics = Table(
        [[
            metric_cell("15", "gerçek yanıt", styles),
            metric_cell(f"{sum(latencies) / len(latencies) / 1000:.2f} sn", "ortalama cevap", styles),
            metric_cell(f"{min(latencies) / 1000:.2f}-{max(latencies) / 1000:.2f} sn", "ölçülen aralık", styles),
        ]],
        colWidths=[52 * mm, 52 * mm, 52 * mm],
        hAlign="LEFT",
    )
    metrics.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend(
        [
            metrics,
            Spacer(1, 9 * mm),
            Paragraph("Çalışma koşulu", styles["section"]),
            Paragraph(
                "Kullanılan hasta kayıtları sentetik değildir ve özel sağlık verisi içerir. Her soru bağımsız tek turda çalıştırıldı. CCO JSON içindeki uzun text alanı değiştirilmeden modele verildi. Bu ilk çıktı retrieval, fact JSON ve verifier olmadan üretildi. Temperature 0, thinking kapalı, en fazla 900 yeni token kullanıldı.",
                styles["body"],
            ),
            Paragraph(
                "Kalite dağılımı: "
                f"{successful} başarılı, {partial} kısmen başarılı, {weak} zayıf. "
                "Kalite notları kısa tutuldu ve kaynak CCO ile doğrudan karşılaştırmaya dayandırıldı.",
                styles["small"],
            ),
        ]
    )

    for patient_index, (patient_id, question_numbers) in enumerate(SELECTED.items()):
        story.append(PageBreak())
        story.append(Paragraph(f"{patient_id} • 5 soru", styles["patient"]))
        for display_number, question_number in enumerate(question_numbers, start=1):
            item = indexed[(patient_id, question_number)]
            verdict, note = QUALITY[(patient_id, question_number)]
            story.extend(
                [
                    CondPageBreak(66 * mm),
                    Paragraph(f"SORU {display_number}", styles["question_no"]),
                    Paragraph(
                        escape(DISPLAY_QUESTIONS[question_number]), styles["question"]
                    ),
                    Paragraph(
                        f"ALINAN YANIT • {item['latency_ms'] / 1000:.2f} sn",
                        styles["answer_label"],
                    ),
                ]
            )
            story.extend(answer_flowables(item["answer"], styles))
            story.append(
                Paragraph(
                    f"<b>Kalite: {escape(verdict)}</b> — {escape(note)}",
                    styles["quality"],
                )
            )

    story.extend(
        [
            PageBreak(),
            Paragraph("Genel değerlendirme", styles["patient"]),
            Paragraph(
                "İlk çıktı umut verici. Güncel ve açık sayısal verilerde tarih, değer ve trend aktarımı çoğunlukla doğru. Veri tamamen eksik olduğunda model özellikle P03 uyku ve çarpıntı sorularında uygun biçimde yorumdan kaçındı.",
                styles["body"],
            ),
            Paragraph(
                "Reasoning kalitesi doğrudan kanıt bulunan sorularda iyi, semptomu dolaylı verilerle açıklama gereken sorularda değişken. Başlıca hatalar; kaynakta bulunmayan normal/düşük etiketleri, eksik veriyi klinik düzensizlik gibi yorumlama, eski kayıtları bugünkü belirtiyle gereğinden güçlü ilişkilendirme ve zayıf nedensellik kurma.",
                styles["body"],
            ),
            Paragraph(
                f"Seçilen 15 yanıtta ortalama cevap süresi {sum(latencies) / len(latencies) / 1000:.2f} saniye; aralık {min(latencies) / 1000:.2f}-{max(latencies) / 1000:.2f} saniye.",
                styles["body"],
            ),
            Paragraph(
                "Retrieval denemeleri devam ediyor. Sonraki iyileştirme odağı; soruyla doğrudan ilgili kayıtları daha temiz seçmek, tarih/güncellik ağırlığını güçlendirmek, kaynakta olmayan klinik etiketleri engellemek ve dolaylı bulgulardan neden üretme eğilimini azaltmak. Mevcut model ve donanımla hasta dostu, hızlı ve kişiselleştirilmiş yanıt üretimi mümkün görünüyor; klinik güvenilirlik için bu sınırların sistematik biçimde daraltılması gerekiyor.",
                styles["body"],
            ),
        ]
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    build()
