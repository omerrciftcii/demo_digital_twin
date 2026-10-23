"""S5 = S4b + (a) özet 'Son reçete/Son vizit … (tarih)' satırlarına hangi kaleme ait olduğu yazılır (yanlış tarih aktarımını önler),
(b) katalogda kalıcı kalemler ayrı ve önce listelenir, seçiciden her kalıcı kalemi tek tek değerlendirmesi istenir.
Genel kurallar; hastalık/soru listesi yok."""
import re
from . import s4frame as S4
from .s4frame import *          # noqa
from .sframe import parse_user
SUMLINE = re.compile(r"^(Son (?:reçete|vizit|kayıt|başvuru)[^()\n]*?)\((\d{2}\.\d{2}\.\d{4})\)\.?\s*$")

def _head(line, sec):
    t = line[2:] if line.startswith("- ") else line
    if sec.startswith("Hastane"):          # '- 27.08.2026 [..], Üroloji, genel muayene.'
        p = [x.strip() for x in t.split(",")]
        return p[1] if len(p) > 1 else t[:40]
    return t.split(" — ")[0].strip()

def label_summaries(user, ctx):
    """ctx içindeki 'Son reçete … (tarih).' satırlarına, aynı bölümde o tarihi SON tarih olarak taşıyan kalem adlarını ekler."""
    _, secs, _ = parse_user(user)
    owners = {}
    for h, g in secs:
        for l in g.split("\n"):
            if not l.startswith("- "): continue
            ds = re.findall(r"\d{2}\.\d{2}\.\d{4}", l.split("[")[0] if h.startswith("Hastane") else l)
            last = (re.search(r"son reçete (\d{2}\.\d{2}\.\d{4})", l) or re.search(r"(\d{2}\.\d{2}\.\d{4})", l))
            if last: owners.setdefault((h.split()[0], last.group(1)), []).append(_head(l, h))
    out, cur = [], ""
    for l in ctx.split("\n"):
        if l.startswith("## "): cur = l[3:].split()[0] if len(l) > 3 else ""
        m = SUMLINE.match(l.strip())
        if m and (cur, m.group(2)) in owners:
            names = list(dict.fromkeys(owners[(cur, m.group(2))]))
            more = " ve %d kalem daha" % (len(names) - 3) if len(names) > 3 else ""
            l = "%s(%s) — bu tarih yalnızca şu kalem(ler)e aittir: %s%s. Diğer kalemlerin tarihi kendi satırındadır." % (m.group(1), m.group(2), "; ".join(names[:3]), more)
        out.append(l)
    return "\n".join(out)

def build_context(user, sel, items=None):
    ctx, info = S4.build_context(user, sel)
    return label_summaries(user, ctx), info

def catalog(items):
    P = [i for i in items if i["persist"]]; T = [i for i in items if not i["persist"]]
    f = lambda i: S4.S3.short("[%s] %s · %s · %s" % (i["id"], i["age"], i["type"], i["title"]), S4.S3.CAP + 45)
    return ("KALICI KALEMLER (süregelen/tekrarlamış tanı, tetkik, ameliyat, tekrarlı ilaç) — her birini tek tek değerlendir:\n" +
            ("\n".join(map(f, P)) or "(yok)") + "\n\nTEK SEFERLİK KALEMLER (yalnızca aynı şikâyetse seç):\n" + ("\n".join(map(f, T)) or "(yok)"))

def selector_messages(question, cat):
    sys = ("Görevin yalnızca seçim. Hastanın sorusunu ve 6 aydan eski kayıtlarının kataloğunu oku. Katalog iki gruptur. "
           "1) KALICI KALEMLER: bunların HER BİRİNİ tek tek ele al ve şunu sor: bu süregelen/tekrarlamış durum, tetkik veya "
           "tedavi, bugünkü şikâyeti açıklamaya katkıda bulunabilir, onu etkileyebilir ya da doktorun bu soruyu değerlendirirken "
           "bilmesi gereken bir şey midir? Makul bir klinik ilişki varsa seç (neden: ayni_sikayet, ayni_organ_sistem, "
           "kronik_durum, onceki_tetkik, ilac_yan_etki veya ameliyat). Genel şikâyetlerde (halsizlik, yorgunluk, kilo, çarpıntı "
           "gibi) vücudun genelini etkileyen hormon, kan, metabolizma ve beslenme durumlarını da gözden kaçırma. "
           "2) TEK SEFERLİK KALEMLER: yalnızca bugünkü şikâyetle aynı şikâyet/tanıysa seç (neden: ayni_sikayet); "
           "eski akut olayları başka nedenle seçme. En fazla 6 kalem seç, en ilgiliden başlayarak sırala. Hiçbiri ilgili değilse "
           "boş liste döndür.\n"
           'Yalnızca JSON döndür: {"ilgili": [{"id": "kN", "neden": <neden>}]}\nneden: ' + ", ".join(REASONS) + ".")
    return [dict(role="system", content=sys), dict(role="user", content="Soru: " + question + "\n\nEski kayıt kataloğu:\n" + cat)]
