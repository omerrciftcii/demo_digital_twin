"""S6 = S4b düz katalog (satır başı kalıcı/tek etiketi) + S5 düzeltmeleri (reçete tarihi sahipliği, 'sen' hitabı sistem isteminde,
kategori ipucu satırı) + olası tekrar sinyali (aynı ICD kökü, ≥2 farklı tarih; kalıcılık ÜRETMEZ, yalnızca katalogda tarihlerle gösterilir).
STAGE: a | b (a + sınır dışı kanıt paketleri) | c (b + zaman-sinyali ifadesi seçicide; cevap istemi ayrı dosyada).
Genel kurallar; hastalık/soru listesi yok."""
import re, os
from . import s4frame as S4
from . import s5frame as S5
from .s4frame import label_targets, label_messages, label_schema, parse_labels, apply_labels, selector_schema, parse_selector, REASONS  # noqa
from .sframe import parse_user
STAGE = "b"  # demo: S6b sabit (bench: S6_STAGE ortam değişkeni)
ICD = re.compile(r"\(([A-Z]\d{2})(?:\.\d+)?\)")
D3 = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
PACK_MAX = 6
STOP = set("""bana beni benim bende bugün dün yine daha önce sonra şimdi şuan galiba gibi olabilir olur olmuş muydu mıydı
ne neden nasıl yapayım yapmalıyım yapmam gerek gerekir şikâyetim şikayetim şikâyet şikayet başladı başlamış oldu olmuştu
hissediyorum son zamanlarda kendimi kadar için ile ve veya bir bu şu o diğer tanımlanmamış birden fazla yerlerin yeri kayıt kayıtlarım""".split())

def _tok(s):
    s = s.replace("İ", "i").replace("I", "ı").lower()
    return [w for w in re.findall(r"[a-zçğıöşü]+", s) if len(w) >= 5 and w not in STOP]
def _stem(w): return w[:6]

def annotate(user):
    pre, secs, tail, plan, items = S4.annotate(user)
    _, sx, _ = parse_user(user)
    roots = {}
    for h, g in sx:
        if not h.startswith("Tanılar"): continue
        for l in g.split("\n"):
            m = ICD.search(l)
            if m and l.startswith("- "):
                for d in D3.findall(l): roots.setdefault(m.group(1), set()).add(d)
    for it in items:
        it["recur"] = None
        if it.get("kind") == "tanı":
            m = ICD.search(it["text"])
            if m and len(roots.get(m.group(1), ())) >= 2:
                ds = sorted(roots[m.group(1)], key=lambda d: d[6:] + d[3:5] + d[:2])
                it["recur"] = dict(root=m.group(1), dates=ds)
    return pre, secs, tail, plan, items

def catalog(items):
    def f(i):
        tag = "kalıcı" if i["persist"] else "tek seferlik"
        rec = " · olası tekrar (%s): %s" % (i["recur"]["root"], ", ".join(i["recur"]["dates"][-4:])) if i.get("recur") else ""
        return S4.S3.short("[%s] %s · %s · %s · %s%s" % (i["id"], i["age"], i["type"], tag, i["title"], rec), S4.S3.CAP + 110)
    return "\n".join(map(f, items))

def selector_messages(question, cat):
    sys = ("Görevin yalnızca seçim. Hastanın sorusunu ve hastanın 6 aydan eski kayıtlarının kataloğunu oku. Her kalem "
           "'kalıcı' (süregelen durum, tetkik/görüntüleme/patoloji/ameliyat sonucu, tekrarlamış tanı veya ilaç) ya da "
           "'tek seferlik' (tek bir akut epizot, tekil ilaç/başvuru) olarak işaretlidir. Bazı kalemlerde 'olası tekrar' notu "
           "vardır: aynı tanı grubunun hastada farklı tarihlerde birden çok kez kaydedildiğini gösterir; bunu önem sinyallerinden "
           "biri olarak tart. Soruyu cevaplarken işe yarayabilecek "
           "eski kayıtları seç. Kurallar: 'tek seferlik' bir kalemi yalnızca bugünkü şikâyetle aynı şikâyet/tanıysa seç "
           "(neden: ayni_sikayet). ayni_organ_sistem, kronik_durum ve onceki_tetkik nedenleri yalnızca 'kalıcı' kalemler "
           "için kullanılabilir. Genel şikâyetlerde (halsizlik, yorgunluk, kilo, çarpıntı gibi) vücudun genelini etkileyen "
           "hormon, kan, metabolizma ve beslenme durumlarını da gözden kaçırma. ")
    if STAGE == "c":
        sys += ("Kalemin yaşı (ne kadar önce olduğu) da bir önem sinyalidir: daha yakın tarihli kayıtlar genellikle daha "
                "fazla bilgi taşır, ama yaş tek başına seçme ya da eleme nedeni değildir; soruyla ilişkiyle birlikte tart. ")
    sys += ("En fazla 6 kalem seç, en ilgiliden başlayarak sırala. Kullanıcı geçmişini açıkça soruyorsa "
            "ilgili kalıcı kayıtları öncele. Hiçbiri ilgili değilse boş liste döndür.\n"
            'Yalnızca JSON döndür: {"ilgili": [{"id": "kN", "neden": <neden>}]}\nneden: ' + ", ".join(REASONS) + ".")
    return [dict(role="system", content=sys), dict(role="user", content="Soru: " + question + "\n\nEski kayıt kataloğu:\n" + cat)]

enforce = S4.enforce

def evidence_packs(question, items, taken):
    """Soru sözcükleri ↔ hastanın KENDİ tanı başlıkları (6 harf kök eşleşmesi; kataloğun >%20'sinde geçen kökler genel sayılır ve
    yok sayılır) + bulunan kalemin olası-tekrar grubu (aynı ICD kökü). Seçici sınırına sayılmaz; en fazla PACK_MAX."""
    if STAGE not in ("b", "c"): return []
    qs = {_stem(w) for w in _tok(question)}
    qcodes = set(re.findall(r"\b([A-Z]\d{2})", question))
    dx = [i for i in items if i.get("kind") == "tanı"]
    df = {}
    for i in dx:
        for s in {_stem(w) for w in _tok(i["title"])}: df[s] = df.get(s, 0) + 1
    lim = max(2, int(0.2 * len(dx)))
    hit = []
    for i in dx:
        ts = {_stem(w) for w in _tok(i["title"]) if df.get(_stem(w), 0) <= lim}
        m = ICD.search(i["text"]); root = m.group(1) if m else None
        if len(qs & ts) >= min(2, len(ts)) and ts or (root and root in qcodes): hit.append(i)   # en az 2 ortak kök (başlıkta tek kök varsa 1)
    roots = {i["recur"]["root"] for i in hit if i.get("recur")}
    for i in dx:
        if i.get("recur") and i["recur"]["root"] in roots and i not in hit: hit.append(i)
    out = []
    for i in hit:
        if i["id"] in taken: continue
        out.append((i["id"], "ayni_sikayet"))
        if len(out) >= PACK_MAX: break
    return out

def build_context(user, sel, items=None):
    return S5.build_context(user, sel)
