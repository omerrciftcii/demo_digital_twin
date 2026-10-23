"""S4: S3 + kalıcılık (yapısal sinyaller, isteğe bağlı model etiketi) + kod-düzeyinde seçim kuralları + ayrı geçmiş bölümü.
Yalnızca stdlib. Kurallar CCO YAPISINA dayanır (bölüm/alt başlık, ICD kodu biçimi, tarih, reçete sayısı); hastalık listesi yoktur."""
import json, re, datetime as dt
from . import s3frame as S3
from .sframe import join_user
REASONS = S3.REASONS
PERSIST_ONLY = {"ayni_organ_sistem", "kronik_durum", "onceki_tetkik"}   # yalnızca kalıcı kalemlere
ONEOFF_OK = {"ayni_sikayet"}                                           # tek seferlik kalemlere izinli tek neden
CAP_PICKS = 6
GAP = 60
PAST_H = "Geçmiş kayıtlar (bugünkü durumu yansıtmaz, ilgili olabilir)"
ICD = re.compile(r"\(([A-Z]\d{2})(?:\.\d+)?\)")
D3 = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})")
RX = re.compile(r"ilk reçete (\d{2})\.(\d{4}).*?son reçete (\d{2})\.(\d{2})\.(\d{4}).*?(\d+) reçete")

def _d(m): return dt.date(int(m[2]), int(m[1]), int(m[0]))
def dates(s): return [_d(m) for m in D3.findall(s)]
def norm(t):
    t = t[2:] if t.startswith("- ") else t
    t = t.split(" — ")[0]; t = re.sub(r"\([^()]*\)", "", t)
    return re.sub(r"[^a-zçğıöşü0-9 ]", "", t.replace("İ", "i").replace("I", "ı").lower()).strip()
def spread(ds): return len(ds) >= 2 and (max(ds) - min(ds)).days > GAP

def annotate(user):
    """-> (pre,secs,tail,plan,items) ve her kaleme: sub, kind, persist(bool), signals[list]."""
    pre, secs, tail, plan, items = S3.analyze(user)
    byid = {i["id"]: i for i in items}
    dx, kron, meds = [], [], []          # tüm tanı satırları (yeni+eski), kronik blok, reçeteler
    for h, rows in plan:
        sub = ""
        for kind, l, k in rows:
            if kind == "sub": sub = l; continue
            if kind == "oldgrp":
                for kk in k: byid[kk]["sub"] = sub
                continue
            if not l.startswith("- "): continue
            if k: byid[k].setdefault("sub", sub)
            if h.startswith("Tanılar"):
                rec = dict(icd=(ICD.search(l).group(1) if ICD.search(l) else None), nt=norm(l), d=dates(l),
                           pre_dx=("ön tanı" in l or "ÖN TANI" in sub), line=l)
                dx.append(rec)
                if "kronik" in sub.lower() and not sub.startswith("Geçmiş"): kron.append(rec)
            if h.startswith("İlaçlar"):
                m = RX.search(l)
                if m:
                    first = dt.date(int(m.group(2)), int(m.group(1)), 1); last = dt.date(int(m.group(5)), int(m.group(4)), int(m.group(3)))
                    meds.append(dict(line=l, n=int(m.group(6)), first=first, last=last, id=k))
    rep_meds = [m for m in meds if m["n"] >= 2 and (m["last"] - m["first"]).days > GAP]
    # (d) ilaç->tanı bağı yalnızca tekil eşleşmede: ilaç >=3 reçete, son reçete günü tam olarak TEK kesin tanı var
    dcount = {}
    for r in dx:
        if not r["pre_dx"]:
            for d in set(r["d"]): dcount[d] = dcount.get(d, 0) + 1
    rep_dates = {m["last"] for m in rep_meds if m["n"] >= 3 and dcount.get(m["last"]) == 1}
    for it in items:
        sig, h, sub = [], it["sec"], it.get("sub", "")
        line = it["text"].split("\n")[0]
        if it["type"] == "tahlil paneli" or h.startswith(("Görüntüleme", "Patoloji")): sig.append("a:tetkik")
        elif h.startswith("Ameliyat") and sub.startswith("Ameliyat"): sig.append("a:ameliyat")
        elif h.startswith("Tanılar"):
            it["kind"] = "tanı"
            if "ön tanı" in line or "ÖN TANI" in sub: it["pre_dx"] = True
            else:
                ic = ICD.search(line); ic = ic.group(1) if ic else None; nt = norm(line)
                same = lambda r: (ic and r["icd"] == ic) or r["nt"] == nt
                if any(same(r) for r in kron): sig.append("b:kronik_grup")
                ds = [d for r in dx if same(r) and not r["pre_dx"] for d in r["d"]]
                if spread(ds): sig.append("c:tekrar_tani")
                if set(dates(line)) & rep_dates: sig.append("d:tekrarli_ilac")
        elif h.startswith("İlaçlar"):
            m = next((m for m in meds if m["id"] == it["id"]), None)
            if m and m in rep_meds: sig.append("d:tekrarli_recete")
        elif h.startswith("Hastane"):
            if spread(dates(it["text"])): sig.append("c:tekrar_basvuru")
        it.setdefault("kind", "diğer"); it["signals"] = sig; it["persist_struct"] = bool(sig)
        it["persist"] = it["persist_struct"]
    return pre, secs, tail, plan, items

def label_targets(items):
    "Model etiketine gidecek kalemler: yapısal kalıcılığı olmayan, ön tanı olmayan eski tanılar."
    return [i for i in items if i["kind"] == "tanı" and not i.get("pre_dx") and not i["persist_struct"]]
def label_messages(targets):
    sys = ("Görevin yalnızca etiketleme. Her eski tanı kaydı için, tanının doğası gereği genellikle süregelen/kronik ya da "
           "tekrarlayan bir durum mu (kalici), yoksa tek seferlik akut bir epizot mu (tek_seferlik) olduğunu belirt. "
           "Emin değilsen tek_seferlik de. Hastanın bugünkü durumu hakkında yorum yapma.\n"
           'Yalnızca JSON döndür: {"etiketler": [{"id": "kN", "durum": "kalici" | "tek_seferlik"}]}')
    usr = "Eski tanı kayıtları:\n" + "\n".join("[%s] %s" % (i["id"], i["title"]) for i in targets)
    return [dict(role="system", content=sys), dict(role="user", content=usr)]
def label_schema(ids):
    return {"type": "json_schema", "json_schema": {"name": "etiket", "schema": {
        "type": "object", "additionalProperties": False, "required": ["etiketler"],
        "properties": {"etiketler": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["id", "durum"], "properties": {"id": {"type": "string", "enum": ids or [""]},
                                                       "durum": {"type": "string", "enum": ["kalici", "tek_seferlik"]}}}}}}}}
def parse_labels(txt, ids):
    if not txt: return None
    s = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    for m in [s] + re.findall(r"\{.*\}", s, re.S):
        try: j = json.loads(m); break
        except Exception: j = None
    if not isinstance(j, dict) or not isinstance(j.get("etiketler"), list): return None
    return {e["id"]: e.get("durum") for e in j["etiketler"] if isinstance(e, dict) and e.get("id") in ids and e.get("durum") in ("kalici", "tek_seferlik")}
def apply_labels(items, labels):
    for i in items:
        if labels and labels.get(i["id"]) == "kalici" and not i.get("pre_dx"):
            i["persist"] = True; i["signals"] = i["signals"] + ["m:model"]

def catalog(items):
    return "\n".join(S3.short("[%s] %s · %s · %s · %s" % (i["id"], i["age"], i["type"], "kalıcı" if i["persist"] else "tek seferlik", i["title"]), S3.CAP + 55) for i in items)
def selector_messages(question, cat):
    sys = ("Görevin yalnızca seçim. Hastanın sorusunu ve hastanın 6 aydan eski kayıtlarının kataloğunu oku. Her kalem "
           "'kalıcı' (süregelen durum, tetkik/görüntüleme/patoloji/ameliyat sonucu, tekrarlamış tanı veya ilaç) ya da "
           "'tek seferlik' (tek bir akut epizot, tekil ilaç/başvuru) olarak işaretlidir. Soruyu cevaplarken işe yarayabilecek "
           "eski kayıtları seç. Kurallar: 'tek seferlik' bir kalemi yalnızca bugünkü şikâyetle aynı şikâyet/tanıysa seç "
           "(neden: ayni_sikayet). ayni_organ_sistem, kronik_durum ve onceki_tetkik nedenleri yalnızca 'kalıcı' kalemler "
           "için kullanılabilir. En fazla 6 kalem seç, en ilgiliden başlayarak sırala. Kullanıcı geçmişini açıkça soruyorsa "
           "ilgili kalıcı kayıtları öncele. Hiçbiri ilgili değilse boş liste döndür.\n"
           'Yalnızca JSON döndür: {"ilgili": [{"id": "kN", "neden": <neden>}]}\nneden: ' + ", ".join(REASONS) + ".")
    return [dict(role="system", content=sys), dict(role="user", content="Soru: " + question + "\n\nEski kayıt kataloğu:\n" + cat)]
selector_schema = S3.selector_schema
parse_selector = S3.parse_selector

def enforce(sel, items):
    """Kod düzeyi kurallar + 6 sınırı. -> (kept [(id,neden)], dropped [{id,neden,why}])"""
    byid = {i["id"]: i for i in items}; ok, drop = [], []
    for k, n in sel:
        p = byid[k]["persist"]
        if not p and n not in ONEOFF_OK: drop.append(dict(id=k, neden=n, why="tek_seferlik_neden")); continue
        if p is False and n in PERSIST_ONLY: drop.append(dict(id=k, neden=n, why="tek_seferlik_neden")); continue
        ok.append((k, n))
    if len(ok) > CAP_PICKS:
        pri = [x for x in ok if byid[x[0]]["persist"]] + [x for x in ok if not byid[x[0]]["persist"]]
        keep = {x[0] for x in pri[:CAP_PICKS]}
        drop += [dict(id=k, neden=n, why="cap6") for k, n in ok if k not in keep]
        ok = [x for x in ok if x[0] in keep]
    return ok, drop

def build_context(user, sel, items=None):
    """sel: [(id,neden)] (uygulanmış) veya None => yedek (tam B bağlamı). Seçilenler ayrı bölümde, neden etiketi YOK."""
    pre, secs, tail, plan, its = annotate(user)
    if sel is None: return user, dict(fallback=True, n_old=len(its), n_sel=len(its))
    want = [k for k, _ in sel]; byid = {i["id"]: i for i in its}
    out = []
    for h, rows in plan:
        lines = []
        for kind, l, k in rows:
            if kind == "keep": lines.append(("x", l))
            elif kind == "sub": lines.append(("s", l))
        res = []
        for i, (t, l) in enumerate(lines):
            if t == "s":
                nxt = next((x for x in lines[i + 1:] if x[0] == "s" or x[1].startswith("- ")), None)
                if not nxt or nxt[0] == "s": continue
            res.append(l)
        body = "\n".join(res).strip("\n")
        if body.strip(): out.append((h, body + "\n\n"))
    if want:
        pl = []
        for k in want:
            it = byid[k]; sec = S3.stype(it["sec"])
            txt = it["text"]
            if txt.startswith("- "): txt = txt[2:]
            txt = txt.replace("\n- ", "; ")
            if it["type"] == "tahlil paneli": txt = "tahliller (%s): %s" % (it["age"], txt)
            elif not S3.AGE.search(txt): txt = "(%s) %s" % (it["age"], txt)
            pl.append("- %s: %s" % (sec, txt))
        out.append((PAST_H, "Aşağıdaki eski kayıtlar bu soruyla ilgili olabileceği için gösteriliyor; bugünkü durumu yansıtmaz.\n" + "\n".join(pl) + "\n\n"))
    if its:
        out.append(("Gösterilmeyen eski kayıtlar", "Daha eski diğer kayıtlar: %d (6 aydan eski; gösterilmedi).\n\n" % (len(its) - len(want))))
    return join_user(pre, out, tail), dict(fallback=False, n_old=len(its), n_sel=len(want), n_hidden=len(its) - len(want))
