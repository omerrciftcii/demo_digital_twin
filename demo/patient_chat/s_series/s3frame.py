"""S3: kayıt düzeyinde ilgililik seçimi. Yalnızca stdlib. Kurallar CCO YAPISINA dayanır
(## bölüm, '- ' kalemleri, '[... önce]' yaş etiketleri, alt başlıklar); hastalık/soru içeriğine bağlı kural yoktur."""
import json, re
from .sframe import parse_user, join_user
OLD_DAYS = 180
REASONS = ["ayni_sikayet", "ayni_organ_sistem", "ilac_yan_etki", "kronik_durum", "onceki_tetkik", "ameliyat"]
LABEL = {"ayni_sikayet": "aynı şikâyet", "ayni_organ_sistem": "aynı organ/sistem", "ilac_yan_etki": "ilaç yan etkisi",
         "kronik_durum": "süregelen durum", "onceki_tetkik": "önceki tetkik", "ameliyat": "ameliyat"}
AGE = re.compile(r"\[(?:(\d+) (gün|hafta|ay|yıl) önce|(bugün|dün|bu ay))\]")
DATE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})(?: \d{2}:\d{2})?")
MULT = {"gün": 1, "hafta": 7, "ay": 30, "yıl": 365}
CAP = 130

def age_days(line):
    m = AGE.search(line)
    if not m: return None
    if m.group(3): return {"bugün": 0, "dün": 1, "bu ay": 0}[m.group(3)]
    return int(m.group(1)) * MULT[m.group(2)]
def age_txt(line):
    m = AGE.search(line); return m.group(0)[1:-1] if m else ""
def is_sub(l):
    s = l.strip(); return bool(s) and not s.startswith("- ") and (s.endswith(":") or s.startswith("Geçmiş kayıtlar"))
def stype(h):  # bölüm başlığından kısa tür adı (genel)
    return re.sub(r"\s*\(.*?\)", "", h).strip().replace("İ", "i").replace("I", "ı").lower()
def short(s, n=CAP):
    s = re.sub(r"\s+", " ", s).strip(); return s if len(s) <= n else s[:n - 1].rstrip() + "…"
def title(line):
    t = line[2:] if line.startswith("- ") else line
    cert = re.search(r"\b(kesin tanı|ön tanı|kesinlik belirtilmemiş)\b", t)
    rx = re.search(r"(\d+) reçete\b", t)
    head = t.split(" — ")[0] if " — " in t else t
    head = AGE.sub("", DATE.sub("", head)); head = re.sub(r"\s+([,.)])", r"\1", head).strip(" ,.;")
    if "Tanı:" in t and "Tanı:" not in head: head += " · " + t.split("Tanı:", 1)[1]
    ext = [x for x in ((cert.group(1) if cert else None), (rx.group(0) if rx else None)) if x]
    return short(head, CAP - 18) + (" (" + ", ".join(ext) + ")" if ext else "")
def labname(l):
    t = re.sub(r"\s*\([^()]*\)", "", l[2:]); return re.split(r" [<>≤≥]?%?-?\d", t, 1)[0].strip()
def years(dates):
    ys = sorted({d[2] for d in dates}); return ys[0] if len(ys) == 1 else ys[0] + "–" + ys[-1]

def analyze(user):
    """-> (pre, secs, tail, plan). plan: bölüm başına satır listesi; her satır ('keep'|'old'|'sub', metin, item_id)."""
    pre, secs, tail = parse_user(user)
    items, plan = [], []
    for h, g in secs:
        rows, sub, lab_groups = [], "", {}
        for l in g.split("\n"):
            if is_sub(l): sub = l; rows.append(["sub", l, None]); continue
            if not l.startswith("- "): rows.append(["keep", l, None]); continue
            if "kronik" in sub.lower() and not sub.startswith("Geçmiş"):   # kronik blok: her zaman
                rows.append(["keep", l, None]); continue
            a = age_days(l)
            if a is None:
                ds = DATE.findall(l)
                if " · " in l and len(ds) > 1:      # sıkıştırılmış eski liste: bölümsel/branş gruplu kalemlere ayır
                    body = l[2:].split(":", 1)[-1]; grp = {}
                    for part in body.split(" · "):
                        d = DATE.search(part)
                        if not d: continue
                        key = part[d.end():].split(",")[0].strip(" .") or "?"
                        grp.setdefault(key, []).append((d.groups(), part.strip(" .")))
                    ids = []
                    for key, v in grp.items():
                        k = "k%d" % (len(items) + 1)
                        items.append(dict(id=k, sec=h, age=years([x[0] for x in v]), type=stype(h),
                                          title=short("%s, %d eski kayıt" % (key, len(v))), text="- " + "; ".join(x[1] for x in v) + "."))
                        ids.append(k)
                    rows.append(["oldgrp", l, ids]); continue
                rows.append(["keep", l, None]); continue
            if a <= OLD_DAYS: rows.append(["keep", l, None]); continue
            if h.startswith("Laboratuvar"):        # aynı tarihli eski tahliller tek kalem (panel)
                d = DATE.search(l); key = d.group(0) if d else l
                if key in lab_groups:
                    it = items[lab_groups[key]]; it["text"] += "\n" + l; it["names"].append(labname(l))
                    if re.search(r"\b(yüksek|düşük|anormal)\b", l): it["abn"].append(labname(l))
                    rows.append(["old", l, it["id"]]); continue
                k = "k%d" % (len(items) + 1); lab_groups[key] = len(items)
                items.append(dict(id=k, sec=h, age=age_txt(l), type="tahlil paneli", names=[labname(l)],
                                  abn=[labname(l)] if re.search(r"\b(yüksek|düşük|anormal)\b", l) else [], text=l, date=key))
                rows.append(["old", l, k]); continue
            k = "k%d" % (len(items) + 1)
            items.append(dict(id=k, sec=h, age=age_txt(l), type=stype(h), title=title(l), text=l))
            rows.append(["old", l, k])
        plan.append((h, rows))
    for it in items:
        if it["type"] == "tahlil paneli":
            it["title"] = short("%s: %s" % (it["date"], ", ".join(it["names"])) if not it["abn"] else "%s: anormal: %s; diğer: %s" % (it["date"], ", ".join(it["abn"]), ", ".join(n for n in it["names"] if n not in it["abn"])))
    return pre, secs, tail, plan, items

def catalog(items):
    return "\n".join(short("[%s] %s · %s · %s" % (i["id"], i["age"], i["type"], i["title"]), CAP + 40) for i in items)

def selector_messages(question, cat):
    sys = ("Görevin yalnızca seçim. Hastanın sorusunu ve hastanın 6 aydan eski kayıtlarının kataloğunu oku. "
           "Soruyu cevaplarken bakılması makul olarak işe yarayabilecek eski kayıtları seç. Kapsayıcı ol: makul bir ilişki "
           "varsa seç (aynı veya benzer şikâyetin daha önce olması, aynı organ/sistemle ilgili kayıt, ilaç yan etkisi veya "
           "alerji/reaksiyon, süregelen olabilecek bir hastalık, ilgili önceki tetkik, ilgili ameliyat). Soruyla ilgisiz eski "
           "akut olayları (başka bölgeyle ilgili enfeksiyon, tek seferlik başvuru vb.) seçme. Kullanıcı geçmişini soruyorsa "
           "ilgili tüm eski kayıtları seç. Hiçbiri ilgili değilse boş liste döndür.\n"
           'Yalnızca JSON döndür: {"ilgili": [{"id": "kN", "neden": <neden>}]}\nneden: ' + ", ".join(REASONS) + ".")
    usr = "Soru: " + question + "\n\nEski kayıt kataloğu:\n" + cat
    return [dict(role="system", content=sys), dict(role="user", content=usr)]
def selector_schema(ids):
    return {"type": "json_schema", "json_schema": {"name": "secim", "schema": {
        "type": "object", "additionalProperties": False, "required": ["ilgili"],
        "properties": {"ilgili": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "required": ["id", "neden"], "properties": {"id": {"type": "string", "enum": ids or [""]},
                                                       "neden": {"type": "string", "enum": REASONS}}}}}}}}
def parse_selector(txt, ids):
    """-> [(id, neden)] veya None (None => yedek: tüm eski kayıtlar)."""
    if not txt: return None
    s = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    j = None
    for m in [s] + re.findall(r"\{.*\}", s, re.S):
        try: j = json.loads(m); break
        except Exception: continue
    if not isinstance(j, dict) or not isinstance(j.get("ilgili"), list): return None
    out, seen = [], set()
    for e in j["ilgili"]:
        if isinstance(e, str): e = {"id": e, "neden": None}
        if not isinstance(e, dict): continue
        k = str(e.get("id", "")).strip("[] ")
        if k in ids and k not in seen:
            seen.add(k); out.append((k, e.get("neden") if e.get("neden") in REASONS else None))
    return out

def build_context(user, sel):
    """sel: [(id, neden)] veya None (yedek => tüm eski kalemler, etiketsiz = B bağlamı)."""
    pre, secs, tail, plan, items = analyze(user)
    if sel is None: return user, dict(fallback=True, n_old=len(items), n_sel=len(items))
    want = dict(sel); out = []
    for h, rows in plan:
        lines = []
        for kind, l, k in rows:
            if kind == "keep": lines.append(("x", l))
            elif kind == "sub": lines.append(("s", l))
            elif kind == "old" and k in want:
                lab = " [olası ilgi: %s]" % LABEL[want[k]] if want[k] else " [olası ilgi]"
                lines.append(("x", l + lab))
            elif kind == "oldgrp":
                for kk in k:
                    if kk in want:
                        it = next(i for i in items if i["id"] == kk)
                        lab = " [olası ilgi: %s]" % LABEL[want[kk]] if want[kk] else " [olası ilgi]"
                        lines.append(("x", it["text"] + lab))
        # içi boşalan alt başlıkları at
        res = []
        for i, (t, l) in enumerate(lines):
            if t == "s":
                nxt = next((x for x in lines[i + 1:] if x[0] == "s" or x[1].startswith("- ")), None)
                if not nxt or nxt[0] == "s": continue
            res.append(l)
        body = "\n".join(res).strip("\n")
        if not body.strip(): continue
        out.append((h, body + "\n\n"))
    n_hidden = len(items) - len(want)
    if items:
        out.append(("Gösterilmeyen eski kayıtlar", "Daha eski diğer kayıtlar: %d (6 aydan eski; bu soruyla ilgili görülmedi, gösterilmedi).\n\n" % n_hidden))
    return join_user(pre, out, tail), dict(fallback=False, n_old=len(items), n_sel=len(want), n_hidden=n_hidden)
