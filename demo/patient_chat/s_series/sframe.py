"""Soruya duyarlı bağlam (Step 1) + plan ayrıştırma (Step 2). Yalnızca stdlib.
Kurallar CCO'nun YAPISINA dayanır (## bölüm başlıkları, 'Geçmiş kayıtlar' alt blokları);
hastalık/ilaç/soru içeriğine bağlı anahtar kelime yoktur."""
import json, re
QTYPES = ["symptom_complaint", "record_lookup", "medication", "wearable_sleep", "lab", "general_info", "other"]
ALWAYS = {"Zaman özeti", "Veri durumu", "Bilinmeyenler", "Alerjiler (hasta beyanı)", "Hastalıklar (hasta beyanı)"}
DIAG = "Tanılar (hastane kaydı)"
PAST_FORCED = {"record_lookup", "lab"}   # kayıt sorgusu türlerinde geçmiş bloklar her zaman açık
PAST_HDR = "Geçmiş kayıtlar"

def parse_user(u):
    """-> (pre, [(başlık, gövde)], soru_bloğu)  ; tekrar birleştirme kayıpsızdır."""
    i = u.rfind("\nHasta Sorusu:")
    body, tail = (u[:i], u[i:]) if i >= 0 else (u, "")
    parts = re.split(r"(?m)^## ", body)
    secs = []
    for p in parts[1:]:
        h, _, g = p.partition("\n"); secs.append((h, g))
    return parts[0], secs, tail

def join_user(pre, secs, tail):
    return pre + "".join("## %s\n%s" % (h, g) for h, g in secs) + tail

def is_subhdr(l):
    s = l.strip()
    return bool(s) and not s.startswith("- ") and s.endswith(":") or s.startswith(PAST_HDR)

def split_blocks(g):
    """Bölüm gövdesini (geçmiş_mi, satırlar) bloklarına ayırır. Geçmiş blok 'Geçmiş kayıtlar' satırıyla
    başlar; sonraki alt başlıkta, boş satırda veya bölüm sonunda biter."""
    out, cur, past = [], [], False
    for l in g.split("\n"):
        if l.strip().startswith(PAST_HDR) or (past and (not l.strip() or is_subhdr(l))):
            if cur: out.append((past, cur))
            cur, past = [], l.strip().startswith(PAST_HDR)
        cur.append(l)
    if cur: out.append((past, cur))
    return out

def drop_past(g):
    bl = split_blocks(g); n = 0; keep = []
    for past, ls in bl:
        if past: n += sum(1 for l in ls if l.strip().startswith("- "))
        else: keep.append("\n".join(ls))
    s = "\n".join(keep).rstrip("\n")
    if n: s += "\nDaha eski kayıtlar: %d kayıt var (bu soru için gösterilmedi; bugünkü durumu yansıtmaz)." % n
    return s + "\n\n", n

def router_messages(question, sections):
    sys = ("Görevin yalnızca sınıflandırma. Hastanın sorusunu oku, cevap için hangi kayıt bölümlerine "
           "bakılması gerektiğini seç ve yalnızca JSON döndür, başka hiçbir şey yazma.\n"
           'Biçim: {"question_type": <tür>, "needed_sections": [<bölüm>...], "needs_past": true|false}\n'
           "question_type: " + ", ".join(QTYPES) + ".\n"
           "needed_sections: yalnızca aşağıda verilen bölüm adlarından, cevap için gerçekten gerekenler (boş olabilir).\n"
           "needs_past: kullanıcı geçmiş/eski kayıtları, önceki başvuruları veya bir şeyin geçmişini soruyorsa ya da "
           "cevap aylar/yıllar önceki kayıtlara bakmayı gerektiriyorsa true; bugünkü bir şikâyet veya genel soru için false.")
    usr = "Kayıttaki bölümler: " + json.dumps(sections, ensure_ascii=False) + "\nSoru: " + question
    return [dict(role="system", content=sys), dict(role="user", content=usr)]

def router_schema(sections):
    return {"type": "json_schema", "json_schema": {"name": "route", "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["question_type", "needed_sections", "needs_past"],
        "properties": {"question_type": {"type": "string", "enum": QTYPES},
                       "needed_sections": {"type": "array", "items": {"type": "string", "enum": sections or [""]}},
                       "needs_past": {"type": "boolean"}}}}}

def parse_router(txt, sections):
    """-> dict veya None (None => tam B bağlamı)."""
    if not txt: return None
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: return None
    try: j = json.loads(m.group(0))
    except Exception: return None
    if not isinstance(j, dict): return None
    qt = j.get("question_type"); ns = j.get("needed_sections"); np_ = j.get("needs_past")
    if qt not in QTYPES or not isinstance(ns, list) or not isinstance(np_, bool): return None
    return dict(question_type=qt, needed_sections=[s for s in ns if s in sections], needs_past=np_,
                dropped_unknown=[s for s in ns if s not in sections])

def selectable(secs):
    return [h for h, _ in secs if h not in ALWAYS]

def build_context(user, route):
    """Deterministik bağlam. route None ise B bağlamı aynen döner."""
    pre, secs, tail = parse_user(user)
    if route is None: return user, dict(fallback=True)
    past = route["needs_past"] or route["question_type"] in PAST_FORCED
    want = set(route["needed_sections"]) | ALWAYS | {DIAG}
    out, info = [], dict(fallback=False, past=past, kept=[], dropped=[], past_hidden=0)
    for h, g in secs:
        if h not in want: info["dropped"].append(h); continue
        if not past and h not in ("Zaman özeti",):
            g, n = drop_past(g); info["past_hidden"] += n
        out.append((h, g)); info["kept"].append(h)
    return join_user(pre, out, tail), info

PLAN_INSTR = (
    "\n\nCEVAP BİÇİMİ: Önce kısa bir plan yaz, sonra cevabı. Plan kullanıcıya gösterilmez:\n"
    '<plan>{"answer_first": "<ilk cümlede vereceğin doğrudan cevap, kısa>", '
    '"used_items": [{"item": "<kullanacağın kayıt öğesi>", "age": "<kaydın yaşı, ör. 3 gün önce>"}], '
    '"not_in_record": ["<soruyla ilgili ama kayıtta olmayan bilgi>"], '
    '"link_past_to_today": false}</plan>\n'
    "link_past_to_today yalnızca geçmiş bir kaydı bugünkü duruma bağlamak için kronik/kesin bir hastalık gibi somut bir "
    "neden varsa o nedeni yazan kısa bir metindir; yoksa false. used_items içinde yaşı aylar/yıllar olan akut öğe varsa ve "
    "link_past_to_today false ise o öğeyi cevapta kullanma. Plandan sonra yalnızca kullanıcıya yönelik Türkçe cevabı yaz.")

def split_plan(txt):
    """-> (cevap, plan_dict|None, plan_ham|None). Plan hiçbir zaman cevaba sızmaz; cevap metni değiştirilmez."""
    if txt is None: return None, None, None
    m = re.search(r"<plan>(.*?)</plan>", txt, re.S)
    if m:
        raw = m.group(1); ans = (txt[:m.start()] + txt[m.end():]).strip()
    elif "<plan>" in txt:            # kapanış etiketi yok: JSON'u ham çözümle bul
        k = txt.index("<plan>") + 6; s = txt[k:].lstrip(); off = len(txt) - len(s)
        try: _, e = json.JSONDecoder().raw_decode(s); raw = s[:e]; ans = (txt[:k-6] + txt[off+e:]).strip()
        except Exception: return txt.replace("<plan>", "").strip(), None, None
    elif txt.lstrip().startswith("{"):  # etiketsiz plan
        s = txt.lstrip()
        try: _, e = json.JSONDecoder().raw_decode(s); raw = s[:e]; ans = s[e:].strip()
        except Exception: return txt.strip(), None, None
    else: return txt.strip(), None, None
    ans = re.sub(r"^</?plan>\s*", "", ans).strip()
    try: pj = json.loads(raw.strip())
    except Exception: pj = None
    return ans, pj, raw
