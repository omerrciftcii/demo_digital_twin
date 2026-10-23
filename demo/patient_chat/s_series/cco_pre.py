"""Genel CCO ön işleyici. Hiçbir hasta/hastalık/ilaç/soru adına özel kural yoktur;
yalnızca tarih aritmetiği, kesinlik etiketleri ve bölüm tipleri kullanılır."""
import re, datetime as dt
# ---- yapılandırılabilir pencereler (gün) ----
COMPACT_PAST = False  # True: pencere dışındaki geçmiş vizitlerin serbest metni (Şikâyet/Bulgular/Epikriz) atılır
WIN = dict(diag_acute=30, diag_other=365, diag_chronic_pre=365, rx=90, visit=30,
           lab=90, imaging=180, pathology=180, home=14, wearable=2, declared=365)
D3 = re.compile(r'(\d{2})\.(\d{2})\.(\d{4})'); D2 = re.compile(r'\((\d{2})\.(\d{2})(?:[ )–;,])')
MY = re.compile(r'beyan (\d{2})\.(\d{4})')
def age(ref, d):
    n = (ref - d).days
    if n < 0: return 'ileri tarih'
    if n == 0: return 'bugün'
    if n < 120: return f'{n} gün önce'
    m = round(n / 30.44)
    return f'{m} ay önce' if m < 24 else f'{round(n/365.25)} yıl önce'
def p3(s):
    m = D3.search(s); return dt.date(int(m[3]), int(m[2]), int(m[1])) if m else None
def p2(s, ref):
    m = D2.search(s)
    if not m: return None
    d = dt.date(ref.year, int(m[2]), int(m[1]))
    return d if d <= ref else dt.date(ref.year - 1, d.month, d.day)
def ann(line, d, ref):  # tarihin hemen arkasına yaş ekle (yalnızca ilk tarih)
    if d is None: return line
    m = D3.search(line)
    return line[:m.end()] + f' [{age(ref, d)}]' + line[m.end():] if m else line + f' [{age(ref, d)}]'
WEAR = ('Uyku', 'Kalp', 'Solunum', 'Hareket', 'Stres')
def stype(h):
    for k, t in [('Tanılar', 'diag'), ('İlaçlar (reçete', 'rx'), ('Hastane başvuru', 'visit'), ('Laboratuvar', 'lab'),
                 ('Görüntüleme', 'imaging'), ('Patoloji', 'pathology'), ('Ameliyat', 'surgery'),
                 ('Tansiyon', 'home'), ('Kan şekeri', 'home'), ('Vücut ölçüleri', 'home'),
                 ('hasta beyanı', 'declared'), ('Veri durumu', 'status'), ('Bilinmeyenler', 'unknown')]:
        if k in h: return t
    if h.startswith(WEAR): return 'wear'
    return 'other'
def split_cco(c):
    lines = c.split('\n'); head = []; secs = []
    for l in lines:
        if l.startswith('## '): secs.append([l[3:], []])
        elif secs: secs[-1][1].append(l)
        else: head.append(l)
    return head, secs
STAT = dict(known=0, passed=0, unknown_fmt=[])
def mark(ok, l):
    if not l.strip(): return
    if ok: STAT['known'] += 1
    else: STAT['passed'] += 1  # demo: format listesi tutulmaz (bellek)
DIAG = re.compile(r'^- (.+?) — (?:(ön tanı|kesin tanı); )?son kayıt (\d{2}\.\d{2}\.\d{4})(.*)$')
def do_diag(body, ref):
    grp = None; buckets = {k: [] for k in ('kron', 'cur_k', 'cur_o', 'past_k', 'past_o')}; extra = []
    for l in body:
        if re.match(r'^(Akut|Kronik|Diğer):$', l): grp = l[:-1]; mark(1, l); continue
        m = DIAG.match(l)
        if not m: mark(0, l); extra.append(l); continue
        mark(1, l); d = p3(m[3]); n = (ref - d).days; pre = m[2] == 'ön tanı'
        txt = f'- {m[1]} — {m[2] or "kesinlik belirtilmemiş"}; son kayıt {m[3]} [{age(ref, d)}]{m[4]}'
        if grp == 'Kronik' and not pre: buckets['kron'].append(txt); continue
        w = WIN['diag_acute'] if grp == 'Akut' else WIN['diag_chronic_pre'] if grp == 'Kronik' else WIN['diag_other']
        buckets[('cur_' if n <= w else 'past_') + ('o' if pre else 'k')].append(txt)
    out = []
    T = [('kron', 'Kronik kesin tanılar (kayıt tarihinden bağımsız süregelen):'),
         ('cur_k', 'Yakın dönem kesin tanılar:'),
         ('cur_o', 'Yakın dönem ÖN TANILAR (doğrulanmamış şüphe; kesin tanı değil):'),
         ('past_k', 'Geçmiş kayıtlar (bugünkü durumu yansıtmaz) — eski kesin tanılar:'),
         ('past_o', 'Geçmiş kayıtlar (bugünkü durumu yansıtmaz) — eski ÖN TANILAR (doğrulanmamış):')]
    for k, t in T:
        if buckets[k]: out += [t] + buckets[k]
    return out + extra
def do_dated_items(body, ref, win, kind):
    """Madde satırlarını tarihe göre güncel/geçmiş ayırır; alt başlıklar ve diğer satırlar korunur."""
    cur, past, other = [], [], []
    for l in body:
        if l.startswith('- '):
            d = p3(l) or p2(l, ref)
            if d:
                mark(1, l); t = ann(l, d, ref)
                if (ref - d).days <= win: cur.append(t)
                else: past.append(compact(t) if COMPACT_PAST and kind == 'visit' else t)
                continue
            mark(0, l); other.append(l); continue
        if l.startswith('Daha eski muayeneler:'):
            mark(1, l); past.append('- ' + (compact_older(l) if COMPACT_PAST and kind == 'visit' else l)); continue
        if re.match(r'^[^-].*:$', l): mark(1, l); continue  # alt grup başlıkları (Kronik takip/Diğer tahliller/Son 5 muayene)
        mark(bool(D3.search(l) or not l.strip()), l)
        if l.strip(): other.append(ann(l, p3(l), ref) if p3(l) and '[' not in l else l)
    out = list(other)
    if cur: out += [f'Güncel (son {win} gün):'] + cur
    elif past: out += [f'Son {win} günde kayıt yok.']
    if past: out += ['Geçmiş kayıtlar (bugünkü durumu yansıtmaz):'] + past
    return out
def compact(t):
    # '- tarih [yaş] saat, Branş, tür — serbest metin' -> serbest metin atılır
    return t.split(' — ', 1)[0] + '.'
def compact_older(l):
    # 'Daha eski muayeneler: tarih Branş, tür: metin · ...' -> her öğede ':' sonrası serbest metin atılır
    head, _, rest = l.partition(': ')
    items = [i.split(': ', 1)[0].strip().rstrip('.') for i in rest.split(' · ')]
    return head + ': ' + ' · '.join(items) + '.'
def wear_latest(secs, ref):
    best = None
    for h, b in secs:
        t = stype(h)
        if t == 'status':
            for l in b:
                if 'Yüzük' in l or 'saat' in l.lower():
                    d = p3(l); best = max(best, d) if best and d else (d or best)
        if t == 'wear':
            for l in b:
                for m in re.finditer(r'\((\d{2})\.(\d{2})', l):
                    d = dt.date(ref.year, int(m[2]), int(m[1]))
                    if d > ref: d = dt.date(ref.year - 1, d.month, d.day)
                    best = max(best, d) if best else d
    return best
def process(cco, ref):
    head, secs = split_cco(cco)
    for l in head: mark(1, l)
    present = {stype(h) for h, _ in secs}
    out = list(head)
    wl = wear_latest(secs, ref); has_wear = any(stype(h) == 'wear' for h, _ in secs)
    # zaman özeti
    zs = [f'## Zaman özeti', f'Bugün: {ref.strftime("%d.%m.%Y")}. Tüm "[... önce]" yaşları bu tarihe göredir.']
    lastnight = None
    for h, b in secs:
        if h.startswith('Uyku'):
            for l in b:
                m = re.match(r'Son gece \((\d{2}\.\d{2})\)', l)
                if m: lastnight = p2('(' + m[1] + ')', ref)
    if has_wear and wl:
        n = (ref - wl).days
        zs.append(f'Giyilebilir cihazdan en son veri: {wl.strftime("%d.%m.%Y")} [{age(ref, wl)}]' +
                  (' — GÜNCEL.' if n <= WIN['wearable'] else ' — ESKİ; bu tarihten sonrası bilinmiyor, bugünkü durumu yansıtmaz.'))
        if lastnight:
            ok = (ref - lastnight).days <= 1
            zs.append(f'Dün geceye ait uyku verisi: {"VAR" if ok else "YOK"} (kayıtlı son gece {lastnight.strftime("%d.%m.%Y")}, {age(ref, lastnight)}).')
        else: zs.append('Dün geceye ait uyku verisi: YOK (kayıtta uyku gecesi yok).')
    miss = []
    for t, name in [('diag', 'hastane tanısı'), ('rx', 'reçete'), ('visit', 'hastane başvurusu'), ('lab', 'laboratuvar/tahlil sonucu'),
                    ('imaging', 'görüntüleme'), ('wear', 'giyilebilir cihaz verisi')]:
        if t not in present: miss.append(name)
    hn = {h for h, _ in secs}
    for k, name in [('Tansiyon', 'evde tansiyon ölçümü'), ('Kan şekeri', 'evde kan şekeri ölçümü'), ('Vücut ölçüleri', 'kilo/vücut ölçüsü')]:
        if not any(h.startswith(k) for h in hn): miss.append(name)
    if miss: zs.append('Kayıtta olmayanlar: ' + ', '.join(miss) + ' — bu konularda değer/tarih verilemez, uydurulmamalı.')
    out += [*zs]
    for h, b in secs:
        t = stype(h); out += ['', '## ' + h]; mark(1, '## ' + h)
        b = [l for l in b]
        while b and not b[-1].strip(): b.pop()
        if t == 'diag': out += do_diag(b, ref)
        elif t in ('rx', 'visit', 'lab', 'imaging', 'pathology'): 
            pre = []
            if b and not b[0].startswith('-') and not b[0].endswith(':') and t != 'pathology':
                pre = [b[0]]; mark(1, b[0]); b = b[1:]
            out += pre + do_dated_items(b, ref, WIN[t], t)
        elif t == 'surgery':
            for l in b:
                d = p3(l); mark(bool(d) or l.endswith(':') or not l.strip(), l); out.append(ann(l, d, ref) if d else l)
        elif t == 'declared':
            for l in b:
                ms = list(MY.finditer(l)); mark(bool(ms) or not l.strip(), l)
                for m in reversed(ms):
                    d = dt.date(int(m[2]), int(m[1]), 15); l = l[:m.end()] + f' [{age(ref, d) if (ref-d).days > 31 else "bu ay"}]' + l[m.end():]
                out.append(l)
        elif t == 'home':
            d0 = None
            for l in b:
                d = p3(l) or p2(l, ref); d0 = d0 or d; mark(bool(d) or not l.strip(), l)
            if d0:
                n = (ref - d0).days
                out.append(f'(Son ölçüm {d0.strftime("%d.%m.%Y")} [{age(ref, d0)}]' + (' — güncel)' if n <= WIN['home'] else ' — ESKİ, bugünkü değeri yansıtmaz)'))
            out += b
        elif t == 'wear':
            for l in b:
                d = p2(l, ref); mark(bool(d) or not l.strip() or bool(re.search(r'\d', l)), l)
                l = re.sub(r'^Son gece \((\d{2}\.\d{2})\)', lambda m: f'Kayıtlı son gece ({m[1]}, {age(ref, p2("(" + m[1] + ")", ref))})', l)
                out.append(l)
            if wl and (ref - wl).days > WIN['wearable']:
                out.append(f'(Bu bölümdeki veriler en geç {wl.strftime("%d.%m")} tarihli; {age(ref, wl)}. Bugünkü durumu yansıtmaz.)')
        else:
            for l in b: mark(t in ('status', 'unknown') and (bool(re.search(r'\d', l)) or l.startswith('- ') or not l.strip()), l)
            out += b
    return '\n'.join(out)
def rewrite_user(u, ref):
    pre, rest = u.split('Klinik Bağlam (CCO):\n', 1); cco, q = rest.split('\n\nHasta Sorusu:', 1)
    return pre + 'Klinik Bağlam (CCO):\n' + process(cco, ref) + '\n\nHasta Sorusu:' + q
def ref_of(row): return dt.date.fromisoformat(row['as_of'][:10])
