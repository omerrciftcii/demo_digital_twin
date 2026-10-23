"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    snapshots: [], snapshot: null, health: null, packet: null, history: [],
    source: null, sourceLoaded: false, busy: false, controller: null,
    generation: 0, tab: "selected", healthController: null, sourceRequest: 0,
  };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function pretty(value) {
    return typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? "—";
  }

  function date(value) {
    if (!value) return "Tarih belirtilmemiş";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleDateString("tr-TR", { day: "2-digit", month: "short", year: "numeric" });
  }

  function concise(value, length = 30) {
    const text = String(value ?? "");
    return text.length > length ? `${text.slice(0, length - 1)}…` : text;
  }

  function errorMessage(detail) {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((item) => typeof item === "object" ? `${item.loc?.join(".") || "Kayıt"}: ${item.msg || pretty(item)}` : String(item)).join("\n");
    if (detail && typeof detail === "object") return detail.message || detail.reason || detail.error || pretty(detail);
    return "İstek tamamlanamadı. Yerel sunucunun çalıştığını kontrol edin.";
  }

  async function api(path, options = {}) {
    const response = await fetch(path, { credentials: "same-origin", cache: "no-store", ...options, headers: { "Accept": "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers } });
    let data;
    try { data = await response.json(); } catch { throw new Error(`Sunucudan okunabilir bir JSON yanıtı alınamadı (HTTP ${response.status}).`); }
    if (!response.ok) throw new Error(errorMessage(data.detail ?? data));
    return data;
  }

  function alert(message = "") {
    $("global-alert").textContent = message;
    $("global-alert").hidden = !message;
  }

  function isSynthetic(snapshot) { return snapshot?.dataClass === "synthetic"; }

  function chatBlockReason() {
    if (!state.snapshot) return "Önce bir CCO seçin.";
    if (!state.health?.llm?.configured) return "Model bağlantısı yapılandırılmamış. Kanıt seçimi yerelde kullanılabilir.";
    if (!isSynthetic(state.snapshot) && !state.health?.llm?.externalRealDataAllowed) return "Gerçek sağlık verisinin dış model API’sine aktarımı kapalı. Yerel kanıt seçimi kullanılabilir.";
    return "";
  }

  function controls() {
    const ready = Boolean(state.snapshot) && !state.busy;
    const hasQuestion = Boolean($("question").value.trim());
    $("question").disabled = !ready;
    $("select-button").disabled = !ready || !hasQuestion;
    $("answer-button").disabled = !ready || !hasQuestion || Boolean(chatBlockReason());
    $("answer-button").title = chatBlockReason();
    $("clear-chat").disabled = state.busy || !state.snapshot;
    $("import-button").disabled = state.busy;
    $("reload-button").disabled = state.busy;
    $("show-full-source").disabled = !state.snapshot || state.busy;
  }

  function setBusy(busy, label = "") {
    state.busy = busy;
    $("activity").hidden = !busy;
    $("activity-label").textContent = label;
    controls();
  }

  function showPrivacy() {
    const notice = $("privacy-notice");
    notice.hidden = !state.snapshot;
    if (!state.snapshot) return;
    const reason = chatBlockReason();
    notice.className = reason ? "notice" : "notice good";
    notice.textContent = reason || (isSynthetic(state.snapshot)
      ? "Sentetik kayıt · Yanıt üretirken soru, sohbet metni ve seçilen kanıtlar model sağlayıcısına gönderilir. Bu moda gerçek hasta bilgisi yazmayın; serbest metin otomatik anonimleştirilmez. Kanıtları inceleme işlemi yereldir."
      : "Dış modele gerçek veri aktarımı bu sunucuda açılmış. Yanıt üretirken soru, sohbet bağlamı ve seçilen kanıtlar dış sağlayıcıya gönderilir.");
  }

  async function loadHealth() {
    state.healthController?.abort();
    const controller = new AbortController();
    state.healthController = controller;
    try {
      state.health = await api("/health", { signal: controller.signal });
      $("health-dot").className = `status-dot ${state.health.status === "ok" ? "online" : "degraded"}`;
      $("health-label").textContent = state.health.status === "ok" ? "Yerel servis hazır" : "Yerel servis · kısıtlı bağlantı";
      const details = $("service-details-body");
      details.replaceChildren();
      const graph = state.health.neo4j;
      details.append(element("p", "", `Neo4j: ${typeof graph === "object" ? (graph.available ? "bağlı" : graph.status || "kullanılamıyor") : graph || "bilinmiyor"}`));
      details.append(element("p", "", `Model: ${state.health.llm?.model || "yapılandırılmamış"}`));
      details.append(element("p", "", `Model bağlantısı: ${state.health.llm?.configured ? "yapılandırılmış" : "kapalı"}`));
      details.append(element("p", "", `Gerçek veri aktarımı: ${state.health.llm?.externalRealDataAllowed ? "açık" : "kapalı"}`));
      if (state.health.index) {
        const indexDetails = element("details", "metadata-details");
        indexDetails.append(element("summary", "", "Yerel indeks durumu"), element("pre", "metadata-pre", pretty(state.health.index)));
        details.append(indexDetails);
      }
    } catch (error) {
      if (error.name === "AbortError") return;
      state.health = null;
      $("health-dot").className = "status-dot error";
      $("health-label").textContent = "Yerel servis erişilemiyor";
      $("service-details-body").textContent = error.message;
    }
    controls();
    showPrivacy();
  }

  function renderSnapshots() {
    const list = $("snapshot-list");
    list.replaceChildren();
    $("snapshot-count").textContent = String(state.snapshots.length);
    const search = $("snapshot-search").value.trim().toLocaleLowerCase("tr-TR");
    const filtered = state.snapshots.filter((item) => `${item.subjectId} ${item.sourceFile} ${item.snapshotId}`.toLocaleLowerCase("tr-TR").includes(search));
    if (!filtered.length) {
      list.append(element("div", "empty-state", state.snapshots.length ? "Aramanızla eşleşen kayıt yok." : "Henüz CCO kaydı yok. Yerel bir JSON dosyası ekleyin."));
      return;
    }
    filtered.forEach((snapshot) => {
      const selected = snapshot.snapshotId === state.snapshot?.snapshotId;
      const button = element("button", `snapshot-button${selected ? " selected" : ""}`);
      button.type = "button";
      button.setAttribute("aria-pressed", String(selected));
      button.title = `${snapshot.subjectId}\n${snapshot.sourceFile || snapshot.snapshotId}`;
      button.append(element("span", "snapshot-title", `Hasta ${concise(snapshot.subjectId, 26)}`));
      button.append(element("span", "snapshot-file", snapshot.sourceFile || snapshot.snapshotId));
      const meta = element("span", "snapshot-meta");
      meta.append(element("span", "", date(snapshot.asOf)), element("span", `badge${isSynthetic(snapshot) ? "" : " real"}`, isSynthetic(snapshot) ? "Sentetik" : "Gerçek veri"));
      button.append(meta);
      button.addEventListener("click", () => selectSnapshot(snapshot));
      list.append(button);
    });
  }

  async function loadSnapshots(preferredId = null) {
    const result = await api("/api/v1/ccos");
    state.snapshots = Array.isArray(result) ? result : result.snapshots || [];
    if (preferredId) {
      const preferred = state.snapshots.find((item) => item.snapshotId === preferredId);
      if (preferred) selectSnapshot(preferred);
    } else if (state.snapshot && !state.snapshots.some((item) => item.snapshotId === state.snapshot.snapshotId)) {
      selectSnapshot(null);
    }
    renderSnapshots();
  }

  function resetMessages() {
    $("messages").replaceChildren();
    const welcome = element("div", "welcome");
    welcome.append(element("span", "welcome-icon", "⌁"));
    welcome.append(element("h3", "", state.snapshot ? "Bu kayda ne sormak istersiniz?" : "Önce bir CCO seçin."));
    welcome.append(element("p", "", "Önce kanıtları inceleyebilir, sonra kaynaklarıyla birlikte yanıt üretebilirsiniz. CCO’da bulunmayan geçmiş bilgileri bu araç tamamlamaz."));
    $("messages").append(welcome);
    state.history = [];
  }

  function resetEvidence() {
    state.sourceRequest += 1;
    state.packet = null;
    state.source = null;
    state.sourceLoaded = false;
    $("packet-meta").hidden = true;
    $("evidence-count").textContent = "—";
    $("view-selected").replaceChildren(element("div", "empty-state", "Seçim yaptığınızda kaynaklar, zaman bilgisi ve kapsam uyarıları burada görünür."));
    $("view-excluded").replaceChildren(element("div", "empty-state", "Henüz bir seçim yapılmadı."));
    $("source-json").textContent = "Bir kanıtın kaynak bağlantısını açın.";
    $("source-pointer").textContent = "Orijinal CCO";
    showTab("selected");
  }

  function selectSnapshot(snapshot) {
    if (snapshot?.snapshotId === state.snapshot?.snapshotId && snapshot !== null) return;
    state.generation += 1;
    state.controller?.abort();
    state.controller = null;
    state.snapshot = snapshot;
    setBusy(false);
    alert();
    $("question").value = "";
    resetMessages();
    resetEvidence();
    const active = $("active-snapshot");
    active.replaceChildren();
    active.hidden = !snapshot;
    if (snapshot) {
      active.append(element("span", "badge neutral", `CCO ${snapshot.schemaVersion || "—"}`));
      active.append(element("span", "", `${date(snapshot.asOf)} · ${snapshot.factCount ?? "—"} fact`));
      const identity = element("span", "snapshot-id", concise(snapshot.snapshotId, 24));
      identity.title = snapshot.snapshotId;
      active.append(identity);
    }
    renderSnapshots();
    controls();
    showPrivacy();
  }

  function showTab(tab, focus = false) {
    state.tab = tab;
    for (const name of ["selected", "excluded", "source"]) {
      const selected = name === tab;
      const button = $(`tab-${name}`);
      button.classList.toggle("active", selected);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
      $(`view-${name}`).hidden = !selected;
    }
    if (focus) $(`tab-${tab}`).focus();
  }

  function resolvePointer(documentValue, pointer) {
    let path = String(pointer || "");
    if (path.startsWith("#")) path = decodeURIComponent(path.slice(1));
    if (!path) return documentValue;
    if (!path.startsWith("/")) throw new Error("Kaynak yolu JSON Pointer biçiminde değil.");
    return path.slice(1).split("/").reduce((value, key) => {
      const decoded = key.replace(/~1/g, "/").replace(/~0/g, "~");
      if (value === null || typeof value !== "object" || !Object.prototype.hasOwnProperty.call(value, decoded)) throw new Error("Kaynak yolu bu CCO içinde bulunamadı.");
      return value[decoded];
    }, documentValue);
  }

  async function showSource(pointer = "") {
    if (!state.snapshot) return;
    const generation = state.generation;
    const snapshotId = state.snapshot.snapshotId;
    const sourceRequest = ++state.sourceRequest;
    showTab("source");
    $("source-pointer").textContent = pointer || "Orijinal CCO · tüm belge";
    $("source-json").textContent = "Kaynak okunuyor…";
    try {
      if (!state.sourceLoaded) {
        const source = await api(`/api/v1/ccos/${encodeURIComponent(snapshotId)}`);
        if (generation !== state.generation || sourceRequest !== state.sourceRequest) return;
        state.source = source;
        state.sourceLoaded = true;
      }
      if (generation !== state.generation || sourceRequest !== state.sourceRequest) return;
      $("source-json").textContent = pretty(resolvePointer(state.source, pointer));
    } catch (error) {
      if (generation === state.generation && sourceRequest === state.sourceRequest) $("source-json").textContent = error.message;
    }
  }

  function addMetadata(parent, name, value) {
    if (value === undefined || value === null) return;
    const details = element("details", "metadata-details");
    details.append(element("summary", "", name), element("pre", "metadata-pre", pretty(value)));
    parent.append(details);
  }

  function evidenceName(item) {
    if (item.label) return item.label;
    if (typeof item.definition === "string") return item.definition;
    return item.definition?.label || item.definition?.display || item.definition?.name || item.definitionId || item.section || "Kanıt";
  }

  const reasonLabels = {
    critical_safety: "Kritik güvenlik bilgisi; zorunlu bağlama alındı.",
    query_match: "Soruyla ilişkili kanıt bulundu.",
    longitudinal_same_definition: "Aynı konudaki önceki / sonraki kayıt; zaman içindeki değerlendirmeyi destekler.",
    insight_dependency: "Seçilen insight’ın dayanak kaydı; kaynak bütünlüğü için eklendi.",
    not_retrieved: "Bu soru için arama adayları arasına girmedi. Klinik olarak önemsiz olduğu anlamına gelmez.",
    access_withheld: "Erişim / onam kısıtı nedeniyle gösterilmiyor.",
    bundle_exceeds_budget: "Kanıt grubu, kaynak bütünlüğü korunarak token bütçesine sığmadı.",
    dependency_missing: "Kanıtın gerekli kaynak bağımlılıkları eksik.",
    semantic_relevance_unverified: "Anlamsal benzerlik bulundu, ancak soruyla ilgisi doğrulanamadı; cevap bağlamına alınmadı.",
  };

  function reasonText(value) {
    if (Array.isArray(value)) return value.map(reasonText).join(" · ");
    return reasonLabels[value] || pretty(value);
  }

  function renderPacket(packet) {
    state.packet = packet;
    const selected = $("view-selected");
    const excluded = $("view-excluded");
    selected.replaceChildren();
    excluded.replaceChildren();
    const evidence = Array.isArray(packet.evidence) ? packet.evidence : [];
    $("evidence-count").textContent = String(evidence.length);
    $("packet-meta").hidden = false;
    $("packet-meta").textContent = `Paket ${concise(packet.packetId, 18)} · ${date(packet.asOf)}${packet.tokenEstimate ? ` · ≈${packet.tokenEstimate} token` : ""}${packet.canAnswer === false ? " · Yeterli kanıt yok" : ""}`;

    const warnings = Array.isArray(packet.warnings) ? packet.warnings : packet.warnings ? [packet.warnings] : [];
    if (warnings.length || packet.canAnswer === false) {
      const box = element("section", "coverage-box warning");
      box.append(element("h3", "", "Kapsam ve güvenlik uyarıları"));
      if (packet.canAnswer === false) box.append(element("p", "", "Bu paket tam bir yanıtı desteklemiyor. Kapsam eksiklerini inceleyin."));
      warnings.forEach((warning) => box.append(element("p", "", typeof warning === "string" ? warning : warning.message || pretty(warning))));
      addMetadata(box, "Yanıtı engelleyen durumlar", packet.blockers?.length ? packet.blockers : null);
      selected.append(box);
    }
    const coverage = element("section", "coverage-box");
    coverage.append(element("h3", "", "Kapsam: yalnız seçilen CCO"));
    coverage.append(element("p", "", "Kayıt bulunmaması, hastalığın veya geçmiş olayın olmadığı anlamına gelmez."));
    addMetadata(coverage, "Kapsam ayrıntıları", packet.coverage);
    addMetadata(coverage, "CCO kaynaklı önemli uyarılar", packet.warningIndex?.length ? packet.warningIndex : null);
    addMetadata(coverage, "Güvenlik bilgileri", packet.safety);
    addMetadata(coverage, "Yayın / inceleme durumu", packet.release);
    addMetadata(coverage, "Erişim / onam durumu", packet.consent);
    addMetadata(coverage, "Arama planı ve Neo4j durumu", packet.retrieval);
    addMetadata(coverage, "Seçim ölçümleri", packet.metrics);
    addMetadata(coverage, "Seçim sayıları", packet.counts);
    addMetadata(coverage, "Paket glossary’si", packet.glossary);
    selected.append(coverage);

    if (!evidence.length) selected.append(element("div", "empty-state", "Bu soru için bir kanıt seçilmedi."));
    evidence.forEach((item, index) => {
      const card = element("article", "evidence-card");
      card.id = `evidence-${index}`;
      const header = element("div", "evidence-header");
      header.append(element("span", "evidence-name", evidenceName(item)), element("span", "evidence-ref", item.ref || `#${index + 1}`));
      card.append(header);
      if (item.sourceKind || item.section) card.append(element("span", "badge neutral", [item.section, item.sourceKind].filter(Boolean).join(" · ")));
      card.append(element("div", "evidence-value", pretty(item.value ?? item.text)));
      const dates = element("div", "evidence-dates");
      if (item.occurredFrom || item.occurredTo) {
        dates.append(element("div", "", `Olay: ${item.occurredFrom ? date(item.occurredFrom) : "başlangıç bilinmiyor"}${item.occurredTo && item.occurredTo !== item.occurredFrom ? ` → ${date(item.occurredTo)}` : ""}`));
      } else dates.append(element("div", "", "Olay tarihi: bilinmiyor"));
      if (item.producedAt) dates.append(element("div", "", `Üretim: ${date(item.producedAt)} · olay tarihi değildir`));
      card.append(dates);
      card.append(element("div", "evidence-reason", `Seçilme nedeni: ${reasonText(item.selectedReason)}`));
      const sourceLink = element("button", "source-link", "Kaynak JSON kaydını aç ↗");
      sourceLink.type = "button";
      sourceLink.disabled = !item.sourcePointer;
      sourceLink.addEventListener("click", () => showSource(item.sourcePointer));
      card.append(sourceLink);
      selected.append(card);
    });

    const omitted = Array.isArray(packet.excluded) ? packet.excluded : [];
    if (!omitted.length) excluded.append(element("div", "empty-state", "Elenen kayıt bildirilmedi."));
    else {
      excluded.append(element("p", "muted", `${omitted.length} kayıt bağlam paketine alınmadı.`));
      omitted.forEach((item) => {
        const card = element("article", "evidence-card");
        const withheld = item.reason === "access_withheld";
        const header = element("div", "evidence-header");
        header.append(element("span", "evidence-name", withheld ? "Erişimi kısıtlı kayıt" : evidenceName(item)));
        header.append(element("span", "evidence-ref", item.ref || "Kayıt"));
        card.append(header);
        if (!withheld && item.sourceKind) card.append(element("span", "badge neutral", item.sourceKind));
        card.append(element("div", "evidence-reason", reasonText(item.reason)));
        if (!withheld && item.sourcePointer) {
          const link = element("button", "source-link", "Kaynak kaydı aç ↗");
          link.type = "button";
          link.addEventListener("click", () => showSource(item.sourcePointer));
          card.append(link);
        }
        excluded.append(card);
      });
    }
    showTab("selected");
  }

  function focusEvidence(ref, packet) {
    if (state.packet?.packetId !== packet.packetId) renderPacket(packet);
    showTab("selected");
    const index = (packet.evidence || []).findIndex((item) => item.ref === ref);
    const card = $(`evidence-${index}`);
    if (!card) return;
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    card.classList.remove("highlight");
    void card.offsetWidth;
    card.classList.add("highlight");
  }

  function metricsText(metrics = {}) {
    const output = [];
    const labels = { totalMs: "Toplam", elapsedMs: "Geçen süre", latencyMs: "Gecikme", retrievalMs: "Arama", selectionMs: "Seçim", llmMs: "Model", inputTokens: "Giriş tokenı", outputTokens: "Çıkış tokenı", promptTokens: "Giriş tokenı", completionTokens: "Çıkış tokenı", estimatedTokens: "Tahmini token", evidenceTokens: "Kanıt tokenı" };
    for (const key of ["totalMs", "elapsedMs", "latencyMs", "retrievalMs", "selectionMs", "llmMs"]) {
      if (typeof metrics[key] === "number") output.push(`${labels[key]}: ${(metrics[key] / 1000).toFixed(2)} sn`);
    }
    for (const key of ["inputTokens", "outputTokens", "promptTokens", "completionTokens", "estimatedTokens", "evidenceTokens"]) {
      if (typeof metrics[key] === "number") output.push(`${labels[key]}: ${metrics[key].toLocaleString("tr-TR")}`);
    }
    const stages = [
      { name: "Arama planı", value: metrics.planner },
      { name: "Yanıt", value: metrics.answer },
    ].filter((stage) => stage.value && (stage.value.usage || typeof stage.value.durationMs === "number"));
    for (const stage of stages) {
      if (typeof stage.value.durationMs === "number") output.push(`${stage.name}: ${(stage.value.durationMs / 1000).toFixed(2)} sn`);
    }
    if (stages.length) {
      output.push(`${stages.length} ölçülmüş model çağrısı`);
      if (metrics.planner?.error) output.push("Planlayıcı hatası · yerel arama kullanıldı");
    } else if (metrics.planner?.error) output.push("Planlayıcı hatası · yerel arama kullanıldı");
    const usages = stages.length ? stages.map((stage) => stage.value.usage).filter(Boolean) : metrics.usage ? [metrics.usage] : [];
    const tokenKeys = { prompt_tokens: "Toplam giriş tokenı", completion_tokens: "Toplam çıkış tokenı", total_tokens: "Toplam model tokenı" };
    for (const [key, label] of Object.entries(tokenKeys)) {
      const values = usages.map((usage) => usage[key]).filter((value) => typeof value === "number");
      if (values.length) {
        output.push(`${label}: ${values.reduce((sum, value) => sum + value, 0).toLocaleString("tr-TR")}${values.length < stages.length ? " (kısmi ölçüm)" : ""}`);
      }
    }
    return output;
  }

  function addMessage(role, text, options = {}) {
    $("messages").querySelector(".welcome")?.remove();
    const article = element("article", `message ${role}${options.error ? " error" : ""}`);
    article.append(element("div", "message-label", role === "user" ? "SİZ" : role === "assistant" ? "MEDICALAI · KAYNAKLI YANIT" : "YEREL İNCELEME"));
    article.append(element("div", "message-content", text));
    if (options.citations?.length && options.packet) {
      const citations = element("div", "citations");
      options.citations.forEach((citation) => {
        const ref = typeof citation === "string" ? citation : citation.ref;
        const button = element("button", "citation", typeof citation === "string" ? citation : citation.label || ref);
        button.type = "button";
        button.title = `Kanıtı incele: ${ref}`;
        button.addEventListener("click", () => focusEvidence(ref, options.packet));
        citations.append(button);
      });
      article.append(citations);
    }
    const metrics = [options.model, options.provider, ...metricsText(options.metrics)].filter(Boolean);
    if (metrics.length) {
      const row = element("div", "message-metrics");
      metrics.forEach((item) => row.append(element("span", "", item)));
      article.append(row);
    }
    if (options.validation) addMetadata(article, "Yanıt kontrolleri", options.validation);
    $("messages").append(article);
    $("messages").scrollTop = $("messages").scrollHeight;
  }

  async function run(mode) {
    const question = $("question").value.trim();
    if (state.busy || !state.snapshot || !question) return;
    if (mode === "chat" && chatBlockReason()) { alert(chatBlockReason()); return; }
    const generation = state.generation;
    const controller = new AbortController();
    state.controller = controller;
    alert();
    resetEvidence();
    addMessage("user", question);
    setBusy(true, mode === "chat" ? "Kanıtlar seçiliyor, ardından kaynaklı yanıt hazırlanıyor…" : "CCO içinde kanıtlar aranıyor ve kapsam kontrol ediliyor…");
    const startedAt = performance.now();
    try {
      const response = await api(mode === "chat" ? "/api/v1/chat" : "/api/v1/context/select", {
        method: "POST", signal: controller.signal,
        body: JSON.stringify({ snapshotId: state.snapshot.snapshotId, question, tokenBudget: 8000, history: state.history.slice(-6) }),
      });
      if (generation !== state.generation) return;
      const packet = mode === "chat" ? response.contextPacket : response;
      if (packet) renderPacket(packet);
      if (mode === "chat") {
        const answer = response.answer || "Bu istek için bir yanıt üretilmedi. Kapsam ve güvenlik bilgilerini inceleyin.";
        addMessage("assistant", answer, { citations: response.citations || [], packet, metrics: { totalMs: performance.now() - startedAt, ...response.metrics }, model: response.model, provider: response.provider, validation: response.validation });
        const answered = response.status === "answered" || response.status === "answered_unvalidated";
        if (answered) {
          state.history.push({ role: "user", content: question }, { role: "assistant", content: answer });
          state.history = state.history.slice(-6);
          $("question").value = "";
        }
        if (response.status && response.status !== "answered") {
          const labels = { answered_unvalidated: "UYARI: Model yanıtı kaynak / biçim kontrolünden geçmedi ve doğrulanmamış haliyle gösteriliyor (ENFORCE_ANSWER_VALIDATION=false)", insufficient_evidence: "Yanıt için yeterli kanıt yok", blocked: "Veri aktarımı güvenlik nedeniyle engellendi", clarification_required: "Sorunun netleştirilmesi gerekiyor", validation_failed: "Model yanıtı kaynak / biçim kontrolünden geçmedi", model_unavailable: "Model bağlantısı kullanılamıyor" };
          addMessage("system", `${labels[response.status] || response.status}. Bu sonuç, eksiksiz bir hasta değerlendirmesi değildir.`);
        }
      } else {
        const count = packet?.evidence?.length || 0;
        addMessage("system", `${count} kanıt seçildi. Kaynak izi bölümünde seçilen kayıtları, tarihleri ve kapsam eksiklerini inceleyin.${packet?.canAnswer === false ? " Bu paket tam bir yanıt için yeterli değil." : ""}`, { metrics: { totalMs: performance.now() - startedAt, ...packet?.metrics } });
      }
    } catch (error) {
      if (generation !== state.generation || error.name === "AbortError") return;
      addMessage("system", error.message, { error: true });
    } finally {
      if (generation === state.generation) {
        state.controller = null;
        setBusy(false);
        $("question").focus();
      }
    }
  }

  async function importFile(file) {
    if (!file || state.busy) return;
    if (file.size > 20 * 1024 * 1024) { alert("CCO dosyası 20 MB sınırını aşıyor."); return; }
    alert();
    setBusy(true, "CCO doğrulanıyor ve yerel indekse ekleniyor…");
    try {
      let cco;
      try { cco = JSON.parse(await file.text()); } catch { throw new Error("Dosya geçerli bir JSON değil."); }
      const result = await api("/api/v1/ccos", { method: "POST", body: JSON.stringify({ cco, fileName: file.name }) });
      await loadSnapshots(result.snapshotId);
      await loadHealth();
    } catch (error) { alert(error.message); }
    finally { setBusy(false); $("file-input").value = ""; }
  }

  $("question-form").addEventListener("submit", (event) => { event.preventDefault(); run("chat"); });
  $("select-button").addEventListener("click", () => run("select"));
  $("question").addEventListener("input", controls);
  $("question").addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); run("select"); } });
  $("snapshot-search").addEventListener("input", renderSnapshots);
  $("import-button").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", () => importFile($("file-input").files[0]));
  $("health-refresh").addEventListener("click", loadHealth);
  $("show-full-source").addEventListener("click", () => showSource(""));
  $("clear-chat").addEventListener("click", () => { resetMessages(); resetEvidence(); $("question").value = ""; controls(); });
  $("reload-button").addEventListener("click", async () => {
    if (state.busy) return;
    alert();
    setBusy(true, "Yerel CCO klasörü yeniden taranıyor…");
    try { await api("/api/v1/ccos/reload", { method: "POST" }); await loadSnapshots(); await loadHealth(); }
    catch (error) { alert(error.message); }
    finally { setBusy(false); }
  });
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => showTab(button.dataset.tab));
    button.addEventListener("keydown", (event) => {
      const tabs = ["selected", "excluded", "source"];
      let index = tabs.indexOf(state.tab);
      if (event.key === "ArrowRight") index = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") index = (index + tabs.length - 1) % tabs.length;
      else if (event.key === "Home") index = 0;
      else if (event.key === "End") index = tabs.length - 1;
      else return;
      event.preventDefault();
      showTab(tabs[index], true);
    });
  });
  Promise.all([loadHealth(), loadSnapshots().catch((error) => alert(error.message))]);
})();
