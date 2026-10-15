(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    patients: [],
    models: [],
    activePatient: null,
    activeModel: null,
    record: null,
    history: [],
    busy: false,
    mode: "stepwise",
    recordTab: "readable",
    highlight: null,
  };

  const starterQuestions = [
    ["Öne çıkan klinik riskler", "Bu hastanın son kayıtlarında hekim açısından öncelikli değerlendirilmesi gereken bulgular neler?"],
    ["Son laboratuvarlar", "En güncel laboratuvar sonuçlarında dikkat çeken değerler neler ve kayıtlar ne kadar güncel?"],
    ["Klinik seyir", "Son muayene ve ölçümlere göre hastanın klinik seyrini kısa ve dengeli biçimde özetler misin?"],
    ["Wearable verileri", "Son wearable verileri klinik kayıtlarla birlikte değerlendirildiğinde dikkat çeken bir durum var mı?"],
  ];

  const statusLabels = {
    ready: "Hazır",
    offline: "Sunucu kapalı",
    setup: "Kurulum gerekli",
    locked: "Dış aktarım kapalı",
  };

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
  }

  function inlineMarkdown(value) {
    return escapeHtml(value)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function markdownToHtml(markdown) {
    const lines = String(markdown).replace(/\r\n/g, "\n").split("\n");
    const output = [];
    let list = null;
    const closeList = () => {
      if (list) output.push(`</${list}>`);
      list = null;
    };
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) { closeList(); continue; }
      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        closeList();
        const level = Math.min(3, heading[1].length + 1);
        output.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
        continue;
      }
      const bullet = line.match(/^[-*]\s+(.+)$/);
      if (bullet) {
        if (list !== "ul") { closeList(); list = "ul"; output.push("<ul>"); }
        output.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
        continue;
      }
      const numbered = line.match(/^\d+[.)]\s+(.+)$/);
      if (numbered) {
        if (list !== "ol") { closeList(); list = "ol"; output.push("<ol>"); }
        output.push(`<li>${inlineMarkdown(numbered[1])}</li>`);
        continue;
      }
      closeList();
      output.push(`<p>${inlineMarkdown(line)}</p>`);
    }
    closeList();
    return output.join("");
  }

  function extractQuickReplies(markdown) {
    const lines = String(markdown).replace(/\r\n/g, "\n").split("\n");
    const optionPattern = /^(?:[-*]\s*)?([A-Fa-f])[).]\s*(.+)$/;
    let activeRun = [];
    let bestRun = [];
    for (let index = 0; index < lines.length; index += 1) {
      const trimmed = lines[index].trim();
      if (!trimmed) continue;
      const match = trimmed.match(optionPattern);
      if (!match) { activeRun = []; continue; }
      const letter = match[1].toUpperCase();
      const expected = String.fromCharCode(65 + activeRun.length);
      if (letter === "A") activeRun = [{ index, text: match[2].trim() }];
      else if (activeRun.length && letter === expected) activeRun.push({ index, text: match[2].trim() });
      else activeRun = [];
      if (activeRun.length > bestRun.length) bestRun = [...activeRun];
    }
    if (bestRun.length < 2) return { markdown, options: [] };
    const optionLines = new Set(bestRun.map((item) => item.index));
    return {
      markdown: lines.filter((_, index) => !optionLines.has(index)).join("\n").trim(),
      options: bestRun.map((item) => item.text),
    };
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail : "İstek tamamlanamadı.";
      throw new Error(detail);
    }
    return payload;
  }

  function activeModel() {
    return state.models.find((model) => model.id === state.activeModel) || null;
  }

  function updateConnection() {
    const model = activeModel();
    const ready = model?.status === "ready";
    $("status-dot").className = `status-dot ${ready ? "ready" : model ? "error" : ""}`;
    $("status-text").textContent = model
      ? `${model.shortLabel} · ${statusLabels[model.status] || model.status}`
      : "Model bulunamadı";
    $("model-note").textContent = model
      ? `${model.note}${model.privacy === "external" ? " · Anonim veri dış servise gönderilir." : " · Veri kurum içinde kalır."}`
      : "";
    $("activity-text").textContent = model
      ? `${model.shortLabel} hasta kaydını değerlendiriyor…`
      : "Model hasta kaydını değerlendiriyor…";
  }

  function updateControls() {
    const ready = activeModel()?.status === "ready";
    const blocked = state.busy || !state.activePatient || !ready;
    $("activity").hidden = !state.busy;
    $("question").disabled = blocked;
    $("send-button").disabled = blocked;
    $("model-select").disabled = state.busy;
    document.querySelectorAll(".quick-replies button,.multi-replies button,.multi-replies input,.starter-prompts button,.patient-button,.mode-button,.featured-model")
      .forEach((control) => {
        const unavailableModel = control.classList.contains("featured-model") && control.dataset.status !== "ready";
        control.disabled = state.busy || unavailableModel;
      });
  }

  function setBusy(busy) {
    state.busy = busy;
    updateControls();
  }

  function scrollToBottom() {
    requestAnimationFrame(() => { $("messages").scrollTop = $("messages").scrollHeight; });
  }

  function openPanel(panel) {
    document.body.classList.remove("patients-open", "record-open");
    document.body.classList.add(panel === "patients" ? "patients-open" : "record-open");
    $("panel-backdrop").hidden = false;
  }

  function closePanels() {
    document.body.classList.remove("patients-open", "record-open");
    $("panel-backdrop").hidden = true;
  }

  function showReference(reference) {
    state.highlight = { start: reference.start, end: reference.end };
    setRecordTab("readable");
    renderReadableRecord();
    $("source-banner").hidden = false;
    if (window.matchMedia("(max-width: 1120px)").matches) openPanel("record");
    requestAnimationFrame(() => {
      const target = $("readable-record").querySelector(".source-block.highlighted");
      target?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  function addReferences(article, references) {
    if (!Array.isArray(references) || !references.length) return;
    const group = document.createElement("div");
    group.className = "source-links";
    const label = document.createElement("span");
    label.textContent = "CCO EŞLEŞMELERİ";
    group.appendChild(label);
    references.forEach((reference, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "source-link";
      button.textContent = `Kaynak ${index + 1}`;
      button.title = reference.sourceText || "Kaynakta göster";
      button.addEventListener("click", () => showReference(reference));
      group.appendChild(button);
    });
    article.appendChild(group);
  }

  function addMessage(role, content, meta = {}) {
    $("welcome")?.remove();
    const article = document.createElement("article");
    article.className = `message ${role}${meta.error ? " error" : ""}`;
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = role === "user" ? "SORU" : role === "assistant" ? "MODEL YANITI" : "SİSTEM";
    const body = document.createElement("div");
    body.className = "message-content";

    if (role === "assistant") {
      const parseOptions = state.mode === "stepwise" || meta.allowMultiSelect;
      const parsed = parseOptions ? extractQuickReplies(content) : { markdown: content, options: [] };
      body.innerHTML = markdownToHtml(parsed.markdown);
      article.append(label, body);
      if (parsed.options.length && state.mode === "multi_select") {
        const form = document.createElement("div");
        form.className = "multi-replies";
        const hint = document.createElement("p");
        hint.className = "multi-hint";
        hint.textContent = "Uyan bir veya daha fazla seçeneği işaretleyin.";
        const choices = document.createElement("div");
        choices.className = "multi-options";
        const submit = document.createElement("button");
        submit.type = "button";
        submit.className = "multi-submit";
        submit.textContent = "Seçilenleri değerlendir";
        submit.disabled = true;
        parsed.options.slice(0, 6).forEach((option, index) => {
          const labelElement = document.createElement("label");
          labelElement.className = "multi-option";
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.value = option;
          checkbox.id = `multi-option-${Date.now()}-${index}`;
          const text = document.createElement("span");
          text.textContent = option;
          checkbox.addEventListener("change", () => {
            submit.disabled = !choices.querySelector("input:checked");
          });
          labelElement.append(checkbox, text);
          choices.appendChild(labelElement);
        });
        submit.addEventListener("click", () => {
          const selectedOptions = [...choices.querySelectorAll("input:checked")]
            .map((checkbox) => checkbox.value);
          if (!selectedOptions.length) return;
          ask(`Seçtiklerim: ${selectedOptions.join(" · ")}`, { quickReply: true, selectedOptions });
        });
        form.append(hint, choices, submit);
        article.appendChild(form);
      } else if (parsed.options.length) {
        const replies = document.createElement("div");
        replies.className = "quick-replies";
        for (const option of parsed.options) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = option;
          button.addEventListener("click", () => {
            const answer = /[?？]\s*$/.test(option) ? `Evet: ${option}` : option;
            ask(answer, { quickReply: true });
          });
          replies.appendChild(button);
        }
        article.appendChild(replies);
      }
      addReferences(article, meta.references);
    } else {
      body.textContent = content;
      article.append(label, body);
    }

    if (meta.model || meta.latencyMs) {
      const detail = document.createElement("div");
      detail.className = "message-meta";
      const model = state.models.find((item) => item.id === meta.modelId);
      detail.textContent = [
        model?.label || meta.model,
        meta.latencyMs ? `${(meta.latencyMs / 1000).toFixed(1)} sn` : "",
      ].filter(Boolean).join(" · ");
      article.appendChild(detail);
    }
    $("messages").appendChild(article);
    scrollToBottom();
  }

  function bindStarterPrompts() {
    document.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => ask(button.dataset.prompt));
    });
  }

  function resetConversation() {
    state.history = [];
    const model = activeModel();
    $("messages").innerHTML = `
      <div id="welcome" class="welcome">
        <div class="welcome-orbit" aria-hidden="true"><span></span></div>
        <p class="eyebrow">DOKTOR DEĞERLENDİRME ALANI</p>
        <h3>Yanıtı hasta kaydıyla birlikte inceleyin</h3>
        <p>${state.mode === "multi_select"
    ? "Model, soruya özel 5–6 takip seçeneği üretir; uygun olanlar birlikte seçilebilir."
    : `${escapeHtml(model?.shortLabel || "Seçilen model")} yanıt üretirken sağdaki CCO kaydı karşılaştırma için açık kalır.`}</p>
        <div id="starter-prompts" class="starter-prompts" aria-label="Örnek sorular"></div>
      </div>`;
    const container = $("starter-prompts");
    for (const [label, question] of starterQuestions) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.prompt = question;
      button.textContent = label;
      container.appendChild(button);
    }
    bindStarterPrompts();
    updateControls();
  }

  function renderPatients() {
    const list = $("patient-list");
    list.innerHTML = "";
    if (!state.patients.length) {
      const empty = document.createElement("div");
      empty.className = "empty-patients";
      empty.innerHTML = "<strong>Henüz CCO yok</strong><p>Anonim JSON dosyalarını <code>data/cco</code> klasörüne ekleyin.</p>";
      list.appendChild(empty);
      return;
    }
    state.patients.forEach((patient, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `patient-button${patient.id === state.activePatient ? " active" : ""}`;
      button.innerHTML = `
        <span class="patient-index">${String(index + 1).padStart(2, "0")}</span>
        <span class="patient-labels"><span class="patient-name">${escapeHtml(patient.label)}</span><span class="patient-id">Anonim CCO</span></span>
        <span class="patient-arrow" aria-hidden="true">›</span>`;
      button.addEventListener("click", async () => {
        if (state.busy || patient.id === state.activePatient) { closePanels(); return; }
        state.activePatient = patient.id;
        $("active-patient").textContent = patient.label;
        renderPatients();
        resetConversation();
        await loadRecord(patient.id);
        closePanels();
      });
      list.appendChild(button);
    });
  }

  function renderModels() {
    const featured = $("featured-models");
    const select = $("model-select");
    featured.innerHTML = "";
    select.innerHTML = "";

    for (const model of state.models) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = `${model.label} — ${statusLabels[model.status] || model.status}`;
      option.disabled = model.status !== "ready";
      option.selected = model.id === state.activeModel;
      select.appendChild(option);

      if (!model.featured) continue;
      const button = document.createElement("button");
      button.type = "button";
      button.className = `featured-model${model.id === state.activeModel ? " active" : ""}`;
      button.dataset.model = model.id;
      button.dataset.status = model.status;
      button.innerHTML = `
        <span class="model-provider">${escapeHtml(model.provider)}</span>
        <strong>${escapeHtml(model.shortLabel)}</strong>
        <span class="model-state"><i class="${model.status === "ready" ? "ready" : ""}"></i>${escapeHtml(statusLabels[model.status] || model.status)}</span>`;
      button.addEventListener("click", () => selectModel(model.id));
      featured.appendChild(button);
    }
    updateConnection();
    updateControls();
  }

  function selectModel(modelId) {
    if (state.busy || modelId === state.activeModel) return;
    const model = state.models.find((item) => item.id === modelId);
    if (!model || model.status !== "ready") return;
    state.activeModel = modelId;
    renderModels();
    resetConversation();
  }

  function setMode(mode) {
    if (state.busy || mode === state.mode) return;
    state.mode = mode;
    document.querySelectorAll(".mode-button").forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    resetConversation();
  }

  function appendReadableLine(container, rawLine) {
    const line = rawLine.trim();
    if (!line) return;
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const title = document.createElement(heading[1].length <= 2 ? "h3" : "h4");
      title.innerHTML = inlineMarkdown(heading[2]);
      container.appendChild(title);
      return;
    }
    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      let list = container.lastElementChild;
      if (!list || list.tagName !== "UL") {
        list = document.createElement("ul");
        container.appendChild(list);
      }
      const item = document.createElement("li");
      item.innerHTML = inlineMarkdown(bullet[1]);
      list.appendChild(item);
      return;
    }
    const paragraph = document.createElement("p");
    paragraph.innerHTML = inlineMarkdown(line);
    container.appendChild(paragraph);
  }

  function renderReadableRecord() {
    const target = $("readable-record");
    target.innerHTML = "";
    const text = state.record?.readableText || "";
    if (!text) {
      target.innerHTML = "<p class=\"record-empty\">Okunabilir CCO metni bulunamadı.</p>";
      return;
    }
    const blockPattern = /\S(?:.*?\S)?(?=\n\s*\n|$)/gs;
    let match;
    while ((match = blockPattern.exec(text)) !== null) {
      const block = document.createElement("section");
      block.className = "source-block";
      block.dataset.start = String(match.index);
      block.dataset.end = String(match.index + match[0].length);
      for (const line of match[0].split("\n")) appendReadableLine(block, line);
      if (state.highlight) {
        const overlaps = match.index < state.highlight.end
          && match.index + match[0].length > state.highlight.start;
        block.classList.toggle("highlighted", overlaps);
      }
      target.appendChild(block);
    }
  }

  function renderRecord() {
    const record = state.record;
    $("record-title").textContent = record?.label || "Hasta kaydı";
    const summary = record?.summary || [];
    $("summary-count").textContent = summary.length ? `${summary.length} bulgu` : "";
    $("summary-list").innerHTML = "";
    if (!summary.length) {
      const item = document.createElement("li");
      item.textContent = "Klinik özet üretmek için yeterli metin bulunamadı.";
      $("summary-list").appendChild(item);
    } else {
      summary.forEach((line) => {
        const item = document.createElement("li");
        item.textContent = line;
        $("summary-list").appendChild(item);
      });
    }
    renderReadableRecord();
    $("json-record").textContent = record ? JSON.stringify(record.rawJson, null, 2) : "";
  }

  async function loadRecord(patientId) {
    state.record = null;
    state.highlight = null;
    $("source-banner").hidden = true;
    $("readable-record").innerHTML = "<div class=\"record-loading\"><span class=\"spinner\"></span> Kayıt yükleniyor…</div>";
    try {
      const record = await api(`/api/patients/${encodeURIComponent(patientId)}/record`);
      if (patientId !== state.activePatient) return;
      state.record = record;
      renderRecord();
    } catch (error) {
      $("readable-record").textContent = error.message;
    }
  }

  function setRecordTab(tab) {
    state.recordTab = tab;
    document.querySelectorAll(".record-tab").forEach((button) => {
      const active = button.dataset.recordTab === tab;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    $("readable-record").hidden = tab !== "readable";
    $("json-record").hidden = tab !== "json";
  }

  async function ask(question, { quickReply = false, selectedOptions = [] } = {}) {
    question = (question || $("question").value).trim();
    if (!question || state.busy || !state.activePatient || activeModel()?.status !== "ready") return;
    addMessage("user", question);
    $("question").value = "";
    setBusy(true);
    try {
      const result = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({
          patientId: state.activePatient,
          modelId: state.activeModel,
          question,
          history: state.history,
          quickReply,
          interactionMode: state.mode,
          selectedOptions,
        }),
      });
      addMessage("assistant", result.answer, {
        ...result,
        allowMultiSelect: state.mode === "multi_select" && !quickReply && selectedOptions.length === 0,
      });
      state.history.push(
        { role: "user", content: question },
        { role: "assistant", content: result.answer },
      );
      state.history = state.history.slice(-6);
    } catch (error) {
      addMessage("system", error.message, { error: true });
    } finally {
      setBusy(false);
      $("question").focus();
    }
  }

  function bindStaticControls() {
    $("composer").addEventListener("submit", (event) => { event.preventDefault(); ask(); });
    $("question").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ask(); }
    });
    $("clear-button").addEventListener("click", resetConversation);
    $("model-select").addEventListener("change", (event) => selectModel(event.target.value));
    document.querySelectorAll(".mode-button").forEach((button) => {
      button.addEventListener("click", () => setMode(button.dataset.mode));
    });
    document.querySelectorAll(".record-tab").forEach((button) => {
      button.addEventListener("click", () => setRecordTab(button.dataset.recordTab));
    });
    $("clear-highlight").addEventListener("click", () => {
      state.highlight = null;
      $("source-banner").hidden = true;
      renderReadableRecord();
    });
    $("open-patients").addEventListener("click", () => openPanel("patients"));
    $("open-record").addEventListener("click", () => openPanel("record"));
    $("panel-backdrop").addEventListener("click", closePanels);
    document.querySelectorAll("[data-close-panel]").forEach((button) => {
      button.addEventListener("click", closePanels);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closePanels();
    });
  }

  async function initialize() {
    bindStaticControls();
    try {
      const [patients, models] = await Promise.all([api("/api/patients"), api("/api/models")]);
      state.patients = patients;
      state.models = models;
      state.activePatient = patients[0]?.id || null;
      const preferred = models.find((model) => model.id === "gpu-gemma4" && model.status === "ready")
        || models.find((model) => model.id === "openai-luna" && model.status === "ready")
        || models.find((model) => model.status === "ready")
        || models[0];
      state.activeModel = preferred?.id || null;
      $("active-patient").textContent = patients[0]?.label || "Hasta kaydı bekleniyor";
      renderPatients();
      renderModels();
      resetConversation();
      if (state.activePatient) await loadRecord(state.activePatient);
    } catch (error) {
      $("status-dot").className = "status-dot error";
      $("status-text").textContent = "Bağlantı yok";
      addMessage("system", error.message, { error: true });
    }
  }

  initialize();
})();
