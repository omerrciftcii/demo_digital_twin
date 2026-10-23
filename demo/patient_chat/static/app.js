(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MODEL_PROMPT_STORAGE_KEY = "medicalai:system-prompts:v1";
  const MAX_MODEL_PROMPT_LENGTH = 6000;
  const state = {
    patients: [],
    models: [],
    activePatient: null,
    activeModel: null,
    record: null,
    history: [],
    transcript: [],
    busy: false,
    mode: "stepwise",
    preprocessMode: "raw",
    recordTab: "readable",
    highlight: null,
    modelPrompts: loadModelPrompts(),
    defaultSystemPrompt: "",
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
    auth_error: "Anahtar geçersiz",
  };

  function loadModelPrompts() {
    try {
      const stored = JSON.parse(window.localStorage.getItem(MODEL_PROMPT_STORAGE_KEY) || "{}");
      if (!stored || typeof stored !== "object" || Array.isArray(stored)) return {};
      return Object.fromEntries(
        Object.entries(stored)
          .filter(([, value]) => typeof value === "string")
          .map(([modelId, value]) => [modelId, value.slice(0, MAX_MODEL_PROMPT_LENGTH)]),
      );
    } catch (_error) {
      return {};
    }
  }

  function persistModelPrompts() {
    try {
      window.localStorage.setItem(MODEL_PROMPT_STORAGE_KEY, JSON.stringify(state.modelPrompts));
      return true;
    } catch (_error) {
      return false;
    }
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
  }

  function inlineMarkdown(value) {
    const codeTokens = [];
    let html = escapeHtml(value).replace(/`([^`]+)`/g, (_match, code) => {
      const token = `\u0000CODE${codeTokens.length}\u0000`;
      codeTokens.push(`<code>${code}</code>`);
      return token;
    });
    html = html
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/__(.+?)__/g, "<strong>$1</strong>")
      .replace(/~~(.+?)~~/g, "<del>$1</del>")
      .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1<em>$2</em>");
    return html.replace(/\u0000CODE(\d+)\u0000/g, (_match, index) => codeTokens[Number(index)]);
  }

  function markdownToHtml(markdown) {
    const lines = String(markdown).replace(/\r\n/g, "\n").split("\n");
    const output = [];
    let list = null;
    let codeFence = null;
    let codeLines = [];
    const closeList = () => {
      if (list) output.push(`</${list}>`);
      list = null;
    };
    const closeCodeFence = () => {
      if (codeFence === null) return;
      const language = codeFence ? ` class="language-${escapeHtml(codeFence)}"` : "";
      output.push(`<pre><code${language}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      codeFence = null;
      codeLines = [];
    };
    for (const rawLine of lines) {
      const line = rawLine.trim();
      const fence = line.match(/^```([\w-]*)\s*$/);
      if (fence) {
        closeList();
        if (codeFence === null) codeFence = fence[1] || "";
        else closeCodeFence();
        continue;
      }
      if (codeFence !== null) {
        codeLines.push(rawLine);
        continue;
      }
      if (!line) { closeList(); continue; }
      if (/^([-*_])\1\1+$/.test(line.replace(/\s/g, ""))) {
        closeList();
        output.push("<hr>");
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        closeList();
        const level = Math.min(4, heading[1].length + 1);
        output.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
        continue;
      }
      const quote = line.match(/^>\s?(.+)$/);
      if (quote) {
        closeList();
        output.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`);
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
    closeCodeFence();
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

  function activePatient() {
    return state.patients.find((patient) => patient.id === state.activePatient) || null;
  }

  function normalizePrompt(value) {
    return String(value || "").replace(/\r\n/g, "\n").trim();
  }

  function isDefaultPrompt(value) {
    const candidate = normalizePrompt(value);
    const fallback = normalizePrompt(state.defaultSystemPrompt);
    return Boolean(fallback) && candidate === fallback;
  }

  function storedSystemPrompt(modelId = state.activeModel) {
    const stored = modelId ? state.modelPrompts[modelId] : "";
    return stored && !isDefaultPrompt(stored) ? stored : "";
  }

  function activeSystemPrompt(modelId = state.activeModel) {
    return storedSystemPrompt(modelId) || state.defaultSystemPrompt || "";
  }

  function promptSourceLabel(source) {
    return source === "custom" ? "arayüzden düzenlenmiş" : "varsayılan";
  }

  function fencePrompt(prompt) {
    const fence = String(prompt).includes("```") ? "~~~~" : "```";
    return [`${fence}text`, prompt, fence];
  }

  function usedSystemPrompts() {
    const used = [];
    let answerNumber = 0;
    state.transcript.forEach((message) => {
      if (message.role !== "assistant" || !message.systemPrompt) return;
      answerNumber += 1;
      const previous = used[used.length - 1];
      if (previous && previous.prompt === message.systemPrompt) {
        previous.answers.push(answerNumber);
        return;
      }
      used.push({
        prompt: message.systemPrompt,
        source: message.systemPromptSource || (isDefaultPrompt(message.systemPrompt) ? "default" : "custom"),
        answers: [answerNumber],
      });
    });
    return used;
  }

  function exportSlug(value) {
    return String(value || "kayit")
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-zA-Z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .toLowerCase() || "kayit";
  }

  function exportConversation() {
    if (!state.transcript.length) return;
    const patient = activePatient();
    const model = activeModel();
    const createdAt = new Date();
    const lines = [
      "# MedicalAI model değerlendirmesi",
      "",
      `- Hasta: ${patient?.label || "Bilinmiyor"}`,
      `- Model: ${model?.label || "Bilinmiyor"}`,
      `- Sağlayıcı: ${model?.provider || "Bilinmiyor"}`,
      `- Görüşme modu: ${state.mode === "multi_select" ? "Çoklu seçim" : "Adım adım"}`,
      `- Kayıt ön işleme: ${{ raw: "Ham kayıt", s5: "S5", s6b: "S6b" }[state.preprocessMode] || "Ham kayıt"}`,
      `- Dışa aktarma zamanı: ${createdAt.toLocaleString("tr-TR")}`,
      "",
      "## Sistem promptu",
      "",
    ];
    const usedPrompts = usedSystemPrompts();
    if (!usedPrompts.length) {
      const prepared = activeSystemPrompt();
      lines.push("Bu dışa aktarmada modele giden bir yanıt yok.", "");
      if (prepared) {
        lines.push(
          `- Hazırlanan prompt: ${isDefaultPrompt(prepared) ? "varsayılan" : "arayüzden düzenlenmiş"}`,
          "",
          ...fencePrompt(prepared),
          "",
        );
      }
    } else {
      usedPrompts.forEach((item, index) => {
        if (usedPrompts.length > 1) {
          lines.push(`### ${index + 1}. ${promptSourceLabel(item.source)}`, "");
        }
        lines.push(
          `- Durum: ${promptSourceLabel(item.source)}`,
          `- Uygulandığı model yanıtları: ${item.answers.join(", ")}`,
          "",
          ...fencePrompt(item.prompt),
          "",
        );
      });
    }
    lines.push("## Mesaj geçmişi", "");
    state.transcript.forEach((message, index) => {
      const title = message.role === "user"
        ? "Soru"
        : message.role === "assistant" ? "Model yanıtı" : "Sistem";
      lines.push(`### ${index + 1}. ${title}`, "", message.content, "");
      if (message.role === "assistant" && (message.model || message.latencyMs)) {
        const details = [
          message.model ? `Model: ${message.model}` : "",
          message.latencyMs ? `Süre: ${(message.latencyMs / 1000).toFixed(1)} sn` : "",
        ].filter(Boolean).join(" · ");
        lines.push(`_${details}_`, "");
      }
    });
    lines.push("---", "Bu dosya araştırma demosundan dışa aktarılmıştır; tanı veya tedavi kararı değildir.", "");

    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stamp = createdAt.toISOString().replace(/[:.]/g, "-");
    link.href = url;
    link.download = `${exportSlug(patient?.label)}_${exportSlug(model?.shortLabel)}_${stamp}.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
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

  function renderModelPrompt(statusMessage = "") {
    const model = activeModel();
    const custom = storedSystemPrompt(model?.id);
    const prompt = custom || state.defaultSystemPrompt || "";
    $("model-prompt").value = prompt;
    $("model-prompt").disabled = state.busy || !model;
    $("model-prompt-label").textContent = model
      ? `${model.shortLabel} sistem promptu`
      : "Sistem promptu";
    $("model-prompt-summary").textContent = custom
      ? `${model.shortLabel} · düzenlendi`
      : "Varsayılan sistem promptu";
    $("model-prompt-status").textContent = statusMessage || (
      custom ? "Bu tarayıcıya kaydedildi" : "Varsayılan sistem promptu"
    );
    $("clear-model-prompt").disabled = state.busy || !model || !custom;
  }

  function saveActiveModelPrompt(value) {
    const model = activeModel();
    if (!model) return;
    const prompt = String(value).replace(/\r\n/g, "\n").slice(0, MAX_MODEL_PROMPT_LENGTH);
    const custom = Boolean(prompt.trim()) && !isDefaultPrompt(prompt);
    if (custom) state.modelPrompts[model.id] = prompt;
    else delete state.modelPrompts[model.id];
    const saved = persistModelPrompts();
    $("model-prompt-summary").textContent = custom
      ? `${model.shortLabel} · düzenlendi`
      : "Varsayılan sistem promptu";
    if (!saved) $("model-prompt-status").textContent = "Tarayıcı saklama alanına erişilemedi";
    else if (!prompt.trim()) $("model-prompt-status").textContent = "Boş; gönderimde varsayılan kullanılır";
    else if (custom) $("model-prompt-status").textContent = "Bu tarayıcıya kaydedildi";
    else $("model-prompt-status").textContent = "Varsayılan sistem promptu";
    $("clear-model-prompt").disabled = state.busy || normalizePrompt(prompt) === normalizePrompt(state.defaultSystemPrompt);
  }

  function updateControls() {
    const ready = activeModel()?.status === "ready";
    const blocked = state.busy || !state.activePatient || !ready;
    $("activity").hidden = !state.busy;
    $("question").disabled = blocked;
    $("send-button").disabled = blocked;
    $("export-button").disabled = state.busy || !state.transcript.length;
    $("model-select").disabled = state.busy;
    $("model-prompt").disabled = state.busy || !activeModel();
    $("clear-model-prompt").disabled = state.busy || !activeModel()
      || normalizePrompt($("model-prompt").value) === normalizePrompt(state.defaultSystemPrompt);
    document.querySelectorAll(".quick-replies button,.multi-replies button,.multi-replies input,.starter-prompts button,.patient-button,.mode-button,.featured-model")
      .forEach((control) => {
        control.disabled = state.busy;
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

  function showReference(reference, answerNode = null) {
    document.querySelectorAll(".answer-source.active-source").forEach((node) => {
      node.classList.remove("active-source");
    });
    answerNode?.classList.add("active-source");
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

  function normalizeReferenceText(value) {
    return String(value || "")
      .toLocaleLowerCase("tr-TR")
      .replace(/[^\p{L}\p{N}]+/gu, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  const referenceStopwords = new Set([
    "acil", "altında", "ancak", "bir", "bu", "daha", "devam", "göre", "için",
    "ile", "kayıt", "olan", "olarak", "son", "ve", "veya", "yüksek", "düşük",
  ]);

  function referenceMatchScore(node, reference) {
    const nodeText = normalizeReferenceText(node.textContent);
    const answerText = normalizeReferenceText(reference.answerText);
    if (!nodeText || !answerText) return -1;
    const answerTokens = new Set(answerText.split(" "));
    const sourceTokens = new Set(normalizeReferenceText(reference.sourceText).split(" "));
    const nodeTokens = [...new Set(nodeText.split(" "))]
      .filter((token) => token.length >= 3 && !referenceStopwords.has(token));
    const answerShared = nodeTokens.filter((token) => answerTokens.has(token));
    const sourceShared = nodeTokens.filter((token) => sourceTokens.has(token));
    const numbers = nodeTokens.filter((token) => /\d/.test(token) && sourceTokens.has(token));
    const minimumAnswerOverlap = Math.min(3, Math.max(2, Math.ceil(nodeTokens.length * 0.25)));
    if (answerShared.length < minimumAnswerOverlap) return -1;
    if (sourceShared.length < 2 && numbers.length === 0) return -1;
    return sourceShared.length + answerShared.length + (numbers.length * 4);
  }

  function linkAnswerTextToReferences(body, references) {
    if (!Array.isArray(references) || !references.length) return;
    const candidates = [...body.querySelectorAll("p, li")];
    references.forEach((reference, index) => {
      const target = candidates
        .filter((node) => !node.classList.contains("answer-source"))
        .map((node) => ({ node, score: referenceMatchScore(node, reference) }))
        .filter((item) => item.score >= 0)
        .sort((left, right) => right.score - left.score)[0]?.node;
      if (!target || target.classList.contains("answer-source")) return;
      target.classList.add("answer-source");
      target.tabIndex = 0;
      target.setAttribute("role", "button");
      target.setAttribute("aria-label", `Kaynak ${index + 1}: CCO kaydında göster`);
      target.title = "Bu bilginin CCO kaynağını göster";
      target.addEventListener("click", () => showReference(reference, target));
      target.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        showReference(reference, target);
      });
    });
  }

  function addMessage(role, content, meta = {}) {
    document.body.classList.add("has-messages");
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
      linkAnswerTextToReferences(body, meta.references);
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
    state.transcript = [];
    document.body.classList.remove("has-messages");
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
    renderModelPrompt();
    updateControls();
  }

  function selectModel(modelId) {
    if (state.busy || modelId === state.activeModel) return;
    const model = state.models.find((item) => item.id === modelId);
    if (!model) return;
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
    state.transcript.push({ role: "user", content: question, createdAt: new Date().toISOString() });
    $("question").value = "";
    setBusy(true);
    try {
      const systemPrompt = activeSystemPrompt();
      const result = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({
          patientId: state.activePatient,
          modelId: state.activeModel,
          modelPrompt: systemPrompt,
          question,
          history: state.history,
          quickReply,
          interactionMode: state.mode,
          selectedOptions,
          preprocessMode: state.preprocessMode,
        }),
      });
      addMessage("assistant", result.answer, {
        ...result,
        allowMultiSelect: state.mode === "multi_select" && !quickReply && selectedOptions.length === 0,
      });
      state.transcript.push({
        role: "assistant",
        content: result.answer,
        model: result.model,
        modelId: result.modelId,
        latencyMs: result.latencyMs,
        systemPrompt: result.systemPrompt || systemPrompt,
        systemPromptSource: result.systemPromptSource || (isDefaultPrompt(systemPrompt) ? "default" : "custom"),
        createdAt: new Date().toISOString(),
      });
      state.history.push(
        { role: "user", content: question },
        { role: "assistant", content: result.answer },
      );
      state.history = state.history.slice(-6);
    } catch (error) {
      addMessage("system", error.message, { error: true });
      state.transcript.push({
        role: "system",
        content: error.message,
        createdAt: new Date().toISOString(),
      });
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
    $("export-button").addEventListener("click", exportConversation);
    $("model-select").addEventListener("change", (event) => selectModel(event.target.value));
    $("model-prompt").addEventListener("input", (event) => saveActiveModelPrompt(event.target.value));
    $("clear-model-prompt").addEventListener("click", () => {
      const model = activeModel();
      if (model) delete state.modelPrompts[model.id];
      persistModelPrompts();
      renderModelPrompt("Varsayılan sistem promptu");
      $("model-prompt").focus();
    });
    document.querySelectorAll(".pre-button").forEach((button) => {
      button.addEventListener("click", () => {
        state.preprocessMode = ["raw", "s5", "s6b"].includes(button.dataset.pre) ? button.dataset.pre : "raw";
        document.querySelectorAll(".pre-button").forEach((b) => {
          const on = b === button;
          b.classList.toggle("active", on);
          b.setAttribute("aria-pressed", on ? "true" : "false");
        });
      });
    });
    document.querySelectorAll(".mode-button").forEach((button) => {
      button.addEventListener("click", () => setMode(button.dataset.mode));
    });
    document.querySelectorAll(".record-tab").forEach((button) => {
      button.addEventListener("click", () => setRecordTab(button.dataset.recordTab));
    });
    $("clear-highlight").addEventListener("click", () => {
      state.highlight = null;
      document.querySelectorAll(".answer-source.active-source").forEach((node) => {
        node.classList.remove("active-source");
      });
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
      const [patients, models, promptPayload] = await Promise.all([
        api("/api/patients"),
        api("/api/models"),
        api("/api/system-prompt"),
      ]);
      state.patients = patients;
      state.models = models;
      state.defaultSystemPrompt = String(promptPayload.systemPrompt || "");
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
