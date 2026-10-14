(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = { patients: [], activePatient: null, history: [], busy: false };
  const starterQuestions = {
    P01: [
      ["Uykum gerçekten düzensiz mi?", "Geceleri iyi uyuyamadığımı hissediyorum. Son uyku verilerim bunu destekliyor mu?"],
      ["Yorgunluğumu verilerim açıklıyor mu?", "Son günlerde çok yorgun ve halsizim. Son klinik ve wearable verilerim ne gösteriyor?"],
      ["Gözüm kızardı", "Gözüm biraz kızardı ve batıyor. Son kayıtlarıma göre bununla ilişkili bir şey var mı?"],
    ],
    P02: [
      ["Kan sayımım güncel mi?", "En son kan sayımım ne zaman yapılmış? Kansızlık açısından ne gösteriyor ve bugün için güncel sayılır mı?"],
      ["Uykum düzenli mi?", "Son uyku verilerime göre uykum düzenli mi, dikkat çeken bir şey var mı?"],
    ],
    P03: [
      ["Çarpıntı için güncel veri var mı?", "Ara sıra çarpıntı hissediyorum. Son nabız, uyku ve klinik kayıtlarımda dikkate değer bir şey var mı?"],
      ["Gözüm kızardı ve batıyor", "Gözüm biraz kızardı ve batıyor. Son kayıtlarıma göre bununla ilişkili bir şey var mı?"],
    ],
  };

  function escapeHtml(value) {
    return value.replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
  }

  function inlineMarkdown(value) {
    return escapeHtml(value)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function markdownToHtml(markdown) {
    const lines = markdown.replace(/\r\n/g, "\n").split("\n");
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
    const lines = markdown.replace(/\r\n/g, "\n").split("\n");
    const optionPattern = /^(?:[-*]\s*)?([A-Ea-e])[).]\s*(.+)$/;
    let activeRun = [];
    let bestRun = [];
    for (let index = 0; index < lines.length; index += 1) {
      const trimmed = lines[index].trim();
      if (!trimmed) continue;
      const match = trimmed.match(optionPattern);
      if (!match) {
        activeRun = [];
        continue;
      }
      const letter = match[1].toUpperCase();
      const expected = String.fromCharCode(65 + activeRun.length);
      if (letter === "A") activeRun = [{ index, text: match[2].trim() }];
      else if (activeRun.length && letter === expected) {
        activeRun.push({ index, text: match[2].trim() });
      } else activeRun = [];
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
    if (!response.ok) throw new Error(payload.detail || "İstek tamamlanamadı.");
    return payload;
  }

  function setBusy(busy) {
    state.busy = busy;
    $("activity").hidden = !busy;
    $("question").disabled = busy;
    $("send-button").disabled = busy;
    document.querySelectorAll(".quick-replies button,.starter-prompts button,.patient-button")
      .forEach((button) => { button.disabled = busy; });
  }

  function scrollToBottom() {
    requestAnimationFrame(() => { $("messages").scrollTop = $("messages").scrollHeight; });
  }

  function addMessage(role, content, meta = {}) {
    $("welcome")?.remove();
    const article = document.createElement("article");
    article.className = `message ${role}${meta.error ? " error" : ""}`;
    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = role === "user" ? "SEN" : role === "assistant" ? "MEDICALAI" : "SİSTEM";
    const body = document.createElement("div");
    body.className = "message-content";
    if (role === "assistant") {
      const parsed = extractQuickReplies(content);
      body.innerHTML = markdownToHtml(parsed.markdown);
      if (parsed.options.length) {
        const replies = document.createElement("div");
        replies.className = "quick-replies";
        for (const option of parsed.options) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = option;
          button.addEventListener("click", () => {
            const answer = /[?？]\s*$/.test(option) ? `Evet: ${option}` : option;
            ask(answer, true);
          });
          replies.appendChild(button);
        }
        article.append(label, body, replies);
      } else article.append(label, body);
    } else {
      body.textContent = content;
      article.append(label, body);
    }
    if (meta.model || meta.latencyMs) {
      const detail = document.createElement("div");
      detail.className = "message-meta";
      detail.textContent = [meta.model, meta.latencyMs ? `${(meta.latencyMs / 1000).toFixed(1)} sn` : ""]
        .filter(Boolean).join(" · ");
      article.appendChild(detail);
    }
    $("messages").appendChild(article);
    scrollToBottom();
  }

  function resetConversation() {
    state.history = [];
    $("messages").innerHTML = `
      <div id="welcome" class="welcome">
        <div class="welcome-icon" aria-hidden="true">✦</div>
        <h3>Kayıtların hakkında doğal şekilde sor</h3>
        <p>Gemma son ilgili klinik kayıtları ve güncel wearable verilerini önceliklendirerek kısa bir yanıt hazırlar.</p>
        <div id="starter-prompts" class="starter-prompts" aria-label="Önceden denenmiş örnek sorular"></div>
      </div>`;
    const container = $("starter-prompts");
    for (const [label, question] of starterQuestions[state.activePatient] || []) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.prompt = question;
      button.textContent = label;
      container.appendChild(button);
    }
    bindStarterPrompts();
  }

  function renderPatients() {
    $("patient-list").innerHTML = "";
    for (const patient of state.patients) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `patient-button${patient.id === state.activePatient ? " active" : ""}`;
      button.innerHTML = `<span class="patient-avatar">${escapeHtml(patient.id.slice(-2))}</span><span><span class="patient-name">${escapeHtml(patient.label)}</span><span class="patient-id">${escapeHtml(patient.id)}</span></span>`;
      button.addEventListener("click", () => {
        if (state.busy || patient.id === state.activePatient) return;
        state.activePatient = patient.id;
        $("active-patient").textContent = patient.label;
        renderPatients();
        resetConversation();
      });
      $("patient-list").appendChild(button);
    }
  }

  function bindStarterPrompts() {
    document.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => ask(button.dataset.prompt));
    });
  }

  async function ask(question, quickReply = false) {
    question = (question || $("question").value).trim();
    if (!question || state.busy || !state.activePatient) return;
    addMessage("user", question);
    $("question").value = "";
    setBusy(true);
    try {
      const result = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({
          patientId: state.activePatient,
          question,
          history: state.history,
          quickReply,
        }),
      });
      addMessage("assistant", result.answer, result);
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

  async function initialize() {
    bindStarterPrompts();
    $("composer").addEventListener("submit", (event) => { event.preventDefault(); ask(); });
    $("question").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ask(); }
    });
    $("clear-button").addEventListener("click", resetConversation);
    try {
      const [patients, health] = await Promise.all([api("/api/patients"), api("/api/health")]);
      state.patients = patients;
      state.activePatient = patients[0]?.id || null;
      $("active-patient").textContent = patients[0]?.label || "Hasta kaydı bulunamadı";
      $("status-dot").className = `status-dot ${health.status === "ready" ? "ready" : "error"}`;
      $("status-text").textContent = health.status === "ready" ? "Gemma hazır" : "Bağlantı yok";
      renderPatients();
      resetConversation();
    } catch (error) {
      $("status-dot").className = "status-dot error";
      $("status-text").textContent = "Bağlantı yok";
      addMessage("system", error.message, { error: true });
    }
  }

  initialize();
})();
