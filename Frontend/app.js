// LegalSathi AI - chat frontend.
// When opened through FastAPI (http://localhost:8000) the API is same-origin.
// When opened from another server / file://, it falls back to localhost:8000.
const API_BASE =
  location.protocol.startsWith("http") && location.port === "8000" ? "" : "http://localhost:8000";

const els = {
  messages: document.getElementById("messages"),
  welcome: document.getElementById("welcome"),
  form: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("sendBtn"),
  examples: document.getElementById("examples"),
  docList: document.getElementById("docList"),
  docCount: document.getElementById("docCount"),
  status: document.getElementById("status"),
  statusText: document.getElementById("statusText"),
  newChat: document.getElementById("newChat"),
  menuBtn: document.getElementById("menuBtn"),
  sidebar: document.getElementById("sidebar"),
};

let history = []; // [{role, content}] sent to the backend for follow-up questions
let busy = false;

const TYPE_LABEL = { text: "text", table: "table", image: "diagram/image", ocr: "scanned page" };

// ---------------------------------------------------------------- helpers
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderMarkdown(md) {
  const html = DOMPurify.sanitize(marked.parse(md));
  // turn [1] markers into clickable citation refs
  return html.replace(/\[(\d+)\]/g, '<sup class="cite-ref" data-n="$1">[$1]</sup>');
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function addMessage(role, html, extraClass = "") {
  els.welcome?.remove();
  const div = document.createElement("div");
  div.className = `msg ${role} ${extraClass}`.trim();
  div.innerHTML = html;
  els.messages.appendChild(div);
  scrollToBottom();
  return div;
}

function renderCitations(citations) {
  if (!citations?.length) return "";
  const items = citations
    .map(
      (c) => `
      <details class="cite" data-n="${c.n}">
        <summary>
          <span class="num">${c.n}</span>
          <span class="src" title="${escapeHtml(c.source)}">${escapeHtml(c.title || c.source)}</span>
          <span class="meta">p. ${c.page} · ${TYPE_LABEL[c.type] || c.type}</span>
        </summary>
        <div class="snippet"><strong>${escapeHtml(c.source)}</strong>\n${escapeHtml(c.snippet)}…</div>
      </details>`
    )
    .join("");
  return `<div class="citations"><h4>Sources</h4>${items}</div>`;
}

function renderBotAnswer(data) {
  if (!data.answered) {
    return addMessage(
      "bot",
      `<div class="refusal-tag">Not found in knowledge base</div><p>${escapeHtml(data.answer)}</p>`,
      "refusal"
    );
  }
  const div = addMessage("bot", renderMarkdown(data.answer) + renderCitations(data.citations));
  div.querySelectorAll(".cite-ref").forEach((ref) =>
    ref.addEventListener("click", () => {
      const target = div.querySelector(`.cite[data-n="${ref.dataset.n}"]`);
      if (!target) return;
      target.open = true;
      target.classList.add("flash");
      target.scrollIntoView({ behavior: "smooth", block: "nearest" });
      setTimeout(() => target.classList.remove("flash"), 1200);
    })
  );
  return div;
}

// ---------------------------------------------------------------- chat
async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  els.send.disabled = true;
  addMessage("user", escapeHtml(question));
  const typing = addMessage("bot", '<div class="typing"><span></span><span></span><span></span></div>');

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: history.slice(-6) }),
    });
    const data = await res.json();
    typing.remove();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    renderBotAnswer(data);
    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: data.answer });
  } catch (err) {
    typing.remove();
    addMessage("bot", `<div class="refusal-tag">Error</div><p>${escapeHtml(err.message)}</p>`, "refusal");
  } finally {
    busy = false;
    els.send.disabled = false;
    els.input.focus();
  }
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.input.value;
  els.input.value = "";
  autoGrow();
  ask(q);
});

els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.form.requestSubmit();
  }
});

function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 180) + "px";
}
els.input.addEventListener("input", autoGrow);

els.examples.addEventListener("click", (e) => {
  if (e.target.tagName === "LI") {
    els.sidebar.classList.remove("open");
    ask(e.target.textContent);
  }
});

els.newChat.addEventListener("click", () => {
  history = [];
  els.messages.innerHTML = "";
  addMessage("bot", "<p>New conversation started. Ask me a labour law or corporate law question.</p>");
});

els.menuBtn.addEventListener("click", () => els.sidebar.classList.toggle("open"));

// ---------------------------------------------------------------- status + document list
async function loadStatus() {
  try {
    const health = await (await fetch(`${API_BASE}/api/health`)).json();
    els.status.className = "status ok";
    els.statusText.textContent = `Online · ${health.chunks_indexed.toLocaleString()} chunks indexed`;
    const docs = await (await fetch(`${API_BASE}/api/documents`)).json();
    els.docCount.textContent = docs.length;
    els.docList.innerHTML = docs
      .map((d) => `<li title="${escapeHtml(d.source)}">${escapeHtml(d.title || d.source)}</li>`)
      .join("");
  } catch {
    els.status.className = "status err";
    els.statusText.textContent = "Backend offline - start uvicorn";
    els.docCount.textContent = "0";
  }
}

loadStatus();
