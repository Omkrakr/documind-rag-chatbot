/* =========================================================================
   DocuMind UI — client logic
   Talks only to the documented REST API (same endpoints a Slack bot or
   internal portal would use): /health, /documents*, /chat/query.
   No build step, no framework — kept dependency-free on purpose, matching
   the rest of this project's "runs anywhere, zero setup" philosophy.
   ========================================================================= */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const el = {
    html: document.documentElement,
    themeToggle: $("#theme-toggle"),
    sidebar: $("#sidebar"),
    sidebarBackdrop: $("#sidebar-backdrop"),
    sidebarOpenBtn: $("#sidebar-open-btn"),
    newChatBtn: $("#new-chat-btn"),
    dropzone: $("#dropzone"),
    fileInput: $("#file-input"),
    docList: $("#doc-list"),
    docCount: $("#doc-count"),
    docEmptyHint: $("#doc-empty-hint"),
    healthDot: $("#health-dot"),
    healthText: $("#health-text"),
    providerRow: $("#provider-row"),
    chatTitle: $("#chat-title"),
    messages: $("#messages"),
    emptyState: $("#empty-state"),
    suggestionRow: $("#suggestion-row"),
    composer: $("#composer"),
    queryInput: $("#query-input"),
    sendBtn: $("#send-btn"),
  };

  const tpl = {
    userMsg: $("#tpl-user-msg"),
    assistantMsg: $("#tpl-assistant-msg"),
    sourceChip: $("#tpl-source-chip"),
    thinking: $("#tpl-thinking"),
    docItem: $("#tpl-doc-item"),
  };

  const state = {
    conversationId: null,
    documents: [],
    sending: false,
  };

  const SUGGESTIONS = [
    "How many days can I work from home?",
    "What is the maternity leave policy?",
    "What's required for a strong password?",
  ];

  // ---------------------------------------------------------------------
  // Theme
  // ---------------------------------------------------------------------
  function initTheme() {
    const saved = localStorage.getItem("documind-theme");
    const preferred = saved || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    setTheme(preferred);
  }
  function setTheme(theme) {
    el.html.setAttribute("data-theme", theme);
    localStorage.setItem("documind-theme", theme);
    el.themeToggle.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }
  el.themeToggle.addEventListener("click", () => {
    const current = el.html.getAttribute("data-theme");
    setTheme(current === "dark" ? "light" : "dark");
  });

  // ---------------------------------------------------------------------
  // Health check
  // ---------------------------------------------------------------------
  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error();
      const data = await res.json();
      el.healthDot.className = "status-dot online";
      el.healthText.textContent = "Backend connected";
      el.providerRow.textContent = `Embeddings: ${data.embedding_provider} · Generation: ${data.llm_provider}`;
    } catch {
      el.healthDot.className = "status-dot offline";
      el.healthText.textContent = "Backend unreachable";
      el.providerRow.textContent = "";
    }
  }

  // ---------------------------------------------------------------------
  // Documents
  // ---------------------------------------------------------------------
  async function loadDocuments() {
    try {
      const res = await fetch("/documents");
      if (!res.ok) throw new Error();
      const data = await res.json();
      state.documents = data.documents || [];
      renderDocuments();
    } catch {
      // Backend likely still starting up -- fail quietly, health dot covers it.
    }
  }

  function renderDocuments() {
    el.docList.innerHTML = "";
    el.docCount.textContent = `${state.documents.length} document${state.documents.length === 1 ? "" : "s"}`;
    el.docEmptyHint.style.display = state.documents.length ? "none" : "block";

    for (const doc of state.documents) {
      const node = tpl.docItem.content.firstElementChild.cloneNode(true);
      $(".doc-item-name", node).textContent = doc.filename;
      $(".doc-item-sub", node).textContent =
        doc.status === "ready" ? `${doc.chunk_count} chunks` : doc.status;
      const statusEl = $(".doc-item-status", node);
      statusEl.textContent = doc.status;
      statusEl.classList.add(doc.status);
      $(".doc-item-delete", node).addEventListener("click", () => deleteDocument(doc.document_id));
      el.docList.appendChild(node);
    }
  }

  async function uploadFile(file) {
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["txt", "md"].includes(ext)) {
      alert("Only .txt and .md files are supported by this prototype's loader.");
      return;
    }

    const placeholder = { document_id: `pending-${Date.now()}`, filename: file.name, status: "processing", chunk_count: 0 };
    state.documents = [placeholder, ...state.documents];
    renderDocuments();

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/documents/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      state.documents = state.documents.map(d => (d.document_id === placeholder.document_id ? data : d));
    } catch (err) {
      state.documents = state.documents.map(d =>
        d.document_id === placeholder.document_id ? { ...d, status: "failed" } : d
      );
    }
    renderDocuments();
  }

  async function deleteDocument(documentId) {
    state.documents = state.documents.filter(d => d.document_id !== documentId);
    renderDocuments();
    try {
      await fetch(`/documents/${documentId}`, { method: "DELETE" });
    } catch {
      loadDocuments(); // resync on failure
    }
  }

  el.dropzone.addEventListener("click", e => { if (e.target.tagName !== "INPUT") el.fileInput.click(); });
  el.fileInput.addEventListener("change", e => {
    uploadFile(e.target.files[0]);
    e.target.value = "";
  });
  ["dragenter", "dragover"].forEach(evt =>
    el.dropzone.addEventListener(evt, e => { e.preventDefault(); el.dropzone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach(evt =>
    el.dropzone.addEventListener(evt, e => { e.preventDefault(); el.dropzone.classList.remove("dragover"); })
  );
  el.dropzone.addEventListener("drop", e => uploadFile(e.dataTransfer.files[0]));

  // ---------------------------------------------------------------------
  // Mobile sidebar
  // ---------------------------------------------------------------------
  el.sidebarOpenBtn.addEventListener("click", () => el.sidebar.classList.add("open"));
  el.sidebarBackdrop.addEventListener("click", () => el.sidebar.classList.remove("open"));

  // ---------------------------------------------------------------------
  // Chat
  // ---------------------------------------------------------------------
  function showEmptyState(show) {
    el.emptyState.style.display = show ? "block" : "none";
  }

  function renderSuggestions() {
    el.suggestionRow.innerHTML = "";
    for (const text of SUGGESTIONS) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "suggestion-chip";
      chip.textContent = text;
      chip.addEventListener("click", () => { el.queryInput.value = text; sendMessage(); });
      el.suggestionRow.appendChild(chip);
    }
  }

  function appendUserMessage(text) {
    const node = tpl.userMsg.content.firstElementChild.cloneNode(true);
    $(".bubble", node).textContent = text;
    el.messages.appendChild(node);
    scrollToBottom();
  }

  function appendThinking() {
    const node = tpl.thinking.content.firstElementChild.cloneNode(true);
    el.messages.appendChild(node);
    scrollToBottom();
    return node;
  }

  function appendAssistantMessage(answer, sources, isError = false) {
    const node = tpl.assistantMsg.content.firstElementChild.cloneNode(true);
    const bubble = $(".bubble", node);
    bubble.textContent = answer;
    if (isError) bubble.classList.add("error");

    const sourcesEl = $(".sources", node);
    for (const src of sources || []) {
      const chip = tpl.sourceChip.content.firstElementChild.cloneNode(true);
      $(".source-filename", chip).textContent = src.document_id;
      $(".source-text", chip).textContent = src.text;
      const pct = Math.max(2, Math.min(100, Math.round(src.score * 100)));
      $(".score-bar-fill", chip).style.width = pct + "%";
      $(".score-value", chip).textContent = src.score.toFixed(2);
      sourcesEl.appendChild(chip);
    }
    el.messages.appendChild(node);
    scrollToBottom();
  }

  function scrollToBottom() {
    el.messages.scrollTop = el.messages.scrollHeight;
  }

  async function sendMessage() {
    const text = el.queryInput.value.trim();
    if (!text || state.sending) return;

    showEmptyState(false);
    appendUserMessage(text);
    el.queryInput.value = "";
    autoResize();
    updateSendState();

    state.sending = true;
    const thinkingNode = appendThinking();

    try {
      const res = await fetch("/chat/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: state.conversationId, query: text }),
      });
      const data = await res.json();
      thinkingNode.remove();
      if (!res.ok) {
        appendAssistantMessage(data.detail || "Something went wrong answering that.", [], true);
      } else {
        state.conversationId = data.conversation_id;
        if (el.chatTitle.textContent === "New conversation") {
          el.chatTitle.textContent = text.length > 48 ? text.slice(0, 48) + "…" : text;
        }
        appendAssistantMessage(data.answer, data.sources);
      }
    } catch (err) {
      thinkingNode.remove();
      appendAssistantMessage("Couldn't reach the backend. Is the server still running?", [], true);
    } finally {
      state.sending = false;
    }
  }

  el.composer.addEventListener("submit", e => { e.preventDefault(); sendMessage(); });
  el.queryInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  el.queryInput.addEventListener("input", () => { autoResize(); updateSendState(); });

  function autoResize() {
    el.queryInput.style.height = "auto";
    el.queryInput.style.height = Math.min(el.queryInput.scrollHeight, 160) + "px";
  }
  function updateSendState() {
    el.sendBtn.disabled = el.queryInput.value.trim().length === 0;
  }

  el.newChatBtn.addEventListener("click", () => {
    state.conversationId = null;
    el.messages.innerHTML = "";
    el.messages.appendChild(el.emptyState);
    showEmptyState(true);
    el.chatTitle.textContent = "New conversation";
    el.queryInput.focus();
  });

  // ---------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------
  initTheme();
  renderSuggestions();
  checkHealth();
  loadDocuments();
  updateSendState();
  el.queryInput.focus();
})();
