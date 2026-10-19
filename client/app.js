"use strict";

// Named before the rename to GhostSwitch; kept so saved chats and attachments stay.
const STORAGE_KEY = "minibrain-client-v1";
const $ = (id) => document.getElementById(id);
const views = {
  overview: "Statistics",
  chat: "Chat",
  experts: "Experts",
  requests: "Request log",
  api: "API",
  settings: "Settings",
};
const defaults = {
  maxTokens: 4096,
  temperature: 0.7,
  systemPrompt: "",
  theme: "dark",
  baseUrl: location.origin,
  inspector: false,
  webSearch: "off",
  thinking: "auto",
  onboarded: false,
};
const WEB_SEARCH_MODES = {
  off: "Search off",
  auto: "Search: auto",
  on: "Search: always",
};
const THINKING_MODES = {
  auto: "Thinking: auto",
  on: "Thinking: always",
  off: "Thinking off",
};
const state = {
  conversations: [],
  activeId: null,
  view: "chat",
  experts: [],
  expertsAt: null,
  connected: false,
  health: null,
  healthAt: null,
  pending: null,
  inspectorId: null,
  inspectorMobileOpen: false,
  settings: { ...defaults },
  range: "all",
  requestFilter: "all",
  requestQuery: "",
  serverRequests: [],
  journalError: null,
  requestDetailId: null,
  catalogQuery: "",
  autoScroll: true,
  docsTab: "playground",
  playgroundResult: null,
  runtime: null,
  switching: null,
  onboarding: null,
  install: null,
  // Images and files waiting in the composer; their data is in IndexedDB.
  attachments: [],
};
const paths = {
  globe:
    '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.4 2.6 3.7 5.6 3.7 9s-1.3 6.4-3.7 9c-2.4-2.6-3.7-5.6-3.7-9s1.3-6.4 3.7-9Z"/>',
  brain:
    '<path d="M9 18H7a4 4 0 0 1-3-6 4 4 0 0 1 3-7 3 3 0 0 1 5-2v18a3 3 0 0 1-5-2M15 18h2a4 4 0 0 0 3-6 4 4 0 0 0-3-7 3 3 0 0 0-5-2M7 8c3 0 3 4 0 4m10-4c-3 0-3 4 0 4M8 16h4m4 0h-4"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  chat: '<path d="M21 11.5a9 9 0 0 1-9.5 9L3 22l1.5-8.5A9 9 0 1 1 21 11.5Z"/><path d="M8 10h8M8 14h5"/>',
  layers:
    '<path d="m12 3 10 5-10 5L2 8l10-5Zm-10 9 10 5 10-5M2 16l10 5 10-5"/>',
  activity: '<path d="M2 12h5l3-8 4 16 3-8h5"/>',
  code: '<path d="m8 6-6 6 6 6m8-12 6 6-6 6M14 3l-4 18"/>',
  settings:
    '<path d="M9 3h6l1 3 3 1 2 5-2 5-3 1-1 3H9l-1-3-3-1-2-5 2-5 3-1 1-3Z"/><circle cx="12" cy="12" r="3"/>',
  search: '<circle cx="10.5" cy="10.5" r="7.5"/><path d="m16 16 5 5"/>',
  refresh:
    '<path d="M20 7a9 9 0 0 0-15-2L2 8m0-5v5h5m-3 9a9 9 0 0 0 15 2l3-3m0 5v-5h-5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  arrowUp: '<path d="M12 20V4m-6 6 6-6 6 6"/>',
  arrowDown: '<path d="M12 4v16m-6-6 6 6 6-6"/>',
  arrowRight: '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  menu: '<path d="M3 6h18M3 12h18M3 18h18"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
  moon: '<path d="M20 15a9 9 0 0 1-11-11 9 9 0 1 0 11 11Z"/>',
  sparkles:
    '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3ZM20 2v4m-2-2h4"/>',
  thermometer:
    '<path d="M9 14V5a3 3 0 0 1 6 0v9a5 5 0 1 1-6 0Z"/><path d="M12 8v10"/>',
  panel:
    '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  upload: '<path d="M12 16V4m-5 5 5-5 5 5M4 16v5h16v-5"/>',
  copy: '<rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
  edit: '<path d="m14 5 5 5M3 21l5-1L21 7a2 2 0 0 0-4-4L4 16l-1 5Z"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  warning: '<path d="m12 3 10 18H2L12 3Z"/><path d="M12 9v4m0 3v1"/>',
  pin: '<path d="m9 3 9 9-3 1-2 5-7-7 5-2-2-6ZM8 16l-5 5"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
  play: '<path d="m8 4 13 8-13 8V4Z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1"/>',
  terminal: '<path d="m4 6 6 6-6 6m9 0h7"/>',
  more: '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
  sliders: '<path d="M4 7h6m4 0h6M4 17h10m4 0h2"/><circle cx="12" cy="7" r="2"/><circle cx="16" cy="17" r="2"/>',
  file: '<path d="M14 2H4v20h16V8l-6-6Zm0 0v6h6M8 13h8m-8 4h5"/>',
  paperclip:
    '<path d="m20.5 11.5-8.6 8.6a5.5 5.5 0 0 1-7.8-7.8l8.6-8.6a3.7 3.7 0 0 1 5.2 5.2l-8.6 8.6a1.8 1.8 0 0 1-2.6-2.6l7.9-7.9"/>',
};
function icon(name) {
  const span = document.createElement("span");
  span.className = "icon";
  span.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.chat}</svg>`;
  return span;
}
function hydrateIcons(root = document) {
  root
    .querySelectorAll("[data-icon]")
    .forEach((node) => node.replaceChildren(icon(node.dataset.icon)));
}
function text(tag, className, content = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = content;
  return node;
}
function button(label, className, action, iconName) {
  const node = text("button", className, "");
  node.type = "button";
  if (iconName) node.append(icon(iconName));
  node.append(document.createTextNode(label));
  node.onclick = action;
  return node;
}
function uid() {
  return (
    globalThis.crypto?.randomUUID?.() ||
    `${Date.now()}-${Math.random().toString(36).slice(2)}`
  );
}
function finite(value) {
  return (
    value !== null &&
    value !== undefined &&
    value !== "" &&
    Number.isFinite(Number(value))
  );
}
function formatNumber(value) {
  return finite(value)
    ? new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(
        Number(value),
      )
    : "—";
}
function formatMs(value) {
  return finite(value) ? `${(Number(value) / 1000).toFixed(1)} s` : "—";
}
function formatDate(value, withTime = true) {
  return value && Number.isFinite(new Date(value).getTime())
    ? new Intl.DateTimeFormat("en-US", {
        day: "numeric",
        month: "short",
        ...(withTime ? { hour: "2-digit", minute: "2-digit" } : {}),
      }).format(new Date(value))
    : "—";
}
function rate(stage) {
  if (!stage) return null;
  if (finite(stage.tokens_per_second)) return Number(stage.tokens_per_second);
  return stage.usage?.completion_tokens > 0 && stage.latency_ms > 0
    ? (stage.usage.completion_tokens * 1000) / stage.latency_ms
    : null;
}
function formatRate(stage) {
  const value = rate(stage);
  return value === null
    ? "—"
    : `${stage.tokens_per_second == null ? "≈" : ""}${formatNumber(value)} tok/s`;
}
function finalGeneration(response) {
  return response?.ghostswitch?.stages
    ?.filter((s) => s.phase === "generation")
    .at(-1);
}
function average(values) {
  const numbers = values.filter(finite).map(Number);
  return numbers.length
    ? numbers.reduce((a, b) => a + b, 0) / numbers.length
    : null;
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => $("toast").classList.add("hidden"), 4200);
}
async function copy(content) {
  try {
    await navigator.clipboard.writeText(content);
    toast("Copied");
  } catch {
    toast("Clipboard access was denied");
  }
}
function download(name, content, type = "application/json") {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function activeConversation() {
  return state.conversations.find((c) => c.id === state.activeId) || null;
}
function expertLabel(id) {
  return !id || id === "auto"
    ? "Automatic selection"
    : state.experts.find((e) => e.id === id)?.name || id;
}
function validateConversations(items) {
  if (!Array.isArray(items))
    throw new Error("The file must contain a list of conversations");
  const ids = new Set();
  return items.map((c) => {
    if (
      !c ||
      typeof c.id !== "string" ||
      !Array.isArray(c.messages) ||
      ids.has(c.id)
    )
      throw new Error("Invalid format or duplicate conversation IDs");
    ids.add(c.id);
    if (
      c.messages.some(
        (m) =>
          !m ||
          !["user", "assistant"].includes(m.role) ||
          typeof m.content !== "string" ||
          (m.response != null &&
            (typeof m.response !== "object" || Array.isArray(m.response))),
      )
    )
      throw new Error("Invalid conversation messages");
    return {
      id: c.id,
      title: String(c.title || "New chat").slice(0, 100),
      selection: typeof c.selection === "string" ? c.selection : "auto",
      pinned: Boolean(c.pinned),
      draft: typeof c.draft === "string" ? c.draft : "",
      createdAt: finite(c.createdAt)
        ? Number(c.createdAt)
        : c.updatedAt || Date.now(),
      updatedAt: finite(c.updatedAt) ? Number(c.updatedAt) : Date.now(),
      messages: c.messages.map((m) => ({
        ...m,
        response: renamedResponse(m.response),
        id: typeof m.id === "string" ? m.id : uid(),
        attachments: validAttachments(m.attachments),
        createdAt:
          m.createdAt ||
          (m.response?.created ? m.response.created * 1000 : c.updatedAt) ||
          Date.now(),
      })),
    };
  });
}
// Responses saved before the renames keep their telemetry under omnibrain or minibrain.
function renamedResponse(response) {
  if (!response || response.ghostswitch) return response;
  const { omnibrain, minibrain, ...rest } = response;
  const telemetry = omnibrain || minibrain;
  return telemetry ? { ...rest, ghostswitch: telemetry } : response;
}
function restore() {
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    state.conversations = validateConversations(data.conversations || []);
    state.activeId = state.conversations.some((c) => c.id === data.activeId)
      ? data.activeId
      : state.conversations[0]?.id || null;
    const saved = { ...data, ...data.settings };
    if (
      Number.isInteger(saved.maxTokens) &&
      saved.maxTokens >= 1 &&
      saved.maxTokens <= 10240
    )
      state.settings.maxTokens = saved.maxTokens;
    if (
      finite(saved.temperature) &&
      saved.temperature >= 0 &&
      saved.temperature <= 2
    )
      state.settings.temperature = Number(saved.temperature);
    // Upgrade the previous default once; keep other saved user values.
    if (data.generationDefaultsVersion !== 1 && state.settings.temperature === 0)
      state.settings.temperature = defaults.temperature;
    if (["dark", "light"].includes(saved.theme))
      state.settings.theme = saved.theme;
    if (typeof saved.systemPrompt === "string")
      state.settings.systemPrompt = saved.systemPrompt;
    if (saved.webSearch in WEB_SEARCH_MODES) state.settings.webSearch = saved.webSearch;
    if (saved.thinking in THINKING_MODES) state.settings.thinking = saved.thinking;
    if (saved.onboarded === true) state.settings.onboarded = true;
    if (typeof saved.baseUrl === "string") {
      const u = new URL(saved.baseUrl);
      if (["http:", "https:"].includes(u.protocol))
        state.settings.baseUrl = u.origin;
    }
    // Start the simplified interface with diagnostics closed once, then
    // preserve subsequent choices without changing conversations or drafts.
    if (data.interfaceVersion === 3 && typeof saved.inspector === "boolean")
      state.settings.inspector = saved.inspector;
    if (Array.isArray(data.experts))
      state.experts = data.experts.filter(
        (e) =>
          e &&
          typeof e.id === "string" &&
          typeof e.name === "string" &&
          typeof e.description === "string" &&
          (!e.capabilities || Array.isArray(e.capabilities)),
      );
    state.expertsAt = data.expertsAt || null;
  } catch {
    toast(
      "Could not read saved data. The original record remains in this browser.",
    );
  }
  state.view = views[location.hash.slice(1)]
    ? location.hash.slice(1)
    : "chat";
}
function persist() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        version: 2,
        interfaceVersion: 3,
        generationDefaultsVersion: 1,
        conversations: state.conversations,
        activeId: state.activeId,
        settings: state.settings,
        experts: state.experts,
        expertsAt: state.expertsAt,
      }),
    );
  } catch {
    toast(
      "Could not save changes. Check available browser storage.",
    );
  }
}
function saveDraft() {
  const c = activeConversation();
  if (c && state.view === "chat") c.draft = $("messageInput").value;
}
function navigate(view, preserveDraft = true) {
  if (preserveDraft) saveDraft();
  if (state.view === "api") savePlaygroundDraft();
  state.view = view;
  history.replaceState(null, "", `#${view}`);
  document.querySelectorAll(".floating-details").forEach((menu) => {
    menu.open = false;
  });
  closeSidebar();
  render();
}
function openConversation(id) {
  saveDraft();
  state.activeId = id;
  state.inspectorId = null;
  state.autoScroll = true;
  navigate("chat", false);
  $("messageInput").value = activeConversation()?.draft || "";
  resizeInput();
  persist();
}
function createConversation(selection = "auto") {
  if (typeof selection !== "string") selection = "auto";
  saveDraft();
  const c = {
    id: uid(),
    title: "New chat",
    selection,
    messages: [],
    draft: "",
    pinned: false,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  state.conversations.unshift(c);
  state.activeId = c.id;
  state.inspectorId = null;
  state.autoScroll = true;
  navigate("chat", false);
  $("messageInput").value = "";
  resizeInput();
  persist();
  $("messageInput").focus();
  return c;
}
function closeSidebar() {
  $("sidebar").classList.remove("open");
  $("sidebarScrim").classList.add("hidden");
}
function renameConversation(c) {
  if (!c) return;
  const value = prompt("Conversation title", c.title);
  if (value?.trim()) {
    c.title = value.trim().slice(0, 100);
    persist();
    render();
  }
}
function exportConversation(c) {
  if (!c) return toast("Create a conversation first");
  const md =
    `# ${c.title}\n\n` +
    c.messages
      .map(
        (m) =>
          `## ${m.role === "user" ? "You" : "GhostSwitch"}\n\n` +
          (m.attachments?.length
            ? `Attachments: ${m.attachments.map((a) => a.name).join(", ")}\n\n`
            : "") +
          `${m.content}\n`,
      )
      .join("\n");
  download(`ghostswitch-${c.id.slice(0, 8)}.md`, md, "text/markdown");
}
function removeConversation(c) {
  if (state.pending?.conversationId === c.id)
    return toast("Wait for the request to finish or stop generation");
  if (!confirm(`Delete “${c.title}” and its local statistics?`))
    return;
  state.conversations = state.conversations.filter((item) => item.id !== c.id);
  if (state.activeId === c.id) {
    state.activeId = state.conversations[0]?.id || null;
    $("messageInput").value = activeConversation()?.draft || "";
  }
  persist();
  render();
  pruneAttachments();
}
function records() {
  const result = [];
  for (const c of state.conversations) {
    let prompt = "";
    for (const m of c.messages) {
      if (m.role === "user") prompt = messageText(m);
      if (m.role === "assistant" && m.response)
        result.push({
          message: m,
          conversation: c,
          prompt,
          response: m.response,
          date: m.createdAt || c.updatedAt,
          source: m.source || c.source || "Chat",
        });
    }
  }
  for (const entry of state.serverRequests) {
    const request = entry.request || {};
    const response = entry.response || {};
    const source = entry.source || "API";
    result.push({
      external: true,
      source,
      message: { id: entry.id, request, response },
      conversation: { title: source },
      prompt: contentText([...(Array.isArray(request.messages) ? request.messages : [])]
        .reverse().find((m) => m.role === "user")?.content) || "Invalid request",
      response,
      date: entry.date,
    });
  }
  const seen = new Set();
  return result
    .sort(
      (a, b) =>
        Number(Boolean(a.message.sourceMessageId)) -
          Number(Boolean(b.message.sourceMessageId)) || b.date - a.date,
    )
    .filter((r) => {
      const key = r.response.id || r.message.sourceMessageId || r.message.id;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .sort((a, b) => b.date - a.date);
}
function statusLabel(status) {
  return (
    {
      success: "Accepted",
      failed: "Error",
      unsupported: "Expert needed",
      canceled: "Canceled",
      queued: "Queued",
      running: "Running",
    }[status] ||
    status ||
    "No data"
  );
}
function statusBadge(status) {
  return text(
    "span",
    `badge ${status === "success" ? "success" : ["canceled", "queued", "running", "unsupported"].includes(status) ? "neutral" : "danger"}`,
    statusLabel(status),
  );
}
// The reply is rendered as Markdown, so the expert ID goes in backticks to keep its underscores.
function unavailableMessage(expertId) {
  if (!expertId) return "No expert can answer this request.";
  const expert = state.experts.find((e) => e.id === expertId);
  const name = expert?.name ? `the **${expert.name}** expert (\`${expertId}\`)` : `the \`${expertId}\` expert`;
  return `This request fits ${name}, which is not downloaded yet. Download it to answer with the specialist, or answer with the base model.`;
}
// Below a reply that needs a missing expert: download it and answer, or answer with the base model.
function expertOffer(c, message) {
  const id = message.response.routing?.missing_capability;
  const expert = state.experts.find((e) => e.id === id);
  if (!expert) return null;
  const card = text("div", "expert-offer");
  card.dataset.messageId = message.id;
  const heading = text("div", "expert-offer-heading");
  heading.append(icon("layers"), text("strong", "", expert.name));
  card.append(heading);
  if (expert.description) card.append(text("p", "", expert.description));
  const progress = text("div", "expert-download-progress");
  renderExpertDownload(progress, expert);
  card.append(progress);
  if (expert.status === "error" && expert.error) card.append(text("p", "expert-download-error", expert.error));
  // Only the latest reply can be answered again; earlier ones keep the explanation.
  if (c.messages.at(-1) !== message) return card;
  const ready = expert.enabled && expert.downloaded;
  const busy = ["queued", "downloading"].includes(expert.status) || Boolean(message.offer);
  const actions = text("div", "expert-offer-actions");
  const primary = button(
    ready ? "Answer with this expert" : busy ? "Downloading…" : expert.status === "error" ? "Retry download" : "Download and answer",
    "primary-button",
    async () => {
      if (ready) return answerAgain(c, message, { expertId: id });
      message.offer = "download";
      persist();
      refreshExpertOffers();
      if (!(await manageExpert(id, "enable"))) {
        delete message.offer;
        persist();
        refreshExpertOffers();
      }
    },
    ready ? "arrowUp" : "download",
  );
  primary.disabled = Boolean(state.pending) || (busy && !ready);
  const base = button("Answer with base model", "secondary-button", () => answerAgain(c, message, { mode: "base" }), "sparkles");
  base.disabled = Boolean(state.pending);
  actions.append(primary, base);
  card.append(actions);
  return card;
}
// Rebuild visible offers with fresh download progress; answer once a requested download finishes.
function refreshExpertOffers() {
  const c = activeConversation();
  if (!c) return;
  for (const node of document.querySelectorAll(".expert-offer")) {
    const message = c.messages.find((m) => m.id === node.dataset.messageId);
    const fresh = message && expertOffer(c, message);
    if (fresh) node.replaceWith(fresh);
  }
  const last = c.messages.at(-1);
  if (!last?.offer || state.pending) return;
  const expert = state.experts.find((e) => e.id === last.response?.routing?.missing_capability);
  if (expert?.enabled && expert.downloaded) {
    delete last.offer;
    answerAgain(c, last, { expertId: expert.id });
  } else if (!expert?.enabled || expert.status === "error") {
    delete last.offer;
    persist();
    refreshExpertOffers();
  }
}
// Replace a reply and its question with a new answer to the same question.
async function answerAgain(c, message, route) {
  if (state.pending) return toast("A request is already running");
  const index = c.messages.indexOf(message);
  const question = c.messages[index - 1];
  if (question?.role !== "user") return toast("Could not find the original message");
  c.messages.splice(index - 1, 2);
  $("messageInput").value = question.content;
  state.attachments = [...(question.attachments || [])];
  await sendMessage({ preventDefault() {} }, route);
}
function pageHeading(target, eyebrow, title, description, actions = []) {
  const row = text("div", "page-heading");
  const block = text("div");
  if (eyebrow) block.append(text("span", "eyebrow", eyebrow));
  block.append(
    text("h1", "", title),
    text("p", "", description),
  );
  row.append(block);
  const buttons = text("div", "heading-actions");
  buttons.append(...actions);
  row.append(buttons);
  target.append(row);
}
function emptyState(title, description, action, iconName = "activity") {
  const block = text("div", "empty-state");
  block.append(
    icon(iconName),
    text("h3", "", title),
    text("p", "", description),
  );
  if (action) block.append(action);
  return block;
}
function panel(title, subtitle, action) {
  const node = text("section", "panel");
  const header = text("div", "panel-heading");
  const info = text("div");
  info.append(text("h2", "", title));
  if (subtitle) info.append(text("p", "", subtitle));
  header.append(info);
  if (action) header.append(action);
  node.append(header);
  return node;
}
function renderSidebar() {
  const list = $("conversationList");
  list.replaceChildren();
  const sorted = [...state.conversations].sort(
    (a, b) => Number(b.pinned) - Number(a.pinned) || b.updatedAt - a.updatedAt,
  );
  if (!sorted.length)
    list.append(
      text("p", "sidebar-empty", "Start a conversation to see it here."),
    );
  for (const c of sorted) {
    const row = text(
      "div",
      `conversation-row${state.view === "chat" && c.id === state.activeId ? " active" : ""}`,
    );
    const open = button(
      c.title,
      "conversation-open",
      () => openConversation(c.id),
      c.pinned ? "pin" : "chat",
    );
    open.title = c.title;
    open.ondblclick = () => renameConversation(c);
    const controls = text("div", "conversation-controls");
    const pin = button(
      "",
      "quiet-icon",
      () => {
        c.pinned = !c.pinned;
        persist();
        renderSidebar();
      },
      "pin",
    );
    pin.title = c.pinned ? "Unpin" : "Pin";
    pin.setAttribute("aria-label", pin.title);
    const remove = button(
      "",
      "quiet-icon",
      () => removeConversation(c),
      "trash",
    );
    remove.title = "Delete conversation";
    remove.setAttribute("aria-label", `Delete ${c.title}`);
    controls.append(pin, remove);
    row.append(open, controls);
    list.append(row);
  }
  $("conversationCount").textContent = state.conversations.length;
  document.querySelectorAll("[data-view]").forEach((node) => {
    node.classList.toggle("active", node.dataset.view === state.view);
    if (node.dataset.view === state.view)
      node.setAttribute("aria-current", "page");
    else node.removeAttribute("aria-current");
  });
  if (["overview", "requests", "api"].includes(state.view))
    $("toolsNav").open = true;
}
function applyTheme() {
  document.documentElement.dataset.theme = state.settings.theme;
  document.querySelector('meta[name="theme-color"]').content =
    state.settings.theme === "dark" ? "#121317" : "#f8f9fc";
  $("themeButton").replaceChildren(
    icon(state.settings.theme === "dark" ? "sun" : "moon"),
  );
}
function renderWebSearchToggle() {
  const mode = state.settings.webSearch;
  const toggle = $("webSearchToggle");
  toggle.classList.toggle("active", mode !== "off");
  toggle.title = `${WEB_SEARCH_MODES[mode]} (click to change). Searches send the query to the search provider.`;
  toggle.setAttribute("aria-label", WEB_SEARCH_MODES[mode]);
  $("webSearchMode").textContent = { off: "Off", auto: "Auto", on: "On" }[mode];
}
function renderThinkingToggle() {
  const mode = state.settings.thinking;
  const toggle = $("thinkingToggle");
  toggle.classList.toggle("active", mode === "on");
  toggle.title = `${THINKING_MODES[mode]} (click to change). Reasoning first makes calculations and puzzles more reliable, and answers slower.`;
  toggle.setAttribute("aria-label", THINKING_MODES[mode]);
  $("thinkingMode").textContent = { auto: "Auto", on: "On", off: "Off" }[mode];
}
function runtimeLabel(value = state.health) {
  if (!value) return "GhostSwitch";
  return (
    [value.backbone_name || value.name, value.backend?.toUpperCase()]
      .filter(Boolean)
      .join(" · ") || "GhostSwitch"
  );
}
function renderConnection() {
  const switching = state.switching;
  const setup = state.connected && !switching && setupRequired();
  $("connectionDot").className =
    `connection-dot ${switching ? "" : state.connected ? "ok" : "error"}`;
  $("connectionText").textContent = switching
    ? "Switching model…"
    : setup
      ? "No model yet"
      : state.connected
        ? "Server online"
        : "Server offline";
  const host = new URL(state.settings.baseUrl).host;
  $("connectionDetail").textContent = switching
    ? runtimeLabel(switching)
    : setup
      ? "Download a base model"
      : state.connected
        ? runtimeLabel()
        : host;
  $("connectionDetail").parentElement.title = `${host} · choose the base model and backend`;
  $("topbarStatus").textContent = switching
    ? "Switching model"
    : setup
      ? "Online · no model"
      : state.connected
        ? `Online · ${runtimeLabel()}`
        : "Offline";
  $("topbarStatus").classList.toggle("online", state.connected && !switching && !setup);
  $("composerNote").textContent = switching
    ? `Switching to ${runtimeLabel(switching)}. The server restarts and loads the model.`
    : setup
      ? "Download a base model to start chatting."
      : state.connected
        ? ""
        : "Server offline. Start the backend or check the URL in Settings.";
}
async function refreshRuntime() {
  try {
    const response = await fetch(`${state.settings.baseUrl}/v1/runtime`, {
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) throw new Error("Runtime list unavailable");
    state.runtime = await response.json();
    // Pick up a download that is still running, for example after a page reload.
    if (state.runtime.install) {
      state.install = state.runtime.install;
      if (INSTALL_ACTIVE.includes(state.install.status)) pollInstall();
    }
  } catch {
    state.runtime = null;
  }
  if (state.view === "settings") renderSettings();
  renderOnboarding();
}
async function switchRuntime(backbone, backend, ggufPath) {
  if (state.pending || state.playgroundBusy)
    return toast("Wait for the current request to finish");
  const body = { backbone, backend };
  if (ggufPath) body.gguf_path = ggufPath;
  let result;
  try {
    const response = await fetch(`${state.settings.baseUrl}/v1/runtime`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    result = await response.json();
    if (!response.ok) throw new Error(result.error?.message || `HTTP ${response.status}`);
  } catch (error) {
    return toast(`Could not switch: ${error.message}`);
  }
  if (result.status === "current") return toast(`Already running ${runtimeLabel(result)}`);
  state.switching = { ...result, startedAt: Date.now() };
  renderConnection();
  renderOnboarding();
  if (state.view === "settings") renderSettings();
  // The server replaces its process; wait until it reports the new runtime.
  while (state.switching && Date.now() - state.switching.startedAt < 300000) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    try {
      const health = await fetch(`${state.settings.baseUrl}/health`, {
        signal: AbortSignal.timeout(3000),
        cache: "no-store",
      }).then((r) => (r.ok ? r.json() : null));
      if (health?.backbone === backbone && health?.backend === backend && !health.setup_required) {
        state.switching = null;
        state.health = health;
        state.connected = true;
        state.experts = [];
        toast(`Now running ${runtimeLabel(health)}`);
        refreshServer();
        await refreshRuntime();
        renderConnection();
        renderOnboarding();
        return;
      }
    } catch {
      // Expected while the process restarts and loads the model.
    }
  }
  state.switching = null;
  renderConnection();
  renderOnboarding();
  toast("The server did not come back with the new model. Check its log.");
}
async function loadSearchSettings() {
  try {
    const response = await fetch(`${state.settings.baseUrl}/v1/search/settings`, {
      signal: AbortSignal.timeout(5000),
    });
    state.searchSettings = response.ok ? await response.json() : null;
  } catch {
    state.searchSettings = null;
  }
  loadSearchSettings.failed = !state.searchSettings;
  if (state.view === "settings") renderSettings();
}
function webSearchPanel() {
  const node = panel(
    "Web search",
    "Used when search is Auto or On in the chat. Queries go to the chosen provider; page text is treated as untrusted data.",
  );
  const current = state.searchSettings;
  if (!current) {
    // Retry only after the connection check succeeds again, never in a render loop.
    if (!loadSearchSettings.loading && !loadSearchSettings.failed) {
      loadSearchSettings.loading = true;
      loadSearchSettings().finally(() => (loadSearchSettings.loading = false));
    }
    node.append(
      text("p", "muted", loadSearchSettings.failed ? "Unavailable while the server is offline." : "Loading…"),
    );
    return node;
  }
  const label = text("label", "field");
  label.append(text("span", "", "Provider"));
  const select = document.createElement("select");
  select.id = "searchProvider";
  for (const [value, name] of [
    ["duckduckgo", "DuckDuckGo (no key)"],
    ["brave", "Brave Search API (key)"],
    ["searxng", "SearXNG (self-hosted)"],
  ]) {
    const option = text("option", "", name);
    option.value = value;
    option.selected = current.provider === value;
    select.append(option);
  }
  label.append(select);
  node.append(
    label,
    field(
      "Brave API key",
      "searchBraveKey",
      "password",
      "",
      current.brave_api_key_set ? "A key is saved; leave empty to keep it." : "Free tier: 2,000 queries a month.",
      { autocomplete: "off", placeholder: current.brave_api_key_set ? "••••••••" : "" },
    ),
    field("SearXNG URL", "searchSearxng", "url", current.searxng_url || "", "For example, http://127.0.0.1:8888"),
  );
  const actions = text("div", "search-actions");
  actions.append(
    button(
      "Save",
      "secondary-button",
      async () => {
        const update = { provider: $("searchProvider").value, searxng_url: $("searchSearxng").value.trim() };
        if ($("searchBraveKey").value.trim()) update.brave_api_key = $("searchBraveKey").value.trim();
        try {
          const response = await fetch(`${state.settings.baseUrl}/v1/search/settings`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(update),
          });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error?.message || `HTTP ${response.status}`);
          state.searchSettings = result;
          toast("Search settings saved");
          renderSettings();
        } catch (error) {
          toast(`Could not save: ${error.message}`);
        }
      },
      "check",
    ),
    button(
      "Test search",
      "secondary-button",
      async () => {
        toast("Searching…");
        try {
          const response = await fetch(`${state.settings.baseUrl}/v1/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: "GhostSwitch web search test" }),
          });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error?.message || result.error || `HTTP ${response.status}`);
          const read = result.sources.filter((source) => source.read).length;
          toast(`${result.provider}: ${result.sources.length} results, ${read} pages read in ${formatMs(result.latency_ms)}`);
        } catch (error) {
          toast(`Search failed: ${error.message}`);
        }
      },
      "search",
    ),
  );
  node.append(actions);
  return node;
}
function runtimePanel() {
  const node = panel(
    "Base model",
    "Each base model has its own expert library. Switching restarts the server and loads the model.",
    button("Refresh", "secondary-button", refreshRuntime, "refresh"),
  );
  node.classList.add("runtime-panel");
  const runtime = state.runtime;
  if (!runtime) {
    node.append(
      text(
        "p",
        "muted",
        renderSettings.loading ? "Loading models…" : "Models are unavailable while the server is offline.",
      ),
    );
    return node;
  }
  const memory = runtime.memory || {};
  const free = memory.available_gb;
  const releasable = memory.releasable_gb || 0;
  if (memory.total_gb)
    node.append(
      text(
        "p",
        "runtime-memory",
        free == null
          ? `This computer has ${memory.total_gb} GB of memory.`
          : `This computer has ${memory.total_gb} GB of memory; about ${free} GB is free now` +
              (releasable ? `, plus ${releasable} GB from the current model after a switch.` : "."),
      ),
    );
  const list = text("div", "runtime-options");
  for (const option of runtime.options) {
    const row = text("div", "runtime-option");
    const info = text("div", "runtime-name");
    info.append(text("strong", "", option.name));
    if (option.default) info.append(text("small", "", "Default"));
    if (option.uncensored) info.append(text("small", "", "Uncensored"));
    row.append(info);
    const backends = text("div", "runtime-backends");
    for (const backend of ["mlx", "gguf"]) {
      const status = option.backends[backend];
      const current =
        runtime.current.backbone === option.id && runtime.current.backend === backend;
      const choice = button(
        backend.toUpperCase(),
        `runtime-choice${current ? " selected" : ""}`,
        () => {
          const path =
            backend === "gguf" && !status.available
              ? $(`gguf-${option.id}`)?.value.trim()
              : null;
          if (!status.installed && !path)
            return openOnboarding(ONBOARDING_STEPS.length - 1, { backbone: option.id, backend });
          switchRuntime(option.id, backend, path || null);
        },
      );
      choice.disabled = Boolean(state.switching) || current;
      choice.setAttribute("aria-pressed", String(current));
      // Show the GGUF file name; the full path stays in the tooltip.
      const detail = !status.installed && !status.path
        ? `Not downloaded · ${status.download_gb} GB`
        : status.available && status.path ? status.path.split("/").pop() : status.detail;
      choice.title = current ? "Current model" : status.path || status.detail;
      const cell = text("div", "runtime-backend");
      const note = text("small", status.available || !status.installed ? "" : "unavailable", current ? "Current" : detail);
      note.title = status.path || status.detail;
      cell.append(choice, note);
      if (status.memory_gb) {
        // Warn before a switch that would push other programs into swap.
        const tight = !current && free != null && status.memory_gb > free + releasable;
        cell.append(
          text(
            "small",
            tight ? "unavailable" : "",
            current
              ? runtime.current.loaded
                ? `Uses about ${status.memory_gb} GB`
                : `Needs about ${status.memory_gb} GB once loaded`
              : tight
                ? `Needs about ${status.memory_gb} GB; close other apps first`
                : `Needs about ${status.memory_gb} GB`,
          ),
        );
      }
      if (backend === "gguf" && !status.available && status.detail.includes("enter its path"))
        cell.append(field("GGUF file", `gguf-${option.id}`, "text", "", "", { placeholder: "/path/to/model.gguf" }));
      backends.append(cell);
    }
    row.append(backends);
    list.append(row);
  }
  node.append(list);
  return node;
}
// --- First run: an introduction, then a base model to download ----------------
const ONBOARDING_STEPS = ["Welcome", "How it works", "Experts and search", "Base model"];
const INSTALL_ACTIVE = ["starting", "downloading", "verifying"];
function setupRequired() {
  return Boolean(state.health?.setup_required);
}
function openOnboarding(step = 0, choice = {}) {
  state.onboarding = { step, backbone: choice.backbone || null, backend: choice.backend || null };
  if ($("onboardingOverlay").classList.contains("hidden")) {
    dialogReturnFocus = document.activeElement;
    $("onboardingOverlay").classList.remove("hidden");
  }
  if (step === ONBOARDING_STEPS.length - 1) refreshRuntime();
  renderOnboarding();
  $("onboardingOverlay").querySelector(".dialog").focus();
}
// Without a model the chat cannot answer, so the setup stays open until one runs.
function onboardingLocked() {
  return state.connected && setupRequired();
}
function closeOnboarding() {
  if (onboardingLocked()) {
    if (state.onboarding?.step !== ONBOARDING_STEPS.length - 1) showOnboardingStep(ONBOARDING_STEPS.length - 1);
    return toast("Download a base model to continue");
  }
  state.settings.onboarded = true;
  state.onboarding = null;
  persist();
  $("onboardingOverlay").classList.add("hidden");
  dialogReturnFocus?.focus();
}
function finishOnboarding() {
  closeOnboarding();
  if (state.onboarding) return;
  navigate("chat");
  $("messageInput").focus();
}
function showOnboardingStep(step) {
  state.onboarding.step = step;
  if (step === ONBOARDING_STEPS.length - 1) refreshRuntime();
  renderOnboarding();
  $("onboardingBody").scrollTop = 0;
  $("onboardingOverlay").querySelector(".dialog").focus();
}
function onboardingKey() {
  return `${state.connected}:${setupRequired()}`;
}
function renderOnboarding() {
  const o = state.onboarding;
  if (!o) return;
  renderOnboarding.key = onboardingKey();
  const last = ONBOARDING_STEPS.length - 1;
  $("onboardingStepLabel").textContent = `Step ${o.step + 1} of ${ONBOARDING_STEPS.length} · ${ONBOARDING_STEPS[o.step]}`;
  $("onboardingDots").replaceChildren(
    ...ONBOARDING_STEPS.map((_, i) => text("span", i === o.step ? "active" : i < o.step ? "done" : "")),
  );
  $("closeOnboarding").classList.toggle("hidden", onboardingLocked());
  $("onboardingBody").replaceChildren(
    [onboardingWelcome, onboardingFlow, onboardingExperts, onboardingModel][o.step](),
  );
  const footer = $("onboardingFooter");
  footer.replaceChildren();
  const left = text("div", "onboarding-footer-side");
  const right = text("div", "onboarding-footer-side");
  if (o.step > 0)
    left.append(button("Back", "secondary-button", () => showOnboardingStep(o.step - 1)));
  if (o.step < last) {
    left.append(
      button("Skip introduction", "text-button", () =>
        setupRequired() || !state.connected ? showOnboardingStep(last) : finishOnboarding(),
      ),
    );
    right.append(button("Next", "primary-button", () => showOnboardingStep(o.step + 1), "arrowRight"));
  } else {
    if (!onboardingLocked() && !setupRequired())
      left.append(button("Close", "text-button", finishOnboarding));
    right.append(onboardingAction());
  }
  footer.append(left, right);
}
function onboardingFeature(iconName, title, body) {
  const node = text("div", "onboarding-feature");
  const heading = text("div", "onboarding-feature-title");
  heading.append(icon(iconName), text("strong", "", title));
  node.append(heading, text("p", "", body));
  return node;
}
function onboardingHeading(node, title, lead) {
  const heading = text("h2", "", title);
  heading.id = "onboardingTitle";
  node.append(heading, text("p", "onboarding-lead", lead));
}
function onboardingWelcome() {
  const node = text("div", "onboarding-step");
  onboardingHeading(
    node,
    "Welcome to GhostSwitch",
    "A private assistant that runs on this computer. Conversations stay here; only web searches and downloads go online.",
  );
  const grid = text("div", "onboarding-features");
  grid.append(
    onboardingFeature("brain", "One model, many experts",
      "A Qwen3.5 base model answers general questions. Small LoRA experts add skills such as coding, finance, or medicine."),
    onboardingFeature("check", "Checked answers",
      "Code checks every answer before you see it (arithmetic from your message, requested JSON fields, cited sources, cut-off text) and has it written again when a check fails."),
    onboardingFeature("lock", "Runs locally",
      "MLX on Apple Silicon, or llama.cpp (GGUF) on Windows, Linux, and Mac. Tools such as OpenCode can use it through an OpenAI-compatible API."),
  );
  node.append(grid);
  return node;
}
function onboardingFlow() {
  const node = text("div", "onboarding-step");
  onboardingHeading(node, "How an answer is made", "Every message goes through the same four steps.");
  const list = text("ol", "onboarding-flow");
  for (const [iconName, title, body] of [
    ["layers", "Route", "The router reads your message and picks the expert that fits, or the base model for general questions."],
    ["globe", "Search", "Optional. With web search on, GhostSwitch reads a few pages and cites them as [1], [2]."],
    ["sparkles", "Answer", "The base model writes the answer with the expert attached. For calculations, puzzles, plans and programming tasks it thinks step by step first; the brain switch in the composer changes that."],
    ["check", "Check", "Code checks the answer. If a check fails, the answer is written again, up to five attempts."],
  ]) {
    const item = text("li", "onboarding-flow-step");
    const marker = text("span", "onboarding-flow-icon");
    marker.append(icon(iconName));
    const copy = text("div");
    copy.append(text("strong", "", title), text("p", "", body));
    item.append(marker, copy);
    list.append(item);
  }
  node.append(list, text("p", "onboarding-note", "Open Details under any reply to see each step, how long it took, and why an answer was written again."));
  return node;
}
function onboardingExperts() {
  const node = text("div", "onboarding-step");
  onboardingHeading(node, "Experts and web search", "Add skills when you need them, and look things up when facts change.");
  const grid = text("div", "onboarding-features");
  grid.append(
    onboardingFeature("layers", "Experts on demand",
      "A few common experts download right after the base model; the rest the first time you need them, 30–700 MB each. When a question fits an expert you don't have, the reply offers to download it or to answer with the base model. Manage them under Experts."),
    onboardingFeature("globe", "Web search",
      "The globe button in the message box switches search Off, Auto, or On. Auto searches for news, prices, and other current facts. Queries go to the provider chosen in Settings."),
    onboardingFeature("file", "Your data",
      "Conversations are saved in this browser. Settings can export them as JSON or import them on another computer."),
  );
  node.append(grid);
  return node;
}
function hardwareFact(label, value) {
  const node = text("div", "hardware-fact");
  node.append(text("span", "", label), text("strong", "", value));
  return node;
}
const FIT_LABELS = {
  comfortable: ["Runs comfortably", "good"],
  tight: ["Tight: close other apps first", "warn"],
  too_large: ["Too large for this computer", "bad"],
};
function onboardingModel() {
  const node = text("div", "onboarding-step");
  const runtime = state.runtime;
  if (!state.connected) {
    onboardingHeading(node, "Choose a base model", "The GhostSwitch server is not running, so models cannot be downloaded yet. Start it, and this page continues on its own:");
    node.append(text("pre", "onboarding-command", "./.venv/bin/python -m core.start_mlx --serve"));
    return node;
  }
  if (!runtime) {
    onboardingHeading(node, "Choose a base model", "Checking this computer…");
    return node;
  }
  const o = state.onboarding;
  const recommended = runtime.recommended;
  o.backend ||= recommended.backend;
  o.backbone ||= recommended.backbone;
  onboardingHeading(
    node,
    setupRequired() ? "Choose a base model" : "Base models",
    setupRequired()
      ? "GhostSwitch needs one base model to answer. It is downloaded once from Hugging Face; you can add the other one later in Settings."
      : `${runtimeLabel(state.health)} is running. You can download the other model here and switch in Settings.`,
  );
  const hw = runtime.hardware;
  const facts = text("div", "hardware-facts");
  facts.append(
    hardwareFact("Processor", hw.chip || `${hw.platform} ${hw.machine}`),
    hardwareFact("Memory", `${hw.memory.total_gb} GB`),
  );
  if (hw.gpu?.memory_gb) facts.append(hardwareFact("GPU", `${hw.gpu.name}, ${hw.gpu.memory_gb} GB`));
  facts.append(hardwareFact("Free disk", `${hw.disk_free_gb} GB`));
  node.append(facts);
  const backends = text("div", "segmented");
  backends.setAttribute("role", "radiogroup");
  backends.setAttribute("aria-label", "Backend");
  for (const [id, name, note] of [
    ["mlx", "MLX", "Apple Silicon; fastest on a Mac"],
    ["gguf", "GGUF", "llama.cpp; any computer"],
  ]) {
    const choice = button("", `segment${o.backend === id ? " selected" : ""}`, () => {
      o.backend = id;
      renderOnboarding();
    });
    choice.setAttribute("role", "radio");
    choice.setAttribute("aria-checked", String(o.backend === id));
    choice.append(text("strong", "", name), text("small", "", hw.backends_installed[id] ? note : "Not installed in this Python environment"));
    choice.disabled = Boolean(state.switching) || INSTALL_ACTIVE.includes(state.install?.status);
    backends.append(choice);
  }
  node.append(backends);
  const cards = text("div", "model-cards");
  cards.setAttribute("role", "radiogroup");
  cards.setAttribute("aria-label", "Base model");
  for (const option of runtime.options) {
    const status = option.backends[o.backend];
    const selected = o.backbone === option.id;
    const card = button("", `model-card${selected ? " selected" : ""}`, () => {
      o.backbone = option.id;
      renderOnboarding();
    });
    card.setAttribute("role", "radio");
    card.setAttribute("aria-checked", String(selected));
    card.disabled = Boolean(state.switching) || INSTALL_ACTIVE.includes(state.install?.status);
    const title = text("div", "model-card-title");
    title.append(text("strong", "", option.name));
    if (recommended.backbone === option.id && recommended.backend === o.backend)
      title.append(text("span", "badge success", "Recommended"));
    if (option.uncensored) title.append(text("span", "badge neutral", "Uncensored"));
    const running = !setupRequired() && runtime.current.backbone === option.id && runtime.current.backend === o.backend;
    if (running) title.append(text("span", "badge neutral", "Running"));
    card.append(
      title,
      text("p", "", option.id === "qwen35-9b"
        ? `Better answers and the full library of ${option.experts} experts.`
        : `Faster and lighter, with ${option.experts} experts.`),
    );
    const meta = text("div", "model-card-meta");
    meta.append(
      text("span", "", status.installed ? "Downloaded" : `Download ${status.download_gb} GB`),
      text("span", "", status.memory_gb ? `Needs about ${status.memory_gb} GB of memory` : "Memory need unknown"),
    );
    card.append(meta);
    const [fitLabel, fitClass] = FIT_LABELS[status.fit] || [];
    if (fitLabel) card.append(text("span", `fit ${fitClass}`, fitLabel));
    if (!status.installed && !status.enough_disk)
      card.append(text("span", "fit bad", `Needs ${status.download_gb + 1} GB of free disk`));
    cards.append(card);
  }
  node.append(cards, text("p", "onboarding-note", `Recommended for this computer: ${recommended.reason}`));
  const progress = text("div", "install-progress");
  progress.id = "installProgress";
  renderInstallProgress(progress);
  node.append(progress);
  return node;
}
function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "";
  if (seconds < 90) return "less than 2 min";
  if (seconds < 3600) return `about ${Math.round(seconds / 60)} min`;
  return `about ${(seconds / 3600).toFixed(1)} h`;
}
function renderInstallProgress(target = $("installProgress")) {
  if (!target) return;
  target.replaceChildren();
  const job = state.install;
  const o = state.onboarding;
  if (state.switching) {
    target.append(text("strong", "", `Starting ${runtimeLabel(state.switching)}…`),
      text("small", "", "The server restarts and loads the model. This takes up to a minute."));
    return;
  }
  if (!job || job.status === "done" || job.backbone !== o?.backbone || job.backend !== o?.backend) {
    target.hidden = true;
    return;
  }
  target.hidden = false;
  const name = `${job.name || job.backbone} · ${job.backend.toUpperCase()}`;
  if (job.status === "error" || job.status === "canceled") {
    target.append(
      text("strong", job.status === "error" ? "install-error" : "", job.status === "error" ? `Download of ${name} failed` : `Download of ${name} canceled`),
      text("small", "", job.status === "error" ? `${job.error || "Unknown error"}. Downloaded parts are kept, so trying again continues where it stopped.` : "Downloaded parts are kept; download again to continue."),
    );
    return;
  }
  const total = job.total_bytes;
  const done = job.downloaded_bytes || 0;
  target.append(text("strong", "", job.status === "verifying" ? `Checking ${name}…` : `Downloading ${name}`));
  const bar = document.createElement("progress");
  bar.max = 100;
  bar.setAttribute("aria-label", "Download progress");
  if (total) bar.value = Math.min(100, (100 * done) / total);
  target.append(bar);
  if (job.status !== "verifying") {
    const speed = job.bytes_per_second || 0;
    const left = total && speed ? formatDuration((total - done) / speed) : "";
    target.append(text("small", "download-amount", [
      total ? `${formatDownloadBytes(done)} of ${formatDownloadBytes(total)}` : "Preparing…",
      speed ? `${formatDownloadBytes(speed)}/s` : "",
      left ? `${left} left` : "",
    ].filter(Boolean).join(" · ")));
    if (job.file) target.append(text("small", "download-file", job.file));
  }
}
function onboardingAction() {
  const runtime = state.runtime;
  const o = state.onboarding;
  if (!state.connected || !runtime || !o.backbone) {
    const wait = button("Waiting for the server…", "primary-button", null);
    wait.disabled = true;
    return wait;
  }
  const option = runtime.options.find((row) => row.id === o.backbone);
  const status = option.backends[o.backend];
  const job = state.install;
  if (state.switching) {
    const starting = button("Starting…", "primary-button", null);
    starting.disabled = true;
    return starting;
  }
  if (INSTALL_ACTIVE.includes(job?.status))
    return button("Cancel download", "secondary-button", cancelInstall, "close");
  const running = !setupRequired() && runtime.current.backbone === o.backbone && runtime.current.backend === o.backend;
  if (running) return button("Start chatting", "primary-button", finishOnboarding, "arrowRight");
  if (status.installed)
    return button(`Use ${option.name}`, "primary-button", () => switchRuntime(o.backbone, o.backend), "arrowRight");
  const label = ["error", "canceled"].includes(job?.status) && job.backbone === o.backbone && job.backend === o.backend
    ? "Continue download"
    : `Download ${option.name} · ${status.download_gb} GB`;
  const download = button(label, "primary-button", () => startInstall(o.backbone, o.backend), "download");
  download.disabled = !runtime.hardware.backends_installed[o.backend] || !status.enough_disk;
  return download;
}
async function installRequest(body) {
  const response = await fetch(`${state.settings.baseUrl}/v1/runtime/install`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error?.message || `HTTP ${response.status}`);
  return result.install;
}
async function startInstall(backbone, backend) {
  try {
    state.install = await installRequest({ backbone, backend });
  } catch (error) {
    return toast(`Could not start the download: ${error.message}`);
  }
  if (state.install?.status === "done") return afterInstall(state.install);
  renderOnboarding();
  pollInstall();
}
async function cancelInstall() {
  try {
    state.install = await installRequest({ action: "cancel" });
  } catch (error) {
    toast(`Could not cancel: ${error.message}`);
  }
  renderOnboarding();
}
async function pollInstall() {
  if (pollInstall.running) return;
  pollInstall.running = true;
  try {
    while (INSTALL_ACTIVE.includes(state.install?.status)) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      try {
        const response = await fetch(`${state.settings.baseUrl}/v1/runtime/install`, {
          signal: AbortSignal.timeout(5000),
          cache: "no-store",
        });
        const previous = state.install?.status;
        state.install = (await response.json()).install;
        // Redraw the whole step only when the stage changes; progress updates in place.
        if (state.install?.status !== previous) renderOnboarding();
        else renderInstallProgress();
      } catch {
        // The server may be busy; keep waiting.
      }
    }
  } finally {
    pollInstall.running = false;
  }
  if (state.install?.status === "done") afterInstall(state.install);
}
async function afterInstall(job) {
  await refreshRuntime();
  toast(`${job.name || job.backbone} downloaded`);
  // A server without a model starts the new one right away.
  if (setupRequired()) await switchRuntime(job.backbone, job.backend);
  renderOnboarding();
}
function renderHeader() {
  $("viewTitle").textContent =
    state.view === "chat"
      ? activeConversation()?.title || "New chat"
      : views[state.view];
  $("chatActions").classList.toggle("hidden", state.view !== "chat");
  for (const [key, id] of Object.entries({
    overview: "overviewView",
    chat: "chatView",
    experts: "expertsView",
    requests: "requestsView",
    api: "docsView",
    settings: "settingsView",
  }))
    $(id).classList.toggle("hidden", state.view !== key);
  const c = activeConversation();
  const choice = c?.selection || "auto";
  $("composerMode").textContent =
    choice === "auto" ? "Auto · GhostSwitch" : expertLabel(choice);
  renderWebSearchToggle();
  renderThinkingToggle();
  $("conversationSubtitle").textContent =
    `${c?.messages.length || 0} messages · ${choice === "auto" ? "automatic routing" : "manual LoRA selection"}`;
  $("temperature").value = state.settings.temperature.toFixed(1);
  $("maxTokens").value = state.settings.maxTokens;
  const mine = state.pending?.conversationId === state.activeId;
  $("messageInput").disabled = Boolean(state.pending || state.playgroundBusy);
  $("sendButton").disabled = Boolean(
    state.playgroundBusy || (state.pending && !mine),
  );
  $("sendButton").type = mine ? "button" : "submit";
  $("sendButton").replaceChildren(icon(mine ? "stop" : "arrowUp"));
  $("sendButton").setAttribute(
    "aria-label",
    mine ? "Stop generation" : "Send",
  );
  $("sendButton").onclick = mine ? stopGeneration : null;
  $("chatView").classList.toggle("inspector-hidden", !state.settings.inspector);
  const inspectorVisible =
    state.settings.inspector &&
    (!matchMedia("(max-width: 1000px)").matches || state.inspectorMobileOpen);
  $("inspector").classList.toggle("open", inspectorVisible);
  $("inspectorButton").setAttribute("aria-pressed", String(inspectorVisible));
  renderConnection();
}
function render() {
  applyTheme();
  renderSidebar();
  renderHeader();
  if (state.view === "overview") renderOverview();
  if (state.view === "chat") {
    renderMessages();
    renderInspector();
  }
  if (state.view === "experts") renderCatalog();
  if (state.view === "requests") renderRequests();
  if (state.view === "api") renderDocs();
  if (state.view === "settings") renderSettings();
  renderExperts();
}
function renderInline(parent, source) {
  const token =
    /(\$\$[^$]+?\$\$|\$(?!\$)[^\n$]+?\$(?!\$)|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\)|https?:\/\/[^\s<>]+|`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|~~[^~\n]+~~|\*[^*\n]+\*|_[^_\n]+_)/g;
  let cursor = 0;
  for (const match of source.matchAll(token)) {
    if (match.index > cursor)
      parent.append(document.createTextNode(source.slice(cursor, match.index)));
    const value = match[0];
    const link = value.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
    if (value.startsWith("$$") || value.startsWith("$")) {
      renderMath(
        parent,
        value.slice(
          value.startsWith("$$") ? 2 : 1,
          value.startsWith("$$") ? -2 : -1,
        ),
        value.startsWith("$$"),
      );
    } else if (link) {
      const anchor = text("a", "markdown-link", link[1]);
      anchor.href = link[2];
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      parent.append(anchor);
    } else if (/^https?:\/\//.test(value)) {
      const anchor = text("a", "markdown-link", value);
      anchor.href = value.replace(/[.,!?;:]+$/, "");
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      parent.append(anchor);
    } else if (value.startsWith("`")) {
      parent.append(text("code", "inline-code", value.slice(1, -1)));
    } else if (value.startsWith("**") || value.startsWith("__")) {
      parent.append(text("strong", "", value.slice(2, -2)));
    } else if (value.startsWith("~~")) {
      parent.append(text("del", "", value.slice(2, -2)));
    } else {
      parent.append(text("em", "", value.slice(1, -1)));
    }
    cursor = match.index + value.length;
  }
  if (cursor < source.length)
    parent.append(document.createTextNode(source.slice(cursor)));
}

function renderMath(parent, source, displayMode) {
  if (!window.katex) {
    parent.append(
      document.createTextNode(
        `${displayMode ? "$$" : "$"}${source}${displayMode ? "$$" : "$"}`,
      ),
    );
    return;
  }
  const node = document.createElement(displayMode ? "div" : "span");
  node.className = displayMode ? "math-display" : "math-inline";
  try {
    window.katex.render(source, node, {
      displayMode,
      throwOnError: false,
      strict: "ignore",
    });
  } catch {
    node.textContent = `${displayMode ? "$$" : "$"}${source}${displayMode ? "$$" : "$"}`;
    node.classList.add("math-error");
  }
  parent.append(node);
}

function markdownCells(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim().replace(/\\\|/g, "|"));
}

function isTableDelimiter(line) {
  const cells = markdownCells(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function isListItem(line) {
  return /^\s{0,3}(?:[-+*]|\d+[.)])\s+/.test(line);
}

function listItemInfo(line) {
  const match = line.match(/^(\s*)([-+*]|\d+[.)])\s+(.*)$/);
  if (!match) return null;
  return {
    indent: match[1].length,
    ordered: /^\d/.test(match[2]),
    content: match[3],
  };
}

function renderList(container, lines, getIndex, indent) {
  const first = listItemInfo(lines[getIndex()]);
  const list = document.createElement(first.ordered ? "ol" : "ul");
  while (getIndex() < lines.length) {
    const item = listItemInfo(lines[getIndex()]);
    if (!item || item.indent !== indent || item.ordered !== first.ordered)
      break;
    const li = document.createElement("li");
    const task = item.content.match(/^\[([ xX])\]\s+(.*)$/);
    if (task) {
      li.className = "task-item";
      li.append(
        text("span", "task-check", task[1].toLowerCase() === "x" ? "☑" : "☐"),
      );
      renderInline(li, task[2]);
    } else renderInline(li, item.content);
    getIndex(getIndex() + 1);
    while (getIndex() < lines.length) {
      const nested = listItemInfo(lines[getIndex()]);
      if (!nested || nested.indent <= indent) break;
      li.append(
        renderList(
          document.createDocumentFragment(),
          lines,
          getIndex,
          nested.indent,
        ),
      );
    }
    list.append(li);
  }
  container.append(list);
  return list;
}

function renderMarkdownBlocks(container, source) {
  const lines = String(source).replace(/\r\n?/g, "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index++;
      continue;
    }

    const displayMath = line.match(/^\s{0,3}\$\$(.*?)\$\$\s*$/);
    if (displayMath) {
      renderMath(container, displayMath[1], true);
      index++;
      continue;
    }
    if (/^\s{0,3}\$\$\s*$/.test(line)) {
      const mathLines = [];
      index++;
      while (index < lines.length && !/^\s{0,3}\$\$\s*$/.test(lines[index]))
        mathLines.push(lines[index++]);
      if (index < lines.length) index++;
      renderMath(container, mathLines.join("\n"), true);
      continue;
    }

    const fence = line.match(/^\s{0,3}(```+|~~~+)\s*([^\s`]*)?.*$/);
    if (fence) {
      const marker = fence[1].slice(0, 3);
      const codeLines = [];
      index++;
      while (
        index < lines.length &&
        !new RegExp(`^\\s{0,3}${marker[0]}{3,}\\s*$`).test(lines[index])
      ) {
        codeLines.push(lines[index++]);
      }
      if (index < lines.length) index++;
      const pre = document.createElement("pre");
      pre.className = "markdown-code";
      const language = fence[2]
        ? text("span", "code-language", fence[2])
        : null;
      const code = text("code", "", codeLines.join("\n"));
      const copy = text("button", "copy-code", "Copy");
      copy.type = "button";
      copy.onclick = () =>
        navigator.clipboard.writeText(code.textContent).then(
          () => toast("Code copied"),
          () => toast("Could not copy code"),
        );
      if (language) pre.append(language);
      pre.append(copy, code);
      container.append(pre);
      continue;
    }

    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const node = document.createElement(`h${heading[1].length}`);
      renderInline(node, heading[2]);
      container.append(node);
      index++;
      continue;
    }

    if (/^\s{0,3}(?:[-*_]\s*){3,}$/.test(line)) {
      container.append(document.createElement("hr"));
      index++;
      continue;
    }

    if (/^\s{0,3}>/.test(line)) {
      const quoteLines = [];
      while (index < lines.length && /^\s{0,3}>/.test(lines[index])) {
        quoteLines.push(lines[index++].replace(/^\s{0,3}> ?/, ""));
      }
      const quote = document.createElement("blockquote");
      renderMarkdownBlocks(quote, quoteLines.join("\n"));
      container.append(quote);
      continue;
    }

    if (
      line.includes("|") &&
      index + 1 < lines.length &&
      isTableDelimiter(lines[index + 1])
    ) {
      const headers = markdownCells(line);
      const rows = [];
      index += 2;
      while (
        index < lines.length &&
        lines[index].includes("|") &&
        lines[index].trim()
      )
        rows.push(markdownCells(lines[index++]));
      const table = document.createElement("table");
      table.className = "markdown-table";
      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      for (const value of headers) {
        const cell = document.createElement("th");
        renderInline(cell, value);
        headRow.append(cell);
      }
      thead.append(headRow);
      table.append(thead);
      const tbody = document.createElement("tbody");
      for (const row of rows) {
        const tr = document.createElement("tr");
        headers.forEach((_, cellIndex) => {
          const cell = document.createElement("td");
          renderInline(cell, row[cellIndex] || "");
          tr.append(cell);
        });
        tbody.append(tr);
      }
      table.append(tbody);
      container.append(table);
      continue;
    }

    if (isListItem(line)) {
      renderList(
        container,
        lines,
        (value) => {
          if (value !== undefined) index = value;
          return index;
        },
        listItemInfo(line).indent,
      );
      continue;
    }

    const paragraph = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^\s{0,3}(?:#{1,6}\s|```|~~~|>|(?:[-*_]\s*){3,}$)/.test(lines[index]) &&
      !isListItem(lines[index]) &&
      !(
        lines[index].includes("|") &&
        index + 1 < lines.length &&
        isTableDelimiter(lines[index + 1])
      )
    ) {
      paragraph.push(lines[index++]);
    }
    if (!paragraph.length) {
      paragraph.push(lines[index++]);
    }
    const p = document.createElement("p");
    paragraph.forEach((part, partIndex) => {
      const hardBreak = / {2,}$/.test(part) || /\\$/.test(part);
      renderInline(p, part.replace(/(?: {2,}|\\)$/, ""));
      if (partIndex < paragraph.length - 1)
        p.append(
          hardBreak
            ? document.createElement("br")
            : document.createTextNode(" "),
        );
    });
    container.append(p);
  }
}

function renderContent(container, content) {
  renderMarkdownBlocks(container, content || "");
}

function renderWelcome(target) {
  const welcome = text("div", "welcome");
  const mark = text("div", "welcome-mark");
  mark.append(icon("brain"));
  welcome.append(
    mark,
    text("h1", "", "What shall we work on today?"),
    text(
      "p",
      "",
      "Get help with code, ideas, and complex questions.",
    ),
  );
  const grid = text("div", "prompt-grid");
  for (const [name, iconName, content] of [
    [
      "Write code",
      "code",
      "Write a Python function to search a tree and explain its complexity.",
    ],
    [
      "Explore a topic",
      "sparkles",
      "Explain authentication versus authorization with examples.",
    ],
    [
      "Solve a problem",
      "activity",
      "Solve a logic problem step by step and check your conclusion.",
    ],
  ]) {
    const card = button("", "prompt-card", () => {
      $("messageInput").value = content;
      resizeInput();
      $("messageInput").focus();
    });
    card.append(
      icon(iconName),
      text("span", "", name),
    );
    grid.append(card);
  }
  welcome.append(grid);
  target.append(welcome);
}
function scrollToBottom() {
  const scroll = $("messageScroll");
  scroll.scrollTop = scroll.scrollHeight;
  state.autoScroll = true;
  $("scrollBottom").classList.add("hidden");
}
function safeUrl(url) {
  // Search results are untrusted: link only to web pages.
  return /^https?:\/\//i.test(url || "") ? url : null;
}
function sourcesBlock(sources) {
  const block = text("div", "message-sources");
  block.append(text("span", "sources-label", "Sources"));
  const list = document.createElement("ol");
  for (const source of sources) {
    const url = safeUrl(source.url);
    const item = document.createElement("li");
    item.value = source.index;
    const link = text(url ? "a" : "span", "", source.title || source.url);
    if (url) {
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.title = url;
    }
    let host = "";
    try {
      host = new URL(url).hostname.replace(/^www\./, "");
    } catch {}
    item.append(link, text("small", "", host));
    list.append(item);
  }
  block.append(list);
  return block;
}
function reasoningLabel(ms) {
  return finite(ms) ? `Thought for ${formatMs(ms)}` : "Reasoning";
}
function reasoningBlock(reasoning, live = false, ms = null) {
  const block = text("details", `reasoning-block${live ? " live" : ""}`);
  const summary = text("summary", "");
  summary.append(
    icon("sparkles"),
    text("span", "reasoning-label", live ? "Thinking…" : reasoningLabel(ms)),
    icon("chevron"),
  );
  const body = text("div", "reasoning-text message-content");
  renderContent(body, reasoning);
  block.append(summary, body);
  return block;
}
function renderMessages() {
  const scroll = $("messageScroll");
  const previous = scroll.scrollTop;
  const follow = state.autoScroll;
  const target = $("messages");
  target.replaceChildren();
  const c = activeConversation();
  // Without a conversation both ids are undefined, which must not match.
  const pendingHere = Boolean(c && state.pending?.conversationId === c.id);
  $("chatView").classList.toggle(
    "empty-chat",
    !c?.messages.length && !pendingHere,
  );
  if (!c?.messages.length && !pendingHere) {
    renderWelcome(target);
    $("scrollBottom").classList.add("hidden");
    requestAnimationFrame(scrollToBottom);
    return;
  }
  for (const [index, message] of (c?.messages || []).entries()) {
    const row = text("article", `message ${message.role}`);
    row.id = `message-${message.id}`;
    const avatar = text("div", "avatar");
    avatar.append(
      message.role === "assistant" ? icon("brain") : text("span", "", "U"),
    );
    const body = text("div", "message-body");
    const author = text("div", "message-author");
    author.append(
      text("strong", "", message.role === "user" ? "You" : "GhostSwitch"),
      text("time", "", formatDate(message.createdAt)),
    );
    body.append(author);
    if (message.role === "assistant" && message.reasoning)
      body.append(reasoningBlock(message.reasoning, false, message.reasoningMs));
    const content = text(
      "div",
      `message-content${message.response?.status && !["success", "unsupported"].includes(message.response.status) ? " message-error" : ""}`,
    );
    if (message.attachments?.length) body.append(messageAttachments(message));
    renderContent(content, message.content);
    if (message.content || !message.attachments?.length) body.append(content);
    const sources = message.response?.ghostswitch?.web_search?.sources;
    if (message.role === "assistant" && sources?.length) body.append(sourcesBlock(sources));
    const offer = message.response?.status === "unsupported" && expertOffer(c, message);
    if (offer) body.append(offer);
    const actions = text("div", "message-actions");
    const copyButton = button(
      "",
      "quiet-icon",
      () => copy(message.content),
      "copy",
    );
    copyButton.title = "Copy message";
    copyButton.setAttribute("aria-label", copyButton.title);
    actions.append(copyButton);
    if (message.role === "assistant" && message.response) {
      const result = message.response;
      if (result.status !== "success") actions.append(statusBadge(result.status));
      actions.append(
        button(
          "Details",
          "detail-button",
          () => {
            state.inspectorId = message.id;
            state.settings.inspector = true;
            state.inspectorMobileOpen = true;
            renderHeader();
            renderInspector();
            persist();
          },
          "activity",
        ),
      );
      if (index === c.messages.length - 1)
        actions.append(
          button(
            "Retry",
            "detail-button",
            () => regenerate(c, message),
            "refresh",
          ),
        );
    }
    if (message.role === "user") {
      const edit = button(
        "",
        "quiet-icon",
        () => editMessage(c, index),
        "edit",
      );
      edit.title = "Edit and continue in a new conversation";
      edit.setAttribute("aria-label", edit.title);
      actions.append(edit);
    }
    body.append(actions);
    row.append(avatar, body);
    target.append(row);
  }
  if (pendingHere) {
    const row = text("article", "message assistant streaming");
    const body = text("div", "message-body");
    body.append(text("div", "message-author", "GhostSwitch"));
    if (state.pending.reasoning) {
      const live = !state.pending.text;
      const block = reasoningBlock(state.pending.reasoning, live);
      block.classList.add("streaming-reasoning");
      block.open = live;
      body.append(block);
    }
    const content = text("div", "streaming-text message-content");
    renderContent(content, state.pending.text);
    body.append(content);
    const status = text(
      "div",
      "streaming-status",
      `${state.pending.status} · attempt ${state.pending.attempt}`,
    );
    body.append(status);
    const avatar = text("div", "avatar");
    avatar.append(icon("brain"));
    row.append(avatar, body);
    target.append(row);
  }
  requestAnimationFrame(() => {
    if (follow) scrollToBottom();
    else scroll.scrollTop = previous;
  });
}
function editMessage(c, index) {
  if (state.pending) return toast("Wait for the current request to finish");
  const source = c.messages[index];
  const fork = createConversation(c.selection);
  fork.title = `${c.title} · edited`.slice(0, 100);
  fork.messages = c.messages
    .slice(0, index)
    .map((m) => ({
      ...m,
      id: uid(),
      sourceMessageId: m.sourceMessageId || m.id,
    }));
  fork.draft = source.content;
  $("messageInput").value = source.content;
  state.attachments = [...(source.attachments || [])];
  renderComposerAttachments();
  resizeInput();
  persist();
  render();
  $("messageInput").focus();
  toast("Created a new conversation with the previous context");
}
async function regenerate(c, message) {
  if (state.pending) return toast("A request is already running");
  const index = c.messages.indexOf(message);
  const preceding = c.messages[index - 1];
  if (preceding?.role !== "user")
    return toast("Could not find the original message");
  const newConversation = createConversation(c.selection);
  newConversation.title = `${c.title} · retry`.slice(0, 100);
  newConversation.messages = c.messages
    .slice(0, index - 1)
    .map((m) => ({
      ...m,
      id: uid(),
      sourceMessageId: m.sourceMessageId || m.id,
    }));
  $("messageInput").value = preceding.content;
  state.attachments = [...(preceding.attachments || [])];
  await sendMessage({ preventDefault() {} });
}
function metaRow(label, value) {
  const node = text("div", "meta-row");
  node.append(
    text("span", "", label),
    text("strong", "", String(value ?? "—")),
  );
  return node;
}
function statCard(label, value, subtitle, iconName) {
  const card = text("div", "stat-card");
  const top = text("div", "stat-label");
  top.append(text("span", "", label));
  if (iconName) top.append(icon(iconName));
  card.append(top, text("strong", "stat-value", value));
  if (subtitle) card.append(text("small", "", subtitle));
  return card;
}
function thinkingLabel(thinking, status, reasoningMs) {
  if (!thinking) return "—";
  const why =
    {
      request: "set for this request",
      agent_step: "a step after a tool result",
      expert_policy: "set by the expert",
      auto: `decided automatically${thinking.kind ? ` (${thinking.kind})` : ""}`,
    }[thinking.source] || "off by default";
  if (!thinking.enabled) return `No · ${why}`;
  if (status === "success" && !thinking.used)
    return `No · ${why}, but the reasoning hit its token limit`;
  return `Yes · ${why}${finite(reasoningMs) ? ` · thought for ${formatMs(reasoningMs)}` : ""}`;
}
function renderResponseDetails(target, record) {
  target.replaceChildren();
  const { response, message } = record;
  const info = response.ghostswitch || {};
  const heading = text("div", "response-heading");
  heading.append(
    statusBadge(response.status),
    text("span", "", formatDate(record.date)),
  );
  target.append(heading);
  if (record.external) target.append(metaRow("Source", record.source || "API"));
  target.append(metaRow("Thinking", thinkingLabel(info.thinking, response.status, message?.reasoningMs)));
  if (info.current_stage) target.append(metaRow("Current stage", info.current_stage));
  if (info.progress) {
    const progress = info.progress;
    target.append(metaRow("Generated", `${progress.generated_tokens} / ${progress.max_tokens} tokens`));
  }
  const requestId = text(
    "button",
    "request-id",
    response.id || "No ID",
  );
  requestId.type = "button";
  requestId.title = "Copy ID";
  requestId.onclick = () => response.id && copy(response.id);
  target.append(requestId);
  const stats = text("div", "stat-grid compact");
  // Fixed instructions reused from the prompt cache cost no computation.
  const cached = response.usage?.prompt_tokens_details?.cached_tokens;
  for (const [label, value, subtitle] of [
    ["Total time", formatMs(info.latency_ms)],
    ["Speed", formatRate(finalGeneration(response))],
    ["Input tokens", formatNumber(response.usage?.prompt_tokens), cached > 0 ? `${formatNumber(cached)} from cache` : null],
    ["Output tokens", formatNumber(response.usage?.completion_tokens)],
    ["Total tokens", formatNumber(response.usage?.total_tokens)],
    [
      "Attempts",
      finite(info.verification?.attempts)
        ? `${info.verification.attempts} / 5`
        : "—",
    ],
  ])
    stats.append(statCard(label, value, subtitle));
  target.append(stats);
  const route = text("section", "inspector-section");
  route.append(
    text("h3", "", "Routing"),
    metaRow(
      "Mode",
      info.mode ? (info.mode === "direct" ? "Manual" : "Automatic") : "—",
    ),
    metaRow(
      "Router",
      info.router_method || (info.mode === "direct" ? "Bypassed" : "—"),
    ),
    metaRow("Decision", info.route || "—"),
    metaRow(
      "Expert",
      info.expert_id
        ? expertLabel(info.expert_id)
        : info.route === "base"
          ? "Base model"
          : "—",
    ),
    metaRow("Adapter", info.adapter || "—"),
    metaRow("Backend", info.backend || "—"),
  );
  if (info.router_path?.length) {
    const path = text("div", "route-path");
    for (const node of info.router_path) path.append(text("span", "", node));
    route.append(path);
  }
  for (const step of info.router_steps || []) {
    const row = text("div", "route-step");
    row.append(
      text(
        "strong",
        "",
        `${step.node || "root"} → ${step.choice === "__v6_general_or_unavailable__" ? "general route" : step.choice || "—"}`,
      ),
    );
    if (step.expert_fit)
      row.append(
        text(
          "small",
          "",
          `Specialization fit: ${step.expert_fit.match ? "match" : "no match"}`,
        ),
      );
    route.append(row);
  }
  if (response.routing?.missing_capability)
    route.append(
      metaRow("Missing capability", response.routing.missing_capability),
    );
  target.append(route);
  const stages = text("section", "inspector-section");
  stages.append(text("h3", "", "Stages and attempts"));
  const timeline = text("div", "timeline");
  const names = {
    routing: "Finding a specialist",
    selection: "Manual selection",
    thinking: "Deciding whether to think",
    web_search: "Searching the web",
    vision: "Preparing images",
    adapter_check: "LoRA check",
    generation: "Generating response",
    verification: "Verifying response",
    error: "Error",
  };
  for (const stage of info.stages || []) {
    const row = text(
      "div",
      `timeline-item ${["FAIL", "RETRY_WITH_FEEDBACK", "error", "thinking_overflow"].includes(stage.status) || stage.phase === "error" ? "rejected" : ""}`,
    );
    const top = text("div", "timeline-title");
    top.append(
      text("strong", "", names[stage.phase] || stage.phase),
      text(
        "span",
        "",
        finite(stage.latency_ms) ? formatMs(stage.latency_ms) : "",
      ),
    );
    row.append(top);
    row.append(
      text(
        "small",
        "",
        [
          stage.attempt ? `Attempt ${stage.attempt}` : null,
          stage.status,
          stage.images ? `${stage.images} image${stage.images === 1 ? "" : "s"}` : null,
          stage.thinking ? "with reasoning" : null,
          stage.usage?.total_tokens
            ? `${formatNumber(stage.usage.total_tokens)} tok.`
            : null,
          stage.usage?.prompt_tokens_details?.cached_tokens > 0
            ? `${formatNumber(stage.usage.prompt_tokens_details.cached_tokens)} from cache`
            : null,
          stage.phase === "generation" ? formatRate(stage) : null,
        ]
          .filter(Boolean)
          .join(" · "),
      ),
    );
    if (stage.feedback) row.append(text("p", "stage-feedback", stage.feedback));
    if (stage.phase === "thinking")
      row.append(
        text(
          "p",
          "stage-check",
          `Request kind: ${stage.kind || "unclear"} → ${stage.status === "think" ? "reason first" : "answer directly"}`,
        ),
      );
    if (stage.phase === "web_search") {
      const found = stage.sources?.length || 0;
      const read = stage.sources?.filter((source) => source.read).length || 0;
      row.append(
        text(
          "p",
          "stage-check",
          stage.status === "not_needed"
            ? "No search needed for this request"
            : `Query: ${stage.query} · ${stage.provider} · ${found} results, ${read} pages read`,
        ),
      );
      if (stage.error) row.append(text("p", "stage-feedback", stage.error));
    }
    if (stage.check) {
      const verdict = {
        confirmed: "confirms the answer",
        contradicted: "contradicts the answer",
      }[stage.check.override] || (stage.check.agrees ? "matches the answer" : "differs from the answer");
      const value = new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 }).format(stage.check.value);
      row.append(text("p", "stage-check", `Check: ${stage.check.expression} = ${value} · ${verdict}`));
    }
    if (stage.candidate_preview) {
      const rejected = text("details", "stage-candidate");
      rejected.append(
        text("summary", "", "Rejected answer"),
        text("pre", "", stage.candidate_preview),
      );
      row.append(rejected);
    }
    timeline.append(row);
  }
  if (!timeline.childNodes.length)
    timeline.append(
      text("p", "muted", "Detailed stages are not available for this response."),
    );
  stages.append(timeline);
  target.append(stages);
  if (response.error) {
    const error = text("div", "error-callout");
    error.append(
      icon("warning"),
      text("p", "", response.error.message || response.error.type),
    );
    target.append(error);
  }
  const raw = text("details", "raw-details");
  raw.append(text("summary", "", "Raw API response"));
  const pre = text("pre", "json-block", JSON.stringify(response, null, 2));
  raw.append(
    button(
      "Copy JSON",
      "text-button",
      () => copy(pre.textContent),
      "copy",
    ),
    pre,
  );
  target.append(raw);
  const actions = text("div", "detail-actions");
  actions.append(
    button(
      "Export JSON",
      "secondary-button",
      () =>
        download(
          `${response.id || message.id}.json`,
          JSON.stringify(
            { request: message.request || null, response },
            null,
            2,
          ),
        ),
      "download",
    ),
  );
  if (message.request)
    actions.append(
      button(
        "Request",
        "secondary-button",
        () => copy(JSON.stringify(message.request, null, 2)),
        "copy",
      ),
    );
  target.append(actions);
}
function renderInspector() {
  const target = $("inspectorContent");
  const pending =
    state.pending?.conversationId === state.activeId ? state.pending : null;
  if (pending) {
    target.replaceChildren();
    const block = text("div", "live-inspector");
    block.append(
      text("span", "badge success", "LIVE"),
      text("h2", "", pending.status),
      text("p", "muted", "The answer is checked before it is final."),
      metaRow("Attempt", `${pending.attempt} / 5`),
      metaRow("Elapsed", formatMs(Date.now() - pending.startedAt)),
      metaRow(
        "Mode",
        pending.request.mode === "auto" ? "Automatic" : "Manual",
      ),
    );
    if (pending.request.expert_id)
      block.append(metaRow("Expert", expertLabel(pending.request.expert_id)));
    block.append(
      text(
        "p",
        "live-note",
        "Exact tokens, speed, and routing appear when the request finishes.",
      ),
    );
    target.append(block);
    return;
  }
  const c = activeConversation();
  const replies = (c?.messages || []).filter(
    (m) => m.role === "assistant" && m.response,
  );
  const selectedMessage =
    replies.find((m) => m.id === state.inspectorId) || replies.at(-1);
  const selected = selectedMessage
    ? {
        message: selectedMessage,
        response: selectedMessage.response,
        date: selectedMessage.createdAt || c.updatedAt,
        conversation: c,
      }
    : null;
  if (selected) renderResponseDetails(target, selected);
  else {
    target.replaceChildren(
      emptyState(
        "Response details",
        "See routing, the selected expert, speed, and verification results here.",
        null,
        "activity",
      ),
    );
  }
}
function toggleInspector(show) {
  if (matchMedia("(max-width: 1000px)").matches) {
    state.inspectorMobileOpen = show ?? !state.inspectorMobileOpen;
    if (state.inspectorMobileOpen) state.settings.inspector = true;
  } else state.settings.inspector = show ?? !state.settings.inspector;
  renderHeader();
  persist();
}
function resizeInput() {
  const input = $("messageInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}
async function refreshRequestLog() {
  if (refreshRequestLog.busy || document.hidden) return;
  refreshRequestLog.busy = true;
  const base = state.settings.baseUrl;
  try {
    const response = await fetch(`${base}/v1/requests`, { signal: AbortSignal.timeout(5000), cache: "no-store" });
    if (!response.ok) throw new Error("Server request log unavailable");
    const body = await response.json();
    if (base !== state.settings.baseUrl) return;
    state.serverRequests = Array.isArray(body.data) ? body.data : [];
    state.journalError = null;
    if (state.view === "requests" && !document.activeElement?.closest(".table-toolbar")) renderRequests();
    if (state.view === "overview") renderOverview();
    if (state.requestDetailId && !$("detailOverlay").classList.contains("hidden")) {
      const selected = records().find(record => record.response.id === state.requestDetailId);
      if (selected?.external) renderResponseDetails($("requestDetail"), selected);
    }
  } catch (error) {
    if (base === state.settings.baseUrl) state.journalError = error.message;
  } finally {
    refreshRequestLog.busy = false;
  }
}
async function refreshServer() {
  if (refreshServer.busy) return;
  refreshServer.busy = true;
  const base = state.settings.baseUrl;
  try {
    const [health, experts] = await Promise.allSettled([
      fetch(`${base}/health`, { signal: AbortSignal.timeout(5000) }).then(
        async (r) => {
          if (!r.ok) throw new Error("Health endpoint unavailable");
          return r.json();
        },
      ),
      fetch(`${base}/v1/experts`, { signal: AbortSignal.timeout(5000) }).then(
        async (r) => {
          if (!r.ok) throw new Error("Catalog unavailable");
          return r.json();
        },
      ),
    ]);
    if (base !== state.settings.baseUrl) return;
    state.connected =
      health.status === "fulfilled" && health.value.status === "ok";
    if (state.connected) {
      state.health = health.value;
      state.healthAt = Date.now();
    }
    if (experts.status === "fulfilled" && Array.isArray(experts.value.data)) {
      state.experts = experts.value.data;
      state.expertsAt = Date.now();
      persist();
    }
  } catch {
    state.connected = false;
  } finally {
    refreshServer.busy = false;
    renderConnection();
    renderSidebar();
    if (state.view === "overview") renderOverview();
    if (state.view === "experts") {
      const identity = state.experts.map(e => e.id).join("|");
      if (state.catalogIdentity !== identity) renderCatalog();
      else updateVisibleExpertDownloads();
    }
    if (!$("modelOverlay").classList.contains("hidden")) renderExperts();
    refreshExpertOffers();
    if (state.connected && setupRequired() && !state.onboarding && !state.switching)
      openOnboarding(state.settings.onboarded ? ONBOARDING_STEPS.length - 1 : 0);
    else if (state.onboarding && renderOnboarding.key !== onboardingKey()) {
      // The server came online or reported its model: update what the setup can offer.
      if (state.connected && !state.runtime && state.onboarding.step === ONBOARDING_STEPS.length - 1)
        refreshRuntime();
      renderOnboarding();
    }
    if (state.view === "settings" && state.connected &&
        ((!state.runtime && !renderSettings.loading) || (!state.searchSettings && !loadSearchSettings.loading))) {
      loadSearchSettings.failed = false;
      renderSettings();
    }
  }
}
async function manageExpert(id, action) {
  try {
    const response = await fetch(`${state.settings.baseUrl}/v1/experts/${encodeURIComponent(id)}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }), signal: AbortSignal.timeout(10000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error?.message || "Could not update the expert");
    const expert = state.experts.find(e => e.id === id);
    if (expert) Object.assign(expert, result);
    persist();
    if (state.view === "experts") renderCatalog();
    if (!$("modelOverlay").classList.contains("hidden")) renderExperts();
    toast(action === "disable" ? "Expert disabled" : result.downloaded ? "Expert ready" : "Expert queued for download");
    await refreshServer();
    return true;
  } catch (error) {
    toast(error.message);
    return false;
  }
}
function expertDownloadPhase(phase) {
  return {downloading: "Downloading", checking: "Checking adapter", converting: "Converting adapter", saving: "Saving adapter"}[phase] || "Preparing adapter";
}
function formatDownloadBytes(value) {
  if (!Number.isFinite(value)) return "—";
  if (value < 1024) return `${Math.round(value)} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
function downloadPercent(progress) {
  if (!Number.isFinite(progress?.percent)) return "";
  return progress.percent < 1 && progress.downloaded_bytes > 0 ? "<1%" : `${progress.percent.toFixed(1)}%`;
}
function renderExpertDownload(target, e) {
  target.replaceChildren();
  target.hidden = !["queued", "downloading"].includes(e.status);
  if (target.hidden) return;
  const p = e.progress;
  target.append(text("small", "", e.status === "queued" ? "Waiting in queue" : expertDownloadPhase(p?.phase)));
  if (e.status === "queued") return;
  const bar = document.createElement("progress");
  bar.max = 100;
  bar.setAttribute("aria-label", "Current file download progress");
  if (p?.phase === "downloading" && Number.isFinite(p.percent) &&
      (p.percent > 0 || !p.received_bytes)) bar.value = p.percent;
  target.append(bar);
  if (p?.phase === "downloading") {
    const amount = `${formatDownloadBytes(p.downloaded_bytes)}${Number.isFinite(p.total_bytes) ? ` / ${formatDownloadBytes(p.total_bytes)}` : ""}`;
    target.append(text("small", "download-amount", `${amount}${downloadPercent(p) ? ` · ${downloadPercent(p)}` : ""}`));
    if (p.received_bytes > 0) {
      target.append(text("small", "download-network", `Received ${formatDownloadBytes(p.received_bytes)}${p.bytes_per_second > 0 ? ` · ${formatDownloadBytes(p.bytes_per_second)}/s` : ""}`));
    } else if (!p.downloaded_bytes) {
      target.append(text("small", "download-network", "Connecting to Hugging Face…"));
    }
    if (p.file) target.append(text("small", "download-file", p.file));
  }
}
function renderExpertActions(actions, e) {
  actions.replaceChildren();
      const busy = ["queued", "downloading"].includes(e.status);
      if (e.enabled && e.downloaded) {
        const chat = button("Chat", "secondary-button", () => createConversation(e.id), "chat");
        chat.disabled = !state.connected;
        actions.append(chat);
      } else {
        const enable = button(busy ? "Downloading…" : e.status === "error" ? "Try again" : "Turn on",
          "secondary-button", () => manageExpert(e.id, "enable"), e.downloaded ? "check" : "download");
        enable.title = e.downloaded ? "" : "Downloads the adapter once, then turns it on";
        enable.disabled = !state.connected || busy;
        actions.append(enable);
      }
      if (e.enabled) {
        const disable = button("Turn off", "text-button", () => manageExpert(e.id, "disable"));
        disable.disabled = !state.connected;
        actions.append(disable);
      }
      if (e.manual_only) {
        const remove = button("Remove", "text-button", () => removeHubExpert(e), "trash");
        remove.disabled = !state.connected;
        actions.append(remove);
      }
}
function updateVisibleExpertDownloads() {
  for (const card of $("expertsView").querySelectorAll(".expert-card")) {
    const expert = state.experts.find(e => e.id === card.dataset.expertId);
    if (!expert) continue;
    renderExpertDownload(card.querySelector(".expert-download-progress"), expert);
    updateExpertState(card, expert);
    const actions = card.querySelector(".expert-card-actions");
    const actionState = `${expert.enabled}|${expert.downloaded}|${expert.status}|${state.connected}`;
    if (actions.dataset.state !== actionState) {
      renderExpertActions(actions, expert);
      actions.dataset.state = actionState;
    }
    const error = card.querySelector(".expert-download-error");
    if (error) error.remove();
    if (expert.error) card.append(text("p", "expert-download-error", expert.error));
  }
}
function expertStatus(e) {
  if (!state.connected) return "Offline";
  if (e.status === "queued") return "Waiting to download";
  if (e.status === "downloading") {
    const phase = e.progress?.phase;
    if (phase && phase !== "downloading") return expertDownloadPhase(phase);
    return Number.isFinite(e.progress?.percent) ? `Downloading · ${downloadPercent(e.progress)}` : "Downloading";
  }
  if (e.status === "error") return "Download failed";
  return e.enabled && e.downloaded ? "On" : "Off";
}
async function chooseExpert(id) {
  const expert = state.experts.find(e => e.id === id);
  if (id !== "auto" && (!expert?.enabled || !expert?.downloaded)) {
    await manageExpert(id, "enable");
    return;
  }
  const c = activeConversation() || createConversation();
  c.selection = id;
  persist();
  closeDialog("modelOverlay");
  renderHeader();
  renderExperts();
  if (state.view === "chat") renderSidebar();
}
function renderExperts() {
  const target = $("expertList");
  target.replaceChildren();
  const query = $("expertSearch").value.trim().toLocaleLowerCase();
  const selected = activeConversation()?.selection || "auto";
  const options = [
    {
      id: "auto",
      name: "Automatic selection",
      description: "The router picks a GhostSwitch expert or the base model. Adapters added from Hugging Face are used only when you choose them here.",
      capabilities: [],
      adapter: "hierarchical",
    },
    ...state.experts,
  ];
  for (const e of options.filter((e) =>
    `${e.id} ${e.name} ${e.description} ${(e.capabilities || []).join(" ")}`
      .toLocaleLowerCase()
      .includes(query),
  )) {
    const option = button(
      "",
      `expert-option${selected === e.id ? " selected" : ""}`,
      () => chooseExpert(e.id),
    );
    const mark = text("span", "expert-icon");
    mark.append(icon(e.id === "auto" ? "sparkles" : "layers"));
    const info = text("span", "expert-info");
    info.append(
      text("strong", "", e.name),
      text("small", "", e.description),
      text("code", "expert-id", e.id === "auto" ? e.id : `${e.manual_only ? "Hugging Face" : e.verified ? "Official · verified" : "Official"} · ${expertStatus(e)}`),
    );
    option.append(mark, info);
    if (selected === e.id) option.append(icon("check"));
    target.append(option);
  }
  if (!target.childNodes.length)
    target.append(
      emptyState(
        "Nothing found",
        "Try another specialty.",
        null,
        "search",
      ),
    );
  $("expertCount").textContent =
    `${state.experts.length} LoRA · ${state.connected ? "server catalog" : state.expertsAt ? "cached catalog" : "server offline"}`;
}
// Attachments travel as OpenAI content parts: image_url for images, file for
// documents. Messages keep metadata and a small preview; data stays in IndexedDB.
const ATTACHMENT_LIMITS = {
  count: 8,
  imageBytes: 20 * 1024 * 1024,
  fileBytes: 10 * 1024 * 1024,
  // The server scales images to about one megapixel; this keeps uploads small.
  imageSide: 2048,
};
const TEXT_FILE_EXTENSIONS = new Set(
  ("txt text md markdown csv tsv json jsonl ndjson xml yaml yml toml ini cfg conf log env " +
    "py pyi js mjs cjs ts tsx jsx java c h cpp hpp cc cs go rs rb php swift kt kts scala " +
    "sh bash zsh fish ps1 bat sql html htm css scss less vue svelte r lua dart ex exs erl hs " +
    "clj tex rst org ipynb gradle properties diff patch srt vtt svg")
    .split(" "),
);
const TEXT_FILE_NAMES = new Set(["dockerfile", "makefile", "readme", "license"]);
const attachmentData = new Map();
let attachmentDatabase = null;
function attachmentStore(mode, action) {
  attachmentDatabase ||= new Promise((resolve, reject) => {
    const request = indexedDB.open("minibrain-attachments", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("files");
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  return attachmentDatabase.then(
    (db) =>
      new Promise((resolve, reject) => {
        const transaction = db.transaction("files", mode);
        const request = action(transaction.objectStore("files"));
        transaction.oncomplete = () => resolve(request.result);
        transaction.onerror = transaction.onabort = () => reject(transaction.error);
      }),
  );
}
async function saveAttachmentData(id, dataUrl) {
  attachmentData.set(id, dataUrl);
  try {
    await attachmentStore("readwrite", (store) => store.put(dataUrl, id));
  } catch {
    toast("Browser storage is unavailable: attachments are kept until the page reloads");
  }
}
async function loadAttachmentData(id) {
  if (attachmentData.has(id)) return attachmentData.get(id);
  try {
    const value = await attachmentStore("readonly", (store) => store.get(id));
    return typeof value === "string" ? value : null;
  } catch {
    return null;
  }
}
// Remove stored data that no message or composer attachment refers to.
async function pruneAttachments() {
  const used = new Set(state.attachments.map((a) => a.id));
  for (const c of state.conversations)
    for (const m of c.messages) for (const a of m.attachments || []) used.add(a.id);
  for (const id of attachmentData.keys()) if (!used.has(id)) attachmentData.delete(id);
  try {
    const keys = await attachmentStore("readonly", (store) => store.getAllKeys());
    const stale = keys.filter((key) => !used.has(key));
    if (stale.length)
      await attachmentStore("readwrite", (store) => {
        stale.forEach((key) => store.delete(key));
        return store.count();
      });
  } catch {
    // Storage is optional; nothing to clean up without it.
  }
}
function validAttachments(items) {
  if (!Array.isArray(items)) return undefined;
  const valid = items
    .filter(
      (a) =>
        a && typeof a.id === "string" && ["image", "file"].includes(a.kind) && typeof a.name === "string",
    )
    .map((a) => ({
      id: a.id,
      kind: a.kind,
      name: a.name.slice(0, 120),
      mime: typeof a.mime === "string" ? a.mime : "",
      size: finite(a.size) ? Number(a.size) : 0,
      ...(finite(a.width) && finite(a.height) ? { width: Number(a.width), height: Number(a.height) } : {}),
      ...(typeof a.thumb === "string" && a.thumb.startsWith("data:image/") ? { thumb: a.thumb } : {}),
    }));
  return valid.length ? valid : undefined;
}
function contentText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) =>
      part?.type === "text"
        ? part.text
        : part?.type === "image_url"
          ? `[Image${part.name ? `: ${part.name}` : ""}]`
          : part?.type === "file"
            ? `[File: ${part.file?.filename || "file"}]`
            : "",
    )
    .filter(Boolean)
    .join(" ");
}
function messageText(m) {
  const names = (m.attachments || []).map((a) => `[${a.kind === "image" ? "Image" : "File"}: ${a.name}]`);
  return [...names, m.content].filter(Boolean).join(" ");
}
function attachmentKind(file) {
  if (file.type.startsWith("image/") && file.type !== "image/svg+xml") return "image";
  const name = file.name.toLowerCase();
  const extension = name.includes(".") ? name.split(".").pop() : "";
  if (
    file.type.startsWith("text/") ||
    ["application/pdf", "application/json", "application/xml", "application/javascript"].includes(file.type) ||
    file.type.includes("wordprocessingml") ||
    ["pdf", "docx"].includes(extension) ||
    TEXT_FILE_EXTENSIONS.has(extension) ||
    TEXT_FILE_NAMES.has(name)
  )
    return "file";
  return null;
}
function readDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}
async function prepareImage(file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    throw new Error(`${file.name}: this browser cannot read the image. Try PNG or JPEG.`);
  }
  const { width, height } = bitmap;
  const scale = Math.min(1, ATTACHMENT_LIMITS.imageSide / Math.max(width, height));
  const draw = (factor, type, quality) => {
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * factor));
    canvas.height = Math.max(1, Math.round(height * factor));
    const context = canvas.getContext("2d");
    if (type === "image/jpeg") {
      context.fillStyle = "#fff";
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL(type, quality);
  };
  try {
    // Small PNG and JPEG files go as they are; others are scaled and re-encoded.
    const keep = scale === 1 && ["image/png", "image/jpeg"].includes(file.type) && file.size <= 4 * 1024 * 1024;
    let dataUrl = keep
      ? await readDataUrl(file)
      : draw(scale, file.type === "image/png" ? "image/png" : "image/jpeg", 0.9);
    if (dataUrl.length > 8 * 1024 * 1024) dataUrl = draw(scale, "image/jpeg", 0.88);
    return {
      dataUrl,
      thumb: draw(Math.min(1, 240 / Math.max(width, height)), "image/jpeg", 0.78),
      width: Math.round(width * scale),
      height: Math.round(height * scale),
      mime: dataUrl.slice(5, dataUrl.indexOf(";")),
    };
  } finally {
    bitmap.close?.();
  }
}
async function addAttachments(files) {
  for (const file of [...files]) {
    if (state.attachments.length >= ATTACHMENT_LIMITS.count) {
      toast(`Up to ${ATTACHMENT_LIMITS.count} attachments per message`);
      break;
    }
    const kind = attachmentKind(file);
    if (!kind) {
      toast(`${file.name}: unsupported type. Attach images, PDF, DOCX or text and code files.`);
      continue;
    }
    if (kind === "image" && state.health?.vision === false) {
      toast("The current base model does not accept images");
      continue;
    }
    const limit = kind === "image" ? ATTACHMENT_LIMITS.imageBytes : ATTACHMENT_LIMITS.fileBytes;
    if (file.size > limit) {
      toast(`${file.name} is larger than ${limit / (1024 * 1024)} MB`);
      continue;
    }
    const item = { id: uid(), kind, name: file.name.slice(0, 120) || kind, mime: file.type, size: file.size, loading: true };
    state.attachments.push(item);
    renderComposerAttachments();
    try {
      if (kind === "image") {
        const image = await prepareImage(file);
        Object.assign(item, {
          mime: image.mime,
          width: image.width,
          height: image.height,
          thumb: image.thumb,
          size: Math.round(((image.dataUrl.length - image.dataUrl.indexOf(",") - 1) * 3) / 4),
        });
        await saveAttachmentData(item.id, image.dataUrl);
      } else {
        await saveAttachmentData(item.id, await readDataUrl(file));
      }
      delete item.loading;
    } catch (error) {
      state.attachments = state.attachments.filter((a) => a !== item);
      toast(error.message || `Could not read ${file.name}`);
    }
    renderComposerAttachments();
  }
}
function attachmentTile(a, className) {
  const tile = text(className === "attachment-chip" ? "div" : "button", className);
  if (tile.tagName === "BUTTON") tile.type = "button";
  if (a.thumb) {
    const image = document.createElement("img");
    image.src = a.thumb;
    image.alt = "";
    tile.append(image);
  } else {
    const badge = text("span", "attachment-badge");
    badge.append(icon("file"));
    tile.append(badge);
  }
  const label = text("span", "attachment-text");
  label.append(
    text("span", "attachment-name", a.name),
    text(
      "span",
      "attachment-meta",
      a.loading
        ? "Preparing…"
        : a.kind === "image" && a.width
          ? `${a.width}×${a.height}`
          : formatDownloadBytes(a.size),
    ),
  );
  tile.append(label);
  tile.title = a.name;
  return tile;
}
function renderComposerAttachments() {
  const tray = $("composerAttachments");
  tray.replaceChildren(
    ...state.attachments.map((a) => {
      const chip = attachmentTile(a, `attachment-chip${a.loading ? " loading" : ""}`);
      const remove = button(
        "",
        "quiet-icon attachment-remove",
        () => {
          state.attachments = state.attachments.filter((item) => item !== a);
          renderComposerAttachments();
          pruneAttachments();
        },
        "close",
      );
      remove.title = `Remove ${a.name}`;
      remove.setAttribute("aria-label", remove.title);
      chip.append(remove);
      return chip;
    }),
  );
  tray.classList.toggle("hidden", !state.attachments.length);
}
function messageAttachments(message) {
  const box = text("div", "message-attachments");
  for (const a of message.attachments) {
    const tile =
      a.kind === "image" && a.thumb
        ? text("button", "message-image")
        : attachmentTile(a, "message-file");
    if (tile.className === "message-image") {
      tile.type = "button";
      const image = document.createElement("img");
      image.src = a.thumb;
      image.alt = a.name;
      tile.append(image);
    }
    tile.title = `Open ${a.name}`;
    tile.onclick = () => openAttachment(a);
    box.append(tile);
  }
  return box;
}
async function openAttachment(a) {
  const data = await loadAttachmentData(a.id);
  if (!data) return toast("This attachment is no longer stored in this browser");
  const blob = await (await fetch(data)).blob();
  const url = URL.createObjectURL(blob);
  if (a.kind === "image" || blob.type === "application/pdf" || blob.type.startsWith("text/")) {
    window.open(url, "_blank", "noopener");
  } else {
    const link = document.createElement("a");
    link.href = url;
    link.download = a.name;
    link.click();
  }
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
// A user message as API content: attachments first, then the typed text.
async function requestContent(message) {
  if (!message.attachments?.length) return message.content;
  const parts = [];
  for (const a of message.attachments) {
    const data = await loadAttachmentData(a.id);
    if (!data)
      parts.push({ type: "text", text: `[${a.name} was attached earlier but is no longer available]` });
    else if (a.kind === "image") parts.push({ type: "image_url", image_url: { url: data }, name: a.name });
    else parts.push({ type: "file", file: { filename: a.name, file_data: data } });
  }
  if (message.content) parts.push({ type: "text", text: message.content });
  return parts;
}
// Saved requests and the inspector show attachment types and sizes, not their data.
function withoutAttachmentData(value) {
  return JSON.parse(
    JSON.stringify(value, (key, item) =>
      typeof item === "string" && item.startsWith("data:") && item.length > 256
        ? `${item.slice(0, item.indexOf(","))},<${item.length - item.indexOf(",") - 1} characters omitted>`
        : item,
    ),
  );
}
async function sendMessage(event, route = null) {
  event.preventDefault();
  if (state.pending || state.playgroundBusy) return;
  if (state.switching) return toast("Wait until the model switch finishes");
  if (setupRequired()) return openOnboarding(ONBOARDING_STEPS.length - 1);
  const input = $("messageInput");
  const content = input.value.trim();
  if (state.attachments.some((a) => a.loading))
    return toast("Wait until the attachments are ready");
  const attachments = validAttachments(state.attachments);
  if (!content && !attachments) return;
  const maxTokens = Number($("maxTokens").value);
  const temperature = Number($("temperature").value);
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 10240)
    return toast("Token limit: 1–10,240");
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2)
    return toast("Temperature: 0–2");
  const c = activeConversation() || createConversation();
  const selection = route?.mode === "base" ? "auto" : route?.expertId || c.selection || "auto";
  if (selection !== "auto" && !state.experts.some((e) => e.id === selection))
    return toast(
      "The selected LoRA is missing from the catalog. Refresh the list or choose Auto.",
    );
  if (selection !== "auto") {
    const expert = state.experts.find(e => e.id === selection);
    if (!expert?.enabled || !expert?.downloaded)
      return toast("Enable the expert in the library and wait for the download.");
  }
  state.settings.maxTokens = maxTokens;
  state.settings.temperature = temperature;
  if (!c.messages.length)
    c.title = (content || attachments.map((a) => a.name).join(", ")).replace(/\s+/g, " ").slice(0, 48);
  c.messages.push({ id: uid(), role: "user", content, ...(attachments ? { attachments } : {}), createdAt: Date.now() });
  c.updatedAt = Date.now();
  c.draft = "";
  state.attachments = [];
  renderComposerAttachments();
  // A question whose reply failed or was stopped stays out of the history:
  // otherwise the model answers it again instead of the new message.
  const history = c.messages.filter((m, i, all) =>
    m.role === "user"
      ? i === all.length - 1 || all[i + 1]?.response?.status === "success"
      : m.response?.status === "success",
  );
  // Attachment data is read from browser storage after the request shows as pending.
  const request = {
    model: "ghostswitch-v6",
    messages: [],
    mode: route?.mode === "base" ? "base" : selection === "auto" ? "auto" : "direct",
    max_tokens: maxTokens,
    temperature,
    stream: true,
  };
  if (request.mode === "direct") request.expert_id = selection;
  if (state.settings.webSearch !== "off") request.web_search = state.settings.webSearch;
  // Auto leaves the choice to GhostSwitch; on and off decide it for this request.
  if (state.settings.thinking !== "auto")
    request.reasoning_effort = state.settings.thinking === "on" ? "high" : "none";
  const pending = {
    conversationId: c.id,
    text: "",
    reasoning: "",
    status: "Preparing request",
    attempt: 1,
    startedAt: Date.now(),
    controller: new AbortController(),
    request,
    canceled: false,
  };
  state.pending = pending;
  state.autoScroll = true;
  input.value = "";
  resizeInput();
  persist();
  render();
  try {
    const messages = await Promise.all(
      history.map(async (m) => ({
        role: m.role,
        content: m.role === "user" ? await requestContent(m) : m.content,
      })),
    );
    if (state.settings.systemPrompt.trim())
      messages.unshift({
        role: "system",
        content: state.settings.systemPrompt.trim(),
      });
    const body = JSON.stringify({ ...request, messages });
    request.messages = withoutAttachmentData(messages);
    const http = await fetch(`${state.settings.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-GhostSwitch-Client": "web-chat" },
      body,
      signal: pending.controller.signal,
    });
    if (!http.ok) {
      let error;
      try {
        error = await http.json();
      } catch {
        error = {};
      }
      const response = { ...error, status: error.status || "failed" };
      c.messages.push({
        id: uid(),
        role: "assistant",
        content: error.error?.message || `HTTP error ${http.status}`,
        response,
        request,
        createdAt: Date.now(),
      });
      return;
    }
    const streamed = await consumeCompletionStream(http, pending);
    const reasoning =
      streamed.status === "success" && pending.reasoning ? pending.reasoning : undefined;
    const response = {
      id: streamed.id,
      object: "chat.completion",
      created: streamed.created,
      model: streamed.model,
      status: streamed.status,
      usage: streamed.usage,
      ghostswitch: streamed.ghostswitch,
      choices:
        streamed.status === "success"
          ? [
              {
                index: 0,
                message: {
                  role: "assistant",
                  content: pending.text,
                  ...(reasoning ? { reasoning_content: reasoning } : {}),
                },
                finish_reason: streamed.finish_reason || "stop",
              },
            ]
          : [],
    };
    if (streamed.error) response.error = streamed.error;
    if (streamed.routing) response.routing = streamed.routing;
    const answer =
      response.status === "success"
        ? pending.text
        : response.status === "unsupported"
          ? unavailableMessage(response.routing?.missing_capability)
          : response.error?.message || "The response did not pass verification.";
    c.messages.push({
      id: uid(),
      role: "assistant",
      content: answer || "Empty response",
      ...(reasoning ? { reasoning } : {}),
      ...(reasoning && pending.reasoningMs ? { reasoningMs: pending.reasoningMs } : {}),
      response,
      request,
      createdAt: Date.now(),
    });
    state.connected = true;
  } catch (error) {
    const canceled = pending.canceled;
    const message = canceled
      ? "Generation stopped in the client. The server may finish processing the current request."
      : `Request failed: ${error.message}`;
    c.messages.push({
      id: uid(),
      role: "assistant",
      content:
        canceled && pending.text ? `${pending.text}\n\n> ${message}` : message,
      response: {
        status: canceled ? "canceled" : "failed",
        error: {
          type: canceled ? "client_canceled" : "network_error",
          message,
        },
      },
      request,
      createdAt: Date.now(),
    });
    if (!canceled) state.connected = false;
  } finally {
    c.updatedAt = Date.now();
    state.pending = null;
    persist();
    render();
    if (state.view === "chat" && state.activeId === c.id) input.focus();
  }
}
function stopGeneration() {
  if (state.pending) {
    state.pending.canceled = true;
    state.pending.controller.abort();
  }
}
async function consumeCompletionStream(http, pending) {
  if (!http.body) throw new Error("Streaming response unavailable");
  const reader = http.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completion = null;
  let frameRequested = false;
  const scheduleRender = () => {
    if (frameRequested) return;
    frameRequested = true;
    requestAnimationFrame(() => {
      frameRequested = false;
      if (state.view !== "chat" || state.activeId !== pending.conversationId)
        return;
      const scroll = $("messageScroll");
      const scrollTop = scroll.scrollTop;
      const follow = state.autoScroll;
      const content = $("messages").querySelector(".streaming-text");
      let reasoning = $("messages").querySelector(".streaming-reasoning");
      if (!pending.reasoning) reasoning?.remove();
      else if (content) {
        if (!reasoning) {
          reasoning = reasoningBlock(pending.reasoning, !pending.text);
          reasoning.classList.add("streaming-reasoning");
          reasoning.open = !pending.text;
          content.before(reasoning);
        }
        if (pending.reasoningRendered !== pending.reasoning.length) {
          pending.reasoningRendered = pending.reasoning.length;
          const body = reasoning.querySelector(".reasoning-text");
          body.replaceChildren();
          renderContent(body, pending.reasoning);
        }
        // Collapse once when the answer starts; later toggles stay with the user.
        if (pending.text && !pending.reasoningDone) {
          pending.reasoningDone = true;
          reasoning.open = false;
          reasoning.classList.remove("live");
          reasoning.querySelector(".reasoning-label").textContent = reasoningLabel(pending.reasoningMs);
        }
      }
      if (content) {
        content.replaceChildren();
        renderContent(content, pending.text);
      }
      const status = $("messages").querySelector(".streaming-status");
      if (status)
        status.textContent = `${pending.status} · attempt ${pending.attempt}`;
      if (follow) scrollToBottom();
      else scroll.scrollTop = scrollTop;
      renderInspector();
    });
  };
  const readFrame = (frame) => {
    const data = frame
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (!data || data === "[DONE]") return;
    const chunk = JSON.parse(data);
    const delta = chunk.choices?.[0]?.delta || {};
    if (delta.ghostswitch_attempt) pending.attempt = delta.ghostswitch_attempt;
    if (delta.ghostswitch_status) pending.status = delta.ghostswitch_status;
    if (delta.ghostswitch_reset) {
      pending.text = "";
      pending.reasoning = "";
      pending.reasoningDone = false;
      pending.reasoningRendered = 0;
      pending.reasoningStartedAt = null;
      pending.reasoningMs = null;
    }
    if (typeof delta.reasoning_content === "string") {
      pending.reasoningStartedAt ??= Date.now();
      pending.reasoning = (pending.reasoning || "") + delta.reasoning_content;
    }
    if (typeof delta.content === "string") {
      // The answer starts when the reasoning ends.
      if (delta.content && pending.reasoningStartedAt && !pending.reasoningMs)
        pending.reasoningMs = Date.now() - pending.reasoningStartedAt;
      pending.text += delta.content;
    }
    if (chunk.status)
      completion = {
        ...chunk,
        finish_reason: chunk.choices?.[0]?.finish_reason,
      };
    scheduleRender();
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        readFrame(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
      }
      if (done) break;
    }
    if (buffer.trim()) readFrame(buffer);
  } finally {
    reader.releaseLock();
  }
  if (!completion) throw new Error("The stream ended before a result was received");
  return completion;
}

function renderOverview() {
  const target = $("overviewView");
  target.replaceChildren();
  const range = document.createElement("select");
  range.className = "select-control";
  range.setAttribute("aria-label", "Statistics period");
  for (const [value, label] of [
    ["all", "All time"],
    ["7", "Last 7 days"],
    ["30", "Last 30 days"],
  ]) {
    const option = text("option", "", label);
    option.value = value;
    range.append(option);
  }
  range.value = state.range;
  range.onchange = () => {
    state.range = range.value;
    renderOverview();
  };
  pageHeading(
    target,
    "",
    "Statistics",
    "Request activity and results.",
    [
      range,
      button("New chat", "primary-button", createConversation, "plus"),
    ],
  );
  const all = records();
  const cutoff =
    state.range === "all" ? 0 : Date.now() - Number(state.range) * 86400000;
  const rows = all.filter((r) => r.date >= cutoff);
  const completed = rows.filter((r) => r.response.status !== "canceled");
  const success = completed.filter(
    (r) => r.response.status === "success",
  ).length;
  const tokenRows = rows.filter((r) => finite(r.response.usage?.total_tokens));
  const tokens = tokenRows.reduce(
    (sum, r) => sum + Number(r.response.usage.total_tokens),
    0,
  );
  const measured = rows.filter(
    (r) => rate(finalGeneration(r.response)) !== null,
  );
  const speed = average(measured.map((r) => rate(finalGeneration(r.response))));
  const metrics = text("div", "overview-metrics");
  metrics.append(
    statCard(
      "Requests",
      formatNumber(rows.length),
      `${state.conversations.length} conversations in this browser`,
      "chat",
    ),
    statCard(
      "Accepted responses",
      completed.length
        ? `${Math.round((success / completed.length) * 100)}%`
        : "—",
      completed.length
        ? `${success} of ${completed.length} completed requests`
        : "Available after the first response",
      "check",
    ),
    statCard(
      "Average speed",
      speed == null ? "—" : `${formatNumber(speed)} tok/s`,
      `Final generation · ${measured.length} measurements`,
      "activity",
    ),
    statCard(
      "Tokens used",
      formatNumber(tokens),
      `Including routing and verification · ${tokenRows.length} requests`,
      "layers",
    ),
  );
  target.append(metrics);
  const grid = text("div", "overview-grid");
  const activity = panel(
    "Request activity",
    "Last 7 days · this browser's history",
  );
  activity.append(activityChart(all));
  const routePanel = panel("Route distribution", "Selected period");
  const routeEntries = [
    [
      "LoRA experts",
      rows.filter((r) => r.response.ghostswitch?.route === "expert").length,
      "accent",
    ],
    [
      "Base model",
      rows.filter((r) => r.response.ghostswitch?.route === "base").length,
      "blue",
    ],
    [
      "No route",
      rows.filter(
        (r) => !["expert", "base"].includes(r.response.ghostswitch?.route),
      ).length,
      "gray",
    ],
  ];
  const total = rows.length;
  const ring = text("div", "route-distribution");
  const donut = text("div", "donut");
  const percentExpert = total ? (routeEntries[0][1] / total) * 100 : 0;
  const percentBase = total ? (routeEntries[1][1] / total) * 100 : 0;
  donut.style.background = total
    ? `conic-gradient(var(--accent) 0 ${percentExpert}%, var(--blue) ${percentExpert}% ${percentExpert + percentBase}%, var(--muted) ${percentExpert + percentBase}% 100%)`
    : "var(--line)";
  const center = text("div", "donut-center");
  center.append(
    text("strong", "", String(total)),
    text("small", "", "requests"),
  );
  donut.append(center);
  ring.append(donut);
  const legend = text("div", "route-legend");
  for (const [label, count, color] of routeEntries) {
    const row = text("div", "legend-row");
    row.append(
      text("span", `legend-dot ${color}`),
      text("span", "", label),
      text("strong", "", String(count)),
    );
    legend.append(row);
  }
  ring.append(legend);
  routePanel.append(ring);
  if (!total)
    routePanel.append(
      text(
        "p",
        "panel-footnote",
        "Routes appear after requests finish.",
      ),
    );
  grid.append(activity, routePanel);
  target.append(grid);
  const bottom = text("div", "overview-grid");
  const recent = panel(
    "Recent requests",
    "Generation and verification results",
    button(
      "All requests",
      "text-button",
      () => navigate("requests"),
      "arrowRight",
    ),
  );
  if (all.length) {
    const list = text("div", "recent-list");
    for (const r of all.slice(0, 4)) {
      const item = button("", "recent-request", () => openRequest(r));
      const mark = text(
        "span",
        `request-mark ${r.response.status === "success" ? "good" : "bad"}`,
      );
      mark.append(icon(r.response.status === "success" ? "check" : "warning"));
      const description = text("span", "recent-info");
      description.append(
        text("strong", "", r.prompt.slice(0, 80) || r.conversation.title),
        text(
          "small",
          "",
          `${formatDate(r.date)} · ${r.response.ghostswitch?.expert_id ? expertLabel(r.response.ghostswitch.expert_id) : r.response.ghostswitch?.route === "base" ? "Base model" : "No route"}`,
        ),
      );
      item.append(
        mark,
        description,
        text(
          "span",
          "recent-latency",
          formatMs(r.response.ghostswitch?.latency_ms),
        ),
        icon("arrowRight"),
      );
      list.append(item);
    }
    recent.append(list);
  } else
    recent.append(
      emptyState(
        "Your history starts here",
        "Send your first message to see timing, tokens, and verification results.",
        button("Start chat", "secondary-button", createConversation, "plus"),
        "chat",
      ),
    );
  const runtime = panel(
    "Server status",
    "Latest HTTP API check",
    button("Check", "text-button", refreshServer, "refresh"),
  );
  runtime.append(
    metaRow("Connection", state.connected ? "Available" : "Offline"),
    metaRow("URL", new URL(state.settings.baseUrl).host),
    metaRow("API model", state.health?.model || "—"),
    metaRow("Backend", state.health?.backend || "—"),
    metaRow("LoRAs in catalog", state.expertsAt ? state.experts.length : "—"),
    metaRow("Checked", formatDate(state.healthAt)),
  );
  runtime.append(
    text(
      "p",
      "panel-footnote",
      state.connected
        ? "Health checks API availability. Model weights are validated during generation."
        : "Start GhostSwitch or enter a running API URL in Settings.",
    ),
  );
  bottom.append(recent, runtime);
  target.append(bottom);
  target.append(
    text(
      "div",
      "page-footnote",
      "This overview combines browser history with recent server requests from all API clients.",
    ),
  );
}
function activityChart(rows) {
  const wrap = text("div", "activity-chart");
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const days = Array.from({ length: 7 }, (_, i) => {
    const date = new Date(today);
    date.setDate(date.getDate() - 6 + i);
    const end = new Date(date);
    end.setDate(end.getDate() + 1);
    const matches = rows.filter(
      (r) => r.date >= date.getTime() && r.date < end.getTime(),
    );
    return {
      date,
      success: matches.filter((r) => r.response.status === "success").length,
      other: matches.filter((r) => r.response.status !== "success").length,
    };
  });
  const max = Math.max(1, ...days.map((d) => d.success + d.other));
  const bars = text("div", "chart-bars");
  for (const d of days) {
    const column = text("div", "chart-column");
    const count = d.success + d.other;
    column.append(text("span", "chart-count", String(count)));
    const plot = text("div", "chart-plot");
    const bar = text("div", "chart-bar");
    bar.style.height = `${count ? (count / max) * 100 : 2}%`;
    bar.title = `${formatDate(d.date, false)}: ${d.success} accepted, ${d.other} other`;
    const other = text("span", "chart-other");
    other.style.height = `${count ? (d.other / count) * 100 : 0}%`;
    bar.append(other);
    plot.append(bar);
    column.append(
      plot,
      text(
        "span",
        "chart-day",
        new Intl.DateTimeFormat("ru", { weekday: "short" }).format(d.date),
      ),
      text(
        "small",
        "chart-date",
        `${d.date.getDate()}.${String(d.date.getMonth() + 1).padStart(2, "0")}`,
      ),
    );
    bars.append(column);
  }
  wrap.append(bars);
  const legend = text("div", "chart-legend");
  legend.append(
    text("span", "legend-dot accent"),
    text("span", "", "Accepted"),
    text("span", "legend-dot amber"),
    text("span", "", "Other"),
  );
  wrap.append(legend);
  return wrap;
}
// --- Expert tiers and adapters from Hugging Face ------------------------------------
function tierHelp(e) {
  if (e.manual_only)
    return "Not verified for GhostSwitch. Automatic selection never uses it; choose it yourself.";
  const m = e.evaluation?.measured;
  return m
    ? `${Math.round(m.expert * 100)}% vs ${Math.round(m.base * 100)}% for the base model on ${m.task} (${m.n} questions)`
    : "Not compared with the base model yet";
}
async function hubRequest(body) {
  const response = await fetch(`${state.settings.baseUrl}/v1/experts/hub`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error?.message || `HTTP ${response.status}`);
  return result;
}
async function loadHub(refresh = false) {
  state.hub = { ...(state.hub || {}), loading: true, error: null };
  renderHubList();
  try {
    const query = new URLSearchParams({ q: state.hub.query || "", limit: "100", refresh: refresh ? "1" : "0" });
    const response = await fetch(`${state.settings.baseUrl}/v1/experts/hub?${query}`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error?.message || `HTTP ${response.status}`);
    Object.assign(state.hub, { rows: result.data, total: result.total, matches: result.matches });
  } catch (error) {
    state.hub.error = error.message;
  } finally {
    state.hub.loading = false;
    renderHubList();
  }
}
async function addHubExpert(row) {
  row.adding = true;
  renderHubList();
  try {
    const result = await hubRequest({ action: "add", repo_id: row.repo_id });
    row.installed = result.id;
    toast(`${row.name} added. Turn it on above to use it.`);
    await refreshServer();
    renderCatalog();
  } catch (error) {
    toast(`Could not add ${row.repo_id}: ${error.message}`);
  } finally {
    row.adding = false;
    renderHubList();
  }
}
async function removeHubExpert(e) {
  try {
    await hubRequest({ action: "remove", expert_id: e.id });
    const row = state.hub?.rows?.find((item) => item.installed === e.id);
    if (row) row.installed = null;
    toast(`${e.name} removed`);
    await refreshServer();
    renderCatalog();
  } catch (error) {
    toast(`Could not remove: ${error.message}`);
  }
}
function hubBrowser() {
  const node = text("section", "hub-browser");
  const bar = text("div", "hub-bar");
  const field = text("label", "search-field");
  const input = document.createElement("input");
  input.placeholder = "Search adapters by name or author…";
  input.setAttribute("aria-label", "Search Hugging Face adapters");
  input.value = state.hub?.query || "";
  let timer;
  input.oninput = () => {
    state.hub = { ...(state.hub || {}), query: input.value };
    clearTimeout(timer);
    timer = setTimeout(() => loadHub(), 350);
  };
  field.append(icon("search"), input);
  bar.append(field);
  const list = text("div", "hub-list");
  list.id = "hubList";
  node.append(bar, list);
  queueMicrotask(renderHubList);
  return node;
}
function renderHubList() {
  const list = $("hubList");
  if (!list) return;
  list.replaceChildren();
  const hub = state.hub || {};
  if (hub.loading) return list.append(text("p", "muted", "Searching Hugging Face…"));
  if (hub.error) return list.append(text("p", "expert-download-error", hub.error));
  if (!hub.rows) return list.append(text("p", "muted", "Loading adapters…"));
  list.append(text("p", "muted", hub.query
    ? `${hub.matches} adapters match`
    : `${hub.total} adapters, most downloaded first${hub.total > hub.rows.length ? `; showing ${hub.rows.length}, search to find others` : ""}`));
  for (const row of hub.rows) {
    const item = text("div", "hub-row");
    const info = text("div", "hub-info");
    const link = text("a", "", row.repo_id);
    link.href = row.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    const updated = row.updated ? ` · updated ${formatDate(Date.parse(row.updated))}` : "";
    info.append(link, text("small", "", `${row.downloads.toLocaleString()} downloads · ${row.likes} likes${updated}${row.adapter_dir ? ` · folder ${row.adapter_dir}` : ""}`));
    const action = row.installed
      ? text("span", "badge neutral", "Added")
      : button(row.adding ? "Adding…" : "Add", "secondary-button", () => addHubExpert(row), "plus");
    if (row.adding) action.disabled = true;
    item.append(info, action);
    list.append(item);
  }
}
function expertCard(e) {
  const card = text("article", "expert-card");
  card.dataset.expertId = e.id;
  const top = text("div", "expert-card-top");
  const title = text("h2", "", e.name);
  if (e.verified) {
    const verified = text("span", "expert-verified", "Verified");
    verified.title = tierHelp(e);
    title.append(verified);
  }
  top.append(title, text("span", "", ""));
  card.append(top, text("p", "expert-description", e.description));
  const details = text("details", "expert-details");
  const summary = text("summary", "", "Details");
  summary.append(icon("chevron"));
  details.append(summary);
  const meta = text("div", "expert-meta");
  meta.append(metaRow("ID", e.id));
  if (e.evaluation?.measured) meta.append(metaRow("Test", tierHelp(e)));
  else if (!e.manual_only) meta.append(metaRow("Test", "Not compared with the base model yet"));
  if (e.revision) meta.append(metaRow("Revision", e.revision.slice(0, 12)));
  details.append(meta);
  if (e.source_url) {
    const source = text("a", "text-button", "Open on Hugging Face");
    source.href = e.source_url;
    source.target = "_blank";
    source.rel = "noopener noreferrer";
    details.append(source);
  }
  card.append(details);
  const download = text("div", "expert-download-progress");
  renderExpertDownload(download, e);
  card.append(download);
  if (e.error) card.append(text("p", "expert-download-error", e.error));
  const actions = text("div", "expert-card-actions");
  renderExpertActions(actions, e);
  card.append(actions);
  updateExpertState(card, e);
  return card;
}
function updateExpertState(card, e) {
  const slot = card.querySelector(".expert-card-top > span:last-child");
  const on = e.enabled && e.downloaded;
  slot.className = `expert-state ${on ? "on" : e.status === "error" ? "failed" : ""}`;
  slot.textContent = expertStatus(e);
}
function renderCatalog() {
  state.catalogIdentity = state.experts.map((e) => e.id).join("|");
  const target = $("expertsView");
  target.replaceChildren();
  const tab = state.expertTab || "official";
  pageHeading(target, "", "Experts",
    `Extra skills for ${state.health?.backbone_name || "the base model"}. Turn on the ones you need; each downloads once and stays on this device.`);
  const tabs = text("div", "expert-tabs");
  tabs.setAttribute("role", "tablist");
  for (const [id, label] of [["official", "Official"], ["huggingface", "Hugging Face"]]) {
    const choice = button(label, `expert-tab${tab === id ? " selected" : ""}`, () => {
      state.expertTab = id;
      renderCatalog();
      if (id === "huggingface" && !state.hub?.rows && !state.hub?.loading) loadHub();
    });
    choice.setAttribute("role", "tab");
    choice.setAttribute("aria-selected", String(tab === id));
    tabs.append(choice);
  }
  target.append(tabs);
  const official = state.experts.filter((e) => !e.manual_only);
  const added = state.experts.filter((e) => e.manual_only);
  if (tab === "official") {
    target.append(text("p", "expert-hint",
      "Prepared for GhostSwitch. In automatic mode, GhostSwitch picks the right one for each message among those you turned on."));
    const search = text("label", "search-field catalog-search");
    const input = document.createElement("input");
    input.placeholder = "Search experts…";
    input.setAttribute("aria-label", "Search experts");
    input.value = state.catalogQuery;
    search.append(icon("search"), input);
    const grid = text("div", "expert-grid");
    target.append(search, grid);
    const populate = () => {
      grid.replaceChildren();
      const query = state.catalogQuery.toLocaleLowerCase();
      const matches = official.filter((e) =>
        `${e.id} ${e.name} ${e.description} ${(e.capabilities || []).join(" ")}`.toLocaleLowerCase().includes(query));
      for (const e of matches) grid.append(expertCard(e));
      if (!matches.length)
        grid.append(emptyState(
          official.length ? "No matches" : "No experts yet",
          official.length ? "Try another word." : "Start the GhostSwitch server to load its experts.",
          null, "layers"));
    };
    input.oninput = () => {
      state.catalogQuery = input.value;
      populate();
    };
    populate();
    return;
  }
  target.append(text("p", "expert-hint",
    "Any LoRA adapter from Hugging Face made for this base model. They are not checked by GhostSwitch, so automatic mode never uses them: pick one when you start a chat."));
  if (added.length) {
    target.append(text("h2", "expert-subheading", "Added"));
    const grid = text("div", "expert-grid");
    for (const e of added) grid.append(expertCard(e));
    target.append(grid);
  }
  target.append(text("h2", "expert-subheading", "Find on Hugging Face"), hubBrowser());
  const guide = button("How an adapter becomes an official expert", "text-button", () => openDialog("addExpertOverlay"), "arrowRight");
  target.append(guide);
}

function requestTable(rows, target) {
  if (!rows.length) {
    target.append(
      emptyState(
        "No requests found",
        state.requestQuery || state.requestFilter !== "all"
          ? "Change the search or status filter."
          : "After the first response, see routing, speed, tokens, and verification here.",
        button(
          "Open chat",
          "secondary-button",
          () => navigate("chat"),
          "chat",
        ),
      ),
    );
    return;
  }
  const wrap = text("div", "table-scroll");
  const table = text("table", "request-table");
  const head = document.createElement("thead");
  const tr = document.createElement("tr");
  for (const label of [
    "Request",
    "Status",
    "Route / expert",
    "Time",
    "Speed",
    "Tokens",
    "Attempts",
    "",
  ])
    tr.append(text("th", "", label));
  head.append(tr);
  table.append(head);
  const body = document.createElement("tbody");
  for (const r of rows) {
    const row = document.createElement("tr");
    const prompt = document.createElement("td");
    const open = button("", "request-prompt", () => openRequest(r));
    open.append(
      text("strong", "", r.prompt.slice(0, 78) || r.conversation.title),
      text("small", "", `${formatDate(r.date)} · ${r.source || "Chat"} · ${r.response.id || "no ID"}`),
    );
    prompt.append(open);
    const status = document.createElement("td");
    status.append(statusBadge(r.response.status));
    const route = document.createElement("td");
    route.append(
      text(
        "span",
        "table-expert",
        r.response.ghostswitch?.expert_id
          ? expertLabel(r.response.ghostswitch.expert_id)
          : r.response.ghostswitch?.route === "base"
            ? "Base model"
            : "—",
      ),
      text(
        "small",
        "muted",
        r.response.ghostswitch?.mode === "direct"
          ? "direct"
          : r.response.ghostswitch?.mode || "—",
      ),
    );
    row.append(
      prompt,
      status,
      route,
      text("td", "numeric", formatMs(r.response.ghostswitch?.latency_ms)),
      text("td", "numeric", formatRate(finalGeneration(r.response))),
      text("td", "numeric", formatNumber(r.response.usage?.total_tokens)),
      text(
        "td",
        "numeric",
        finite(r.response.ghostswitch?.verification?.attempts)
          ? `${r.response.ghostswitch.verification.attempts}/5`
          : "—",
      ),
    );
    const last = document.createElement("td");
    const detail = button("", "quiet-icon", () => openRequest(r), "arrowRight");
    detail.setAttribute("aria-label", "Request details");
    last.append(detail);
    row.append(last);
    body.append(row);
  }
  table.append(body);
  wrap.append(table);
  target.append(wrap);
}
function renderRequests() {
  const target = $("requestsView");
  target.replaceChildren();
  pageHeading(
    target,
    "REQUEST LOG",
    "Request log",
    "Browser conversations and the last 200 server requests, including OpenCode and API. Updated every 5 seconds.",
    [
      button(
        "Export JSON",
        "secondary-button",
        () =>
          download(
            "ghostswitch-requests.json",
            JSON.stringify(
              records().map((r) => ({
                date: r.date,
                source: r.source || "Chat",
                conversation: r.conversation.title,
                prompt: r.prompt,
                request: r.message.request || null,
                response: r.response,
              })),
              null,
              2,
            ),
          ),
        "download",
      ),
    ],
  );
  const rows = records();
  const summary = text("div", "request-summary");
  summary.append(
    text("span", "", `${rows.length} requests`),
    text(
      "span",
      "summary-success",
      `${rows.filter((r) => r.response.status === "success").length} accepted`,
    ),
    text(
      "span",
      "summary-error",
      `${rows.filter((r) => ["failed", "unsupported"].includes(r.response.status)).length} errors`,
    ),
  );
  target.append(summary);
  const toolbar = text("div", "table-toolbar");
  const search = text("label", "search-field");
  const input = document.createElement("input");
  input.placeholder = "Text, ID, expert…";
  input.setAttribute("aria-label", "Search requests");
  input.value = state.requestQuery;
  search.append(icon("search"), input);
  toolbar.append(search);
  const filters = text("div", "segmented-control");
  for (const [value, label] of [
    ["all", "All"],
    ["success", "Accepted"],
    ["running", "Running"],
    ["queued", "Queued"],
    ["errors", "Errors"],
    ["canceled", "Canceled"],
  ]) {
    const b = button(
      label,
      state.requestFilter === value ? "active" : "",
      () => {
        state.requestFilter = value;
        renderRequests();
      },
    );
    b.setAttribute("aria-pressed", String(state.requestFilter === value));
    filters.append(b);
  }
  toolbar.append(filters);
  target.append(toolbar);
  const table = text("div", "panel table-panel");
  target.append(table);
  const populate = () => {
    table.replaceChildren();
    const query = state.requestQuery.toLocaleLowerCase();
    const matches = rows.filter(
      (r) =>
        (state.requestFilter === "all" ||
          (state.requestFilter === "errors" &&
            ["failed", "unsupported"].includes(r.response.status)) ||
          r.response.status === state.requestFilter) &&
        `${r.prompt} ${r.response.id || ""} ${r.response.ghostswitch?.expert_id || ""} ${r.conversation.title} ${r.source || ""}`
          .toLocaleLowerCase()
          .includes(query),
    );
    requestTable(matches, table);
  };
  input.oninput = () => {
    state.requestQuery = input.value;
    populate();
  };
  populate();
  target.append(
    text(
      "div",
      "page-footnote",
      state.journalError || "API requests persist on the server across restarts. Full agent usage is available in ghostswitch.pipeline_usage. Speed refers to the last generation.",
    ),
  );
}
function openRequest(record) {
  state.requestDetailId = record.response.id;
  renderResponseDetails($("requestDetail"), record);
  const links = text("div", "detail-actions");
  if (!record.external) links.append(
    button(
      "Open conversation",
      "primary-button",
      () => {
        closeDialog("detailOverlay");
        openConversation(record.conversation.id);
        state.inspectorId = record.message.id;
        state.settings.inspector = true;
        state.inspectorMobileOpen = true;
        state.autoScroll = false;
        renderHeader();
        renderInspector();
        requestAnimationFrame(() =>
          $(`message-${record.message.id}`)?.scrollIntoView({ block: "start" }),
        );
      },
      "chat",
    ),
  );
  $("requestDetail").append(links);
  openDialog("detailOverlay");
}

function codeBlock(content, language = "JSON") {
  const wrap = text("div", "code-block");
  const top = text("div", "code-block-header");
  top.append(
    text("span", "", language),
    button("Copy", "text-button", () => copy(content), "copy"),
  );
  wrap.append(top, text("pre", "", content));
  return wrap;
}
function samplePayload() {
  return {
    model: "ghostswitch-v6",
    messages: [{ role: "user", content: "Explain how a hash table works." }],
    mode: "auto",
    max_tokens: state.settings.maxTokens,
    temperature: state.settings.temperature,
    stream: false,
  };
}
// --- Request builder: compose a chat request without writing JSON -------------------
function builderState() {
  state.builder ||= {
    route: "auto", system: "", message: "Explain how a hash table works.", stream: false,
    thinking: "auto", webSearch: "off", verify: true,
    maxTokens: Math.min(state.settings.maxTokens, 4096), temperature: state.settings.temperature,
  };
  return state.builder;
}
function builderPayload(b) {
  const messages = [];
  if (b.system.trim()) messages.push({ role: "system", content: b.system.trim() });
  messages.push({ role: "user", content: b.message.trim() || "Hello" });
  const mode = b.route === "auto" ? "auto" : b.route === "base" ? "base" : "direct";
  const body = { model: "ghostswitch-v6", messages, mode, max_tokens: b.maxTokens, temperature: b.temperature };
  if (mode === "direct") body.expert_id = b.route;
  if (b.stream) body.stream = true;
  if (b.thinking !== "auto") body.reasoning_effort = b.thinking === "on" ? "high" : "none";
  if (b.webSearch !== "off") body.web_search = b.webSearch;
  if (!b.verify) body.verify = false;
  return body;
}
function builderControl(label, control, hint) {
  const node = text("label", "builder-field");
  node.append(text("span", "", label), control);
  if (hint) node.append(text("small", "", hint));
  return node;
}
function builderSelect(options, value) {
  const select = document.createElement("select");
  select.className = "select-control";
  for (const [id, label, disabled] of options) {
    const option = text("option", "", label);
    option.value = id;
    option.disabled = Boolean(disabled);
    select.append(option);
  }
  select.value = value;
  return select;
}
function routeOptions() {
  const ready = (e) => e.enabled && e.downloaded;
  const groups = [["Official experts", state.experts.filter((e) => !e.manual_only)],
                  ["Hugging Face adapters", state.experts.filter((e) => e.manual_only)]];
  const select = document.createElement("select");
  select.className = "select-control";
  for (const [id, label] of [["auto", "Automatic: GhostSwitch chooses"], ["base", "Base model only, no expert"]]) {
    const option = text("option", "", label);
    option.value = id;
    select.append(option);
  }
  for (const [title, experts] of groups) {
    if (!experts.length) continue;
    const group = document.createElement("optgroup");
    group.label = title;
    for (const e of [...experts].sort((a, b) => Number(ready(b)) - Number(ready(a)))) {
      const option = text("option", "", ready(e) ? e.name : `${e.name} (turn on in Experts)`);
      option.value = e.id;
      option.disabled = !ready(e);
      group.append(option);
    }
    select.append(group);
  }
  return select;
}
function requestBuilder(onApply) {
  const b = builderState();
  const node = text("div", "request-builder");
  const route = routeOptions();
  route.value = [...route.options].some((o) => o.value === b.route && !o.disabled) ? b.route : "auto";
  const routeNote = text("small", "", "");
  const message = document.createElement("textarea");
  message.className = "builder-text";
  message.rows = 3;
  message.value = b.message;
  const system = document.createElement("textarea");
  system.className = "builder-text";
  system.rows = 2;
  system.placeholder = "Optional, for example: Answer in Russian, briefly.";
  system.value = b.system;
  const thinking = builderSelect([["auto", "Auto"], ["on", "On"], ["off", "Off"]], b.thinking);
  const search = builderSelect([["off", "Off"], ["auto", "Auto"], ["on", "Always"]], b.webSearch);
  const tokens = document.createElement("input");
  tokens.type = "number";
  tokens.min = "1";
  tokens.max = "10240";
  tokens.value = b.maxTokens;
  const temperature = document.createElement("input");
  temperature.type = "number";
  temperature.min = "0";
  temperature.max = "2";
  temperature.step = "0.1";
  temperature.value = b.temperature;
  const toggle = (label, checked, hint) => {
    const node = text("label", "builder-toggle");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = checked;
    node.append(input, text("span", "", label));
    node.title = hint;
    return [node, input];
  };
  const [streamNode, stream] = toggle("Stream tokens (SSE)", b.stream, "Send the answer piece by piece as it is generated.");
  const [verifyNode, verify] = toggle("Check the answer", b.verify, "The verifier reviews the answer and asks for a retry when it finds a problem.");
  const apply = () => {
    Object.assign(b, {
      route: route.value, message: message.value, system: system.value, thinking: thinking.value,
      webSearch: search.value, stream: stream.checked, verify: verify.checked,
      maxTokens: Math.max(1, Math.min(10240, Number(tokens.value) || 1024)),
      temperature: Math.max(0, Math.min(2, Number(temperature.value) || 0)),
    });
    describe();
    onApply(builderPayload(b));
  };
  const describe = () => {
    const expert = state.experts.find((e) => e.id === route.value);
    routeNote.textContent = route.value === "auto"
      ? "The router picks an official expert that is turned on, or the base model."
      : route.value === "base"
        ? "Answers with the base model alone; no adapter is applied."
        : expert?.manual_only
          ? "Applies this Hugging Face adapter directly. It is not verified for GhostSwitch."
          : "Applies this expert's adapter and prompt directly, without routing.";
  };
  describe();
  for (const control of [route, message, system, thinking, search, tokens, temperature, stream, verify]) {
    control.addEventListener(control.tagName === "SELECT" || control.type === "checkbox" ? "change" : "input", apply);
  }
  const options = text("div", "builder-options");
  options.append(
    builderControl("Thinking", thinking),
    builderControl("Web search", search),
    builderControl("Max tokens", tokens),
    builderControl("Temperature", temperature),
  );
  const toggles = text("div", "builder-toggles");
  toggles.append(streamNode, verifyNode);
  node.append(
    builderControl("Who answers", route),
    routeNote,
    builderControl("Message", message),
    builderControl("System instruction", system),
    options,
    toggles,
  );
  node.apply = apply;
  return node;
}
function curlSnippet(path, payload) {
  const body = JSON.stringify(payload).replace(/'/g, "'\\''");
  return `curl ${payload.stream ? "-N " : ""}${state.settings.baseUrl}${path} \\\n  -H 'Content-Type: application/json' \\\n  -d '${body}'`;
}
function pythonSnippet(path, payload) {
  return `import json\nfrom urllib.request import Request, urlopen\n\npayload = json.loads(r"""${JSON.stringify(payload, null, 2)}""")\nrequest = Request(\n    "${state.settings.baseUrl}${path}",\n    data=json.dumps(payload).encode("utf-8"),\n    headers={"Content-Type": "application/json"},\n)\nwith urlopen(request, timeout=900) as response:\n    ${payload.stream ? "for line in response:\n        print(line.decode(), end=\"\")" : "result = json.load(response)\n    print(result[\"choices\"][0][\"message\"][\"content\"])"}`;
}
function renderDocs() {
  const target = $("docsView");
  target.replaceChildren();
  pageHeading(
    target,
    "DEVELOPER TOOLS",
    "API playground",
    "Inspect requests, explore the API contract, and connect your client.",
    [
      button(
        "Copy Base URL",
        "secondary-button",
        () => copy(`${state.settings.baseUrl}/v1`),
        "copy",
      ),
    ],
  );
  const tabs = text("div", "page-tabs");
  for (const [id, label] of [
    ["playground", "Playground"],
    ["reference", "API reference"],
    ["examples", "Integration examples"],
  ]) {
    const tab = button(label, state.docsTab === id ? "active" : "", () => {
      savePlaygroundDraft();
      state.docsTab = id;
      renderDocs();
    });
    tab.setAttribute("aria-pressed", String(state.docsTab === id));
    tabs.append(tab);
  }
  target.append(tabs);
  if (state.docsTab === "reference") return renderApiReference(target);
  if (state.docsTab === "examples") return renderApiExamples(target);
  const grid = text("div", "playground-grid");
  const editor = panel("Request", "Call the current API");
  const endpoint = text("div", "playground-endpoint");
  const select = document.createElement("select");
  select.className = "select-control";
  select.id = "apiEndpoint";
  select.setAttribute("aria-label", "API endpoint");
  for (const [value, label] of [
    ["POST /v1/chat/completions", "POST /v1/chat/completions"],
    ["GET /v1/experts", "GET /v1/experts"],
    ["GET /v1/models", "GET /v1/models"],
    ["GET /v1/requests", "GET /v1/requests"],
    ["GET /health", "GET /health"],
  ]) {
    const option = text("option", "", label);
    option.value = value;
    select.append(option);
  }
  select.value = state.playgroundEndpoint || "POST /v1/chat/completions";
  endpoint.append(select);
  editor.append(
    endpoint,
    text("code", "endpoint-origin", state.settings.baseUrl),
  );
  const builder = requestBuilder((payload) => {
    select.value = "POST /v1/chat/completions";
    area.value = JSON.stringify(payload, null, 2);
    state.playgroundDraft = area.value;
    updateBody();
  });
  editor.append(builder);
  const jsonLabel = text("span", "builder-json-label", "Request JSON · edit freely; changing a field above rebuilds it");
  editor.append(jsonLabel);
  const area = document.createElement("textarea");
  area.id = "apiBody";
  area.className = "json-editor";
  area.spellcheck = false;
  area.setAttribute("aria-label", "Request JSON");
  area.value =
    state.playgroundDraft || JSON.stringify(builderPayload(builderState()), null, 2);
  area.oninput = () => (state.playgroundDraft = area.value);
  editor.append(area);
  const notes = text(
    "p",
    "panel-footnote",
    "POST runs real inference and may take several minutes. The result appears in the request log.",
  );
  editor.append(notes);
  const actions = text("div", "playground-actions");
  const run = button(
    state.playgroundBusy ? "Running…" : "Run request",
    "primary-button",
    runPlayground,
    "play",
  );
  run.id = "apiRun";
  run.disabled = Boolean(state.pending || state.playgroundBusy);
  const copySnippet = (kind) => {
    try {
      const [method, path] = select.value.split(" ");
      if (method !== "POST") return copy(`curl ${state.settings.baseUrl}${path}`);
      const payload = JSON.parse(area.value);
      copy(kind === "python" ? pythonSnippet(path, payload) : curlSnippet(path, payload));
    } catch (e) {
      toast(`Invalid JSON: ${e.message}`);
    }
  };
  actions.append(
    button("Copy cURL", "text-button", () => copySnippet("curl"), "copy"),
    button("Copy Python", "text-button", () => copySnippet("python"), "copy"),
    button("Format", "text-button", () => {
      try {
        area.value = JSON.stringify(JSON.parse(area.value), null, 2);
        state.playgroundDraft = area.value;
      } catch (e) {
        toast(`Invalid JSON: ${e.message}`);
      }
    }),
    run,
  );
  editor.append(actions);
  const response = panel("Server response", "HTTP status, timing, and raw JSON");
  const output = text("div", "playground-output");
  output.id = "apiOutput";
  response.append(output);
  grid.append(editor, response);
  target.append(grid);
  renderPlaygroundOutput();
  const updateBody = () => {
    const post = select.value.startsWith("POST");
    area.classList.toggle("hidden", !post);
    builder.classList.toggle("hidden", !post);
    jsonLabel.classList.toggle("hidden", !post);
    notes.textContent = post
      ? "POST runs real inference and may take several minutes. The result appears in the request log."
      : "GET reads server state without running inference.";
    state.playgroundEndpoint = select.value;
    state.playgroundDraft = area.value;
  };
  select.onchange = updateBody;
  updateBody();
  target.append(
    text(
      "div",
      "info-callout",
      "GhostSwitch supports text Chat Completions. mode and expert_id select LoRAs; ghostswitch contains telemetry. Chat SSE sends a provisional candidate. Tool requests send only the response that passed the checks.",
    ),
  );
}
function savePlaygroundDraft() {
  if ($("apiBody")) state.playgroundDraft = $("apiBody").value;
  if ($("apiEndpoint")) state.playgroundEndpoint = $("apiEndpoint").value;
}
function renderPlaygroundOutput() {
  const target = $("apiOutput");
  if (!target) return;
  target.replaceChildren();
  const result = state.playgroundResult;
  if (state.playgroundBusy) {
    const waiting = text("div", "empty-state pending-state");
    waiting.append(
      icon("activity"),
      text("h3", "", "Waiting for the server"),
      text("p", "", state.playgroundLive || "GhostSwitch is processing the request…"),
    );
    target.append(waiting);
    return;
  }
  if (!result) {
    target.append(
      emptyState(
        "Ready for your first request",
        "Choose an endpoint, edit the JSON, and click “Run request”.",
        null,
        "terminal",
      ),
    );
    return;
  }
  const meta = text("div", "playground-result-meta");
  meta.append(
    text(
      "span",
      `badge ${result.ok ? "success" : "danger"}`,
      result.status ? `HTTP ${result.status}` : "NETWORK ERROR",
    ),
    text("span", "muted", formatMs(result.latency)),
  );
  target.append(
    meta,
    codeBlock(JSON.stringify(result.body, null, 2), "RESPONSE"),
  );
}
async function runPlayground() {
  if (state.pending || state.playgroundBusy)
    return toast("Wait for the current request to finish");
  const [method, path] = $("apiEndpoint").value.split(" ");
  let payload;
  try {
    if (method === "POST") {
      payload = JSON.parse($("apiBody").value);
      if (!payload || typeof payload !== "object" || Array.isArray(payload))
        throw new Error("A JSON object is required");
    }
  } catch (e) {
    return toast(`Invalid JSON: ${e.message}`);
  }
  savePlaygroundDraft();
  state.playgroundBusy = true;
  state.playgroundLive = "";
  const started = performance.now();
  $("apiRun").disabled = true;
  $("apiRun").textContent = "Running…";
  renderPlaygroundOutput();
  renderHeader();
  try {
    const http = await fetch(`${state.settings.baseUrl}${path}`, {
      method,
      ...(payload
        ? {
            headers: { "Content-Type": "application/json", "X-GhostSwitch-Client": "API playground" },
            body: JSON.stringify(payload),
          }
        : {}),
      signal: AbortSignal.timeout(method === "GET" ? 10000 : 900000),
    });
    let body;
    if (http.headers.get("Content-Type")?.includes("text/event-stream")) {
      const pending = {
        conversationId: null,
        text: "",
        reasoning: "",
        status: "Generation",
        attempt: 1,
      };
      const final = await consumeCompletionStream(http, pending);
      body = {
        ...final,
        object: "chat.completion",
        choices:
          final.status === "success"
            ? [
                {
                  index: 0,
                  message: {
                    role: "assistant",
                    content: pending.text,
                    ...(pending.reasoning
                      ? { reasoning_content: pending.reasoning }
                      : {}),
                  },
                  finish_reason: final.finish_reason || "stop",
                },
              ]
            : [],
      };
    } else {
      const raw = await http.text();
      try {
        body = JSON.parse(raw);
      } catch {
        body = { raw };
      }
    }
    state.playgroundResult = {
      ok:
        http.ok &&
        (!body.status || body.status === "success" || body.status === "ok"),
      status: http.status,
      latency: performance.now() - started,
      body,
    };
    if (method === "POST") {
      const prompt =
        payload.messages?.filter((m) => m.role === "user").at(-1)?.content ||
        "API playground";
      const c = {
        id: uid(),
        title: `API · ${String(prompt).slice(0, 42)}`,
        source: "API playground",
        selection: payload.expert_id || "auto",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
      };
      c.messages.push(
        {
          id: uid(),
          role: "user",
          content: String(prompt),
          createdAt: Date.now(),
        },
        {
          id: uid(),
          role: "assistant",
          content:
            body.choices?.[0]?.message?.content ||
            body.error?.message ||
            "Server response without text",
          response: {
            ...body,
            status: body.status || (http.ok ? "success" : "failed"),
          },
          request: payload,
          createdAt: Date.now(),
        },
      );
      state.conversations.unshift(c);
      persist();
      renderSidebar();
    }
  } catch (e) {
    state.playgroundResult = {
      ok: false,
      status: null,
      latency: performance.now() - started,
      body: { error: { type: "network_error", message: e.message } },
    };
  } finally {
    state.playgroundBusy = false;
    renderHeader();
    if (state.view === "api") renderDocs();
  }
}
function renderApiReference(target) {
  const container = text("div", "reference-content");
  const intro = panel("Connection", "Client and API share an origin");
  intro.append(
    metaRow("Base URL", `${state.settings.baseUrl}/v1`),
    metaRow("Model", "ghostswitch-v6"),
    metaRow("Authentication", "Not provided by the local server"),
  );
  container.append(intro);
  const endpoints = [
    [
      "POST",
      "/v1/chat/completions",
      "Text generation",
      "auto selects one LoRA or the base model. direct requires an enabled expert_id from /v1/experts. stream: true enables SSE.",
    ],
    [
      "GET",
      "/v1/experts",
      "LoRA catalog",
      "Compatible experts for the current backend. Includes metadata, enabled/downloaded state, errors, and download progress.",
    ],
    [
      "GET",
      "/v1/models",
      "Model list",
      "Returns ghostswitch-v6 in model-list format. LoRAs are listed in /v1/experts.",
    ],
    [
      "GET",
      "/v1/requests",
      "Request log",
      "The last 200 requests from all API clients with source, stages, routing, verification, timing, and tokens. Persists across restarts.",
    ],
    [
      "GET",
      "/health",
      "API availability",
      "Returns status, model, and backend. Checks the HTTP server; weights are validated during inference.",
    ],
  ];
  const list = panel("Endpoints", "Available v6 server methods");
  for (const [method, path, title, description] of endpoints) {
    const entry = text("div", "endpoint-reference");
    const heading = text("div", "endpoint");
    heading.append(
      text("span", `method ${method.toLowerCase()}`, method),
      text("code", "", path),
    );
    entry.append(heading, text("h3", "", title), text("p", "", description));
    list.append(entry);
  }
  container.append(list);
  const fields = panel(
    "Chat Completions fields",
    "Unknown fields are rejected",
  );
  const table = text("table", "reference-table");
  const body = document.createElement("tbody");
  for (const [name, type, description] of [
    ["model", "string", "ghostswitch-v6"],
    [
      "messages",
      "array",
      "A nonempty {role, content} array. content is a string; at least one nonempty user message is required. Roles: system, developer, user, assistant, tool.",
    ],
    ["mode", "string", "auto by default; direct for a selected expert."],
    ["expert_id", "string", "Required in direct mode; omitted in auto mode."],
    [
      "max_tokens",
      "integer",
      "1–10,240 tokens per generation; 4,096 by default.",
    ],
    ["temperature", "number", "0–2; 0.7 by default."],
    ["stream", "boolean", "true for SSE; false for JSON (default)."],
    ["metadata", "object", "Optional JSON object."],
  ]) {
    const row = document.createElement("tr");
    row.append(
      text("th", "", name),
      text("td", "muted", type),
      text("td", "", description),
    );
    body.append(row);
  }
  table.append(body);
  const scroll = text("div", "table-scroll");
  scroll.append(table);
  fields.append(scroll);
  container.append(fields);
  const checks = panel(
    "Streaming, verification, and errors",
    "Current runtime behavior",
  );
  checks.append(
    text(
      "p",
      "",
      "The stream contains data: JSON and ends with data: [DONE]. Candidate text arrives in choices[0].delta.content and is accepted only after the final status: success.",
    ),
    text(
      "p",
      "",
      "On retry, delta.ghostswitch_reset clears the previous text and reasoning. delta.ghostswitch_attempt identifies the attempt; delta.ghostswitch_status reports the stage. The final chunk contains usage, status, ghostswitch, and error/routing on failure.",
    ),
    text(
      "p",
      "",
      "Thinking is automatic: a short check names the kind of request, and calculations, puzzles, proofs, plans, programming logic and comparisons reason before answering when the model runs on a GPU; an expert's response_policy.json may fix it either way. In an agent task, only the first step after a user message can reason; steps after tool results answer directly. Reasoning arrives in delta.reasoning_content and message.reasoning_content; content holds only the answer. reasoning_effort: none or minimal turns it off; low, medium, or high turns it on for any route. ghostswitch.thinking reports the decision and its source.",
    ),
    text(
      "p",
      "",
      "web_search: off (default), auto, or on. Sources are cited as [n] and listed in ghostswitch.web_search; POST /v1/search returns search results with page text.",
    ),
    text(
      "p",
      "",
      "Up to 5 generations: the first attempt and 4 retries. ACCEPT finishes immediately; an answer that fails a check is written again with the check's feedback.",
    ),
    text(
      "p",
      "",
      "200: success; 400: invalid format; 422: unavailable specialty; 500: generation or verification failure. With stream: true, HTTP 200 starts the stream; check its final status.",
    ),
    text(
      "p",
      "",
      "In chat, usage covers the full pipeline. For agents, usage covers the accepted generation; ghostswitch.pipeline_usage includes total usage. Maximum HTTP body: 2 MB. The context window, including history, depends on the base model: see GET /health.",
    ),
  );
  container.append(checks);
  const compatibility = text("div", "info-callout");
  compatibility.append(
    icon("code"),
    text(
      "p",
      "",
      "The text contract supports basic Chat Completions: model, messages, choices, usage, SSE, and /v1/models. Function tools, tool_choice, parallel_tool_calls, tool_calls, tool_call_id results, reasoning_effort, and reasoning_content are supported. Images, audio, and multiple response choices are not supported yet.",
    ),
  );
  container.append(compatibility);
  target.append(container);
}
function renderApiExamples(target) {
  const base = state.settings.baseUrl;
  const payload = samplePayload();
  const curl = `curl ${base}/v1/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(payload)}'`;
  const js = `const response = await fetch("${base}/v1/chat/completions", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify(${JSON.stringify(payload, null, 2).split("\n").join("\n  ")})\n});\nconst result = await response.json();\nif (!response.ok || result.status !== "success") {\n  throw new Error(result.error?.message || result.status);\n}\nconsole.log(result.choices[0].message.content);\nconsole.log(result.ghostswitch);`;
  const python = `import json\nfrom urllib.request import Request, urlopen\n\npayload = ${JSON.stringify(payload, null, 2).replace(/false/g, "False")}\nrequest = Request(\n    "${base}/v1/chat/completions",\n    data=json.dumps(payload).encode("utf-8"),\n    headers={"Content-Type": "application/json"},\n)\nwith urlopen(request, timeout=900) as response:\n    result = json.load(response)\nprint(result["choices"][0]["message"]["content"])`;
  const blocks = text("div", "reference-content");
  for (const [title, description, code, language] of [
    ["cURL", "A simple JSON request in auto mode.", curl, "SHELL"],
    ["JavaScript", "Check HTTP and final response status.", js, "JAVASCRIPT"],
    [
      "Python",
      "Standard library only; no extra dependencies.",
      python,
      "PYTHON",
    ],
  ]) {
    const section = panel(title, description);
    section.append(codeBlock(code, language));
    blocks.append(section);
  }
  target.append(blocks);
}
function renderSettings() {
  const target = $("settingsView");
  target.replaceChildren();
  pageHeading(
    target,
    "",
    "Settings",
    "Connection, generation settings, and browser data.",
  );
  const grid = text("div", "settings-grid");
  if (!state.runtime && !renderSettings.loading) {
    renderSettings.loading = true;
    refreshRuntime().finally(() => (renderSettings.loading = false));
  }
  grid.append(runtimePanel());
  const connection = panel("API connection", "Local GhostSwitch URL");
  connection.append(
    field(
      "Base URL",
      "settingsBase",
      "url",
      state.settings.baseUrl,
      "For example, http://127.0.0.1:8765. A separate origin requires server CORS support.",
    ),
  );
  connection.append(
    button(
      "Save and check",
      "secondary-button",
      () => {
        try {
          const url = new URL($("settingsBase").value.trim());
          if (
            !["http:", "https:"].includes(url.protocol) ||
            url.username ||
            url.password ||
            !["", "/"].includes(url.pathname) ||
            url.search ||
            url.hash
          )
            throw new Error("Enter an HTTP(S) origin without a path or query");
          if (state.pending || state.playgroundBusy)
            return toast("Wait for the current request to finish");
          state.settings.baseUrl = url.origin;
          state.health = null;
          state.healthAt = null;
          state.connected = false;
          state.experts = [];
          state.expertsAt = null;
          state.serverRequests = [];
          state.journalError = null;
          refreshRequestLog();
          persist();
          renderConnection();
          refreshServer();
          toast("URL saved");
        } catch (e) {
          toast(e.message);
        }
      },
      "refresh",
    ),
  );
  const appearance = panel("Appearance", "Theme and workspace");
  const themes = text("div", "theme-options");
  for (const [id, label, iconName] of [
    ["dark", "Dark", "moon"],
    ["light", "Light", "sun"],
  ]) {
    const b = button(
      label,
      `theme-option ${id}${state.settings.theme === id ? " selected" : ""}`,
      () => {
        state.settings.theme = id;
        persist();
        applyTheme();
        themes.querySelectorAll("button").forEach((n) => {
          n.classList.toggle("selected", n.dataset.theme === id);
          n.setAttribute("aria-pressed", String(n.dataset.theme === id));
        });
      },
      iconName,
    );
    b.dataset.theme = id;
    b.setAttribute("aria-pressed", String(state.settings.theme === id));
    themes.append(b);
  }
  appearance.append(themes);
  const toggle = text("label", "toggle-row");
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = state.settings.inspector;
  checkbox.onchange = () => {
    state.settings.inspector = checkbox.checked;
    persist();
  };
  toggle.append(text("span", "", "Show response details in chat"), checkbox);
  appearance.append(toggle);
  appearance.append(button("Show introduction", "secondary-button", () => openOnboarding(0), "sparkles"));
  grid.append(connection, appearance);
  const generation = panel(
    "Generation",
    "Defaults for new requests",
  );
  generation.append(
    field(
      "Temperature",
      "settingsTemperature",
      "number",
      state.settings.temperature,
      "0 gives more predictable responses; higher values add variety.",
      { min: 0, max: 2, step: 0.1 },
    ),
    field(
      "Maximum output tokens",
      "settingsTokens",
      "number",
      state.settings.maxTokens,
      "Per-response limit: 1–10,240. Retries use additional tokens.",
      { min: 1, max: 10240, step: 1 },
    ),
  );
  const system = text("label", "field");
  system.append(text("span", "", "System instruction"));
  const area = document.createElement("textarea");
  area.id = "settingsPrompt";
  area.rows = 5;
  area.placeholder = "For example: give concise answers with short examples.";
  area.value = state.settings.systemPrompt;
  system.append(
    area,
    text(
      "small",
      "",
      "Browser history and the last 200 server requests.",
    ),
  );
  generation.append(
    system,
    button(
      "Save settings",
      "primary-button",
      () => {
        const t = Number($("settingsTemperature").value);
        const n = Number($("settingsTokens").value);
        if (!Number.isFinite(t) || t < 0 || t > 2)
          return toast("Temperature: 0–2");
        if (!Number.isInteger(n) || n < 1 || n > 10240)
          return toast("Token limit: 1–10,240");
        state.settings.temperature = t;
        state.settings.maxTokens = n;
        state.settings.systemPrompt = area.value;
        persist();
        toast("Generation settings saved");
      },
      "check",
    ),
  );
  const data = panel(
    "History and data",
    "Browser conversations and the last 200 server API requests",
  );
  data.append(
    metaRow("Conversations", state.conversations.length),
    metaRow("Logged responses", records().length),
  );
  const size = new Blob([JSON.stringify(state.conversations)]).size;
  data.append(metaRow("History size", `${formatNumber(size / 1024)} KB`));
  const actions = text("div", "data-actions");
  actions.append(
    button(
      "Export history",
      "secondary-button",
      () =>
        download(
          "ghostswitch-workspace.json",
          JSON.stringify(
            {
              version: 2,
              conversations: state.conversations,
              settings: state.settings,
            },
            null,
            2,
          ),
        ),
      "download",
    ),
    button(
      "Import history",
      "secondary-button",
      () => $("importFile").click(),
      "upload",
    ),
  );
  data.append(
    actions,
    text(
      "p",
      "panel-footnote",
      "Import adds new conversations and skips duplicate IDs. Connection and generation settings are preserved.",
    ),
  );
  const shortcuts = text("div", "shortcuts");
  shortcuts.append(text("h3", "", "Shortcuts"));
  for (const [label, key] of [
    ["Search conversations", "⌘ / Ctrl K"],
    ["New chat", "⌘ / Ctrl N"],
    ["Close dialog", "Esc"],
    ["Send message", "Enter"],
  ])
    shortcuts.append(metaRow(label, key));
  data.append(shortcuts);
  grid.append(generation, data, webSearchPanel());
  target.append(grid);
}
function field(label, id, type, value, hint, attributes = {}) {
  const node = text("label", "field");
  node.append(text("span", "", label));
  const input = document.createElement("input");
  input.id = id;
  input.type = type;
  input.value = value;
  Object.assign(input, attributes);
  node.append(input);
  if (hint) node.append(text("small", "", hint));
  return node;
}
async function importHistory(file) {
  if (!file) return;
  if (file.size > 10 * 1024 * 1024)
    return toast("History file must be smaller than 10 MB");
  try {
    const data = JSON.parse(await file.text());
    const incoming = validateConversations(
      Array.isArray(data) ? data : data.conversations,
    );
    const known = new Set(state.conversations.map((c) => c.id));
    const additions = incoming.filter((c) => !known.has(c.id));
    state.conversations.push(...additions);
    if (!state.activeId) state.activeId = state.conversations[0]?.id || null;
    persist();
    render();
    toast(
      `Conversations added: ${additions.length}. Duplicates skipped: ${incoming.length - additions.length}.`,
    );
  } catch (e) {
    toast(`Import failed: ${e.message}`);
  } finally {
    $("importFile").value = "";
  }
}
let dialogReturnFocus = null;
function openDialog(id) {
  dialogReturnFocus = document.activeElement;
  $(id).classList.remove("hidden");
  $(id).querySelector("input, button, [tabindex]")?.focus();
}
function closeDialog(id) {
  $(id).classList.add("hidden");
  dialogReturnFocus?.focus();
}
function showModels() {
  $("expertSearch").value = "";
  renderExperts();
  openDialog("modelOverlay");
  $("expertSearch").focus();
}
function showSearch() {
  $("workspaceSearch").value = "";
  renderSearch();
  openDialog("searchOverlay");
  $("workspaceSearch").focus();
}
function renderSearch() {
  const target = $("searchResults");
  target.replaceChildren();
  const query = $("workspaceSearch").value.trim().toLocaleLowerCase();
  const matches = state.conversations
    .filter((c) =>
      `${c.title} ${c.messages.map((m) => `${m.content} ${m.response?.id || ""}`).join(" ")}`
        .toLocaleLowerCase()
        .includes(query),
    )
    .slice(0, 30);
  for (const c of matches) {
    const b = button("", "search-result", () => {
      closeDialog("searchOverlay");
      openConversation(c.id);
      if (query) {
        const m = c.messages.find((m) =>
          `${m.content} ${m.response?.id || ""}`
            .toLocaleLowerCase()
            .includes(query),
        );
        if (m) {
          state.autoScroll = false;
          requestAnimationFrame(() =>
            $(`message-${m.id}`)?.scrollIntoView({ block: "center" }),
          );
        }
      }
    });
    const info = text("span");
    info.append(
      text("strong", "", c.title),
      text(
        "small",
        "",
        `${c.messages.length} messages · ${formatDate(c.updatedAt)}`,
      ),
    );
    b.append(icon("chat"), info, icon("arrowRight"));
    target.append(b);
  }
  if (!matches.length)
    target.append(
      emptyState(
        "Nothing found",
        query
          ? "Try different text or a request ID."
          : "Create your first conversation.",
        null,
        "search",
      ),
    );
}
function bindEvents() {
  hydrateIcons();
  $("newChat").onclick = createConversation;
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => (b.onclick = () => navigate(b.dataset.view)));
  document.querySelector(".brand").onclick = (e) => {
    e.preventDefault();
    navigate("chat");
  };
  $("mobileMenu").onclick = () => {
    $("sidebar").classList.toggle("open");
    $("sidebarScrim").classList.toggle(
      "hidden",
      !$("sidebar").classList.contains("open"),
    );
  };
  $("sidebarScrim").onclick = closeSidebar;
  $("globalSearch").onclick = showSearch;
  $("searchButton").onclick = showSearch;
  $("workspaceSearch").oninput = renderSearch;
  $("themeButton").onclick = () => {
    state.settings.theme = state.settings.theme === "dark" ? "light" : "dark";
    persist();
    applyTheme();
    if (state.view === "settings")
      document.querySelectorAll(".theme-option").forEach((n) => {
        const selected = n.dataset.theme === state.settings.theme;
        n.classList.toggle("selected", selected);
        n.setAttribute("aria-pressed", String(selected));
      });
  };
  $("refreshServer").onclick = refreshServer;
  $("refreshExperts").onclick = refreshServer;
  $("expertSearch").oninput = renderExperts;
  $("composerModel").onclick = showModels;
  $("webSearchToggle").onclick = () => {
    const order = Object.keys(WEB_SEARCH_MODES);
    state.settings.webSearch = order[(order.indexOf(state.settings.webSearch) + 1) % order.length];
    persist();
    renderWebSearchToggle();
    toast(
      {
        off: "Web search off",
        auto: "Web search: GhostSwitch searches when the question needs current or specific facts",
        on: "Web search: every message searches the web first",
      }[state.settings.webSearch],
    );
  };
  $("thinkingToggle").onclick = () => {
    const order = Object.keys(THINKING_MODES);
    state.settings.thinking = order[(order.indexOf(state.settings.thinking) + 1) % order.length];
    persist();
    renderThinkingToggle();
    toast(
      {
        auto: "Thinking: GhostSwitch reasons first when the request needs working out",
        on: "Thinking: every answer reasons first, which takes longer",
        off: "Thinking off: answers start right away",
      }[state.settings.thinking],
    );
  };
  $("renameChat").onclick = () => renameConversation(activeConversation());
  $("exportChat").onclick = () => exportConversation(activeConversation());
  $("inspectorButton").onclick = () => toggleInspector();
  $("closeInspector").onclick = () => toggleInspector(false);
  document.addEventListener("click", (event) => {
    document.querySelectorAll(".floating-details[open]").forEach((menu) => {
      if (!menu.contains(event.target) || event.target.closest("button")) menu.open = false;
    });
  });
  for (const [id, close] of [
    ["modelOverlay", "closeModelDialog"],
    ["searchOverlay", "closeSearch"],
    ["detailOverlay", "closeDetail"],
    ["addExpertOverlay", "closeAddExpert"],
  ]) {
    $(close).onclick = () => closeDialog(id);
    $(id).onclick = (e) => {
      if (e.target === $(id)) closeDialog(id);
    };
  }
  $("closeOnboarding").onclick = closeOnboarding;
  $("composerForm").addEventListener("submit", sendMessage);
  $("attachButton").onclick = () => $("attachmentInput").click();
  $("attachmentInput").onchange = (event) => {
    addAttachments(event.target.files);
    event.target.value = "";
  };
  $("messageInput").addEventListener("paste", (event) => {
    const files = [...(event.clipboardData?.files || [])];
    if (!files.length) return;
    // Keep pasted text; an image copied alone becomes an attachment.
    if (!event.clipboardData.getData("text/plain")) event.preventDefault();
    addAttachments(files);
  });
  let dragDepth = 0;
  const hasFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
  $("chatView").addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    dragDepth += 1;
    $("composerForm").classList.add("dragging");
  });
  $("chatView").addEventListener("dragover", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  $("chatView").addEventListener("dragleave", (event) => {
    if (!hasFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) $("composerForm").classList.remove("dragging");
  });
  $("chatView").addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    $("composerForm").classList.remove("dragging");
    addAttachments(event.dataTransfer.files);
  });
  $("composerForm").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.closest(".generation-settings")) {
      event.preventDefault();
      event.target.blur();
    }
  });
  $("messageInput").addEventListener("input", () => {
    resizeInput();
    const c = activeConversation();
    if (c) {
      c.draft = $("messageInput").value;
      clearTimeout(persist.draftTimer);
      persist.draftTimer = setTimeout(persist, 400);
    }
  });
  $("messageInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!state.pending && !state.playgroundBusy)
        $("composerForm").requestSubmit();
    }
  });
  // While a reply streams, the view follows it until the user scrolls up by any amount:
  // trackpads scroll in small steps, so a distance threshold alone never let go.
  let lastScrollTop = $("messageScroll").scrollTop;
  const stopFollowing = () => {
    const s = $("messageScroll");
    if (s.scrollHeight <= s.clientHeight + 2) return;
    state.autoScroll = false;
    $("scrollBottom").classList.remove("hidden");
  };
  $("messageScroll").addEventListener(
    "scroll",
    () => {
      const s = $("messageScroll");
      const distance = s.scrollHeight - s.scrollTop - s.clientHeight;
      // Content only grows, so a smaller scrollTop away from the bottom is the user.
      if (s.scrollTop < lastScrollTop - 1 && distance > 2) state.autoScroll = false;
      else if (distance < 40) state.autoScroll = true;
      lastScrollTop = s.scrollTop;
      $("scrollBottom").classList.toggle("hidden", state.autoScroll);
    },
    { passive: true },
  );
  $("messageScroll").addEventListener(
    "wheel",
    (event) => {
      if (event.deltaY < 0) stopFollowing();
    },
    { passive: true },
  );
  let touchY = null;
  $("messageScroll").addEventListener("touchstart", (event) => (touchY = event.touches[0]?.clientY ?? null), { passive: true });
  $("messageScroll").addEventListener(
    "touchmove",
    (event) => {
      const y = event.touches[0]?.clientY;
      // A finger moving down scrolls the conversation up.
      if (touchY != null && y != null && y > touchY + 4) stopFollowing();
    },
    { passive: true },
  );
  $("messageScroll").addEventListener("keydown", (event) => {
    if (["ArrowUp", "PageUp", "Home"].includes(event.key)) stopFollowing();
  });
  $("scrollBottom").onclick = scrollToBottom;
  $("maxTokens").onchange = () => {
    const n = Number($("maxTokens").value);
    if (Number.isInteger(n) && n > 0 && n <= 10240) {
      state.settings.maxTokens = n;
      persist();
    } else {
      $("maxTokens").value = state.settings.maxTokens;
      toast("Token limit: 1–10,240");
    }
  };
  $("temperature").onchange = () => {
    const t = Number($("temperature").value);
    if (Number.isFinite(t) && t >= 0 && t <= 2) {
      state.settings.temperature = t;
      persist();
    } else toast("Temperature: 0–2");
    $("temperature").value = state.settings.temperature.toFixed(1);
  };
  $("importFile").onchange = () => importHistory($("importFile").files[0]);
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      showSearch();
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") {
      e.preventDefault();
      createConversation();
    }
    const overlay = [...document.querySelectorAll(".overlay")].find(
      (n) => !n.classList.contains("hidden"),
    );
    if (e.key === "Escape") {
      document.querySelectorAll(".floating-details[open]").forEach((menu) => {
        menu.open = false;
        menu.querySelector("summary")?.focus();
      });
      if (overlay?.id === "onboardingOverlay") closeOnboarding();
      else if (overlay) closeDialog(overlay.id);
      else {
        closeSidebar();
        if (matchMedia("(max-width: 1000px)").matches) {
          toggleInspector(false);
        }
      }
    }
    if (e.key === "Tab" && overlay) {
      const focusables = [
        ...overlay.querySelectorAll(
          "button:not(:disabled), input, textarea, select, a[href], summary",
        ),
      ].filter((n) => n.getClientRects().length);
      const first = focusables[0];
      const last = focusables.at(-1);
      if (
        e.shiftKey &&
        (document.activeElement === first ||
          !overlay.contains(document.activeElement))
      ) {
        e.preventDefault();
        last?.focus();
      } else if (
        !e.shiftKey &&
        (document.activeElement === last ||
          !overlay.contains(document.activeElement))
      ) {
        e.preventDefault();
        first?.focus();
      }
    }
  });
  matchMedia("(max-width: 1000px)").addEventListener("change", renderHeader);
  window.addEventListener("hashchange", () => {
    const view = location.hash.slice(1);
    if (views[view]) navigate(view);
  });
  window.addEventListener("beforeunload", () => {
    saveDraft();
    persist();
  });
}
restore();
bindEvents();
render();
if (!state.settings.onboarded) openOnboarding(0);
$("messageInput").value = activeConversation()?.draft || "";
resizeInput();
pruneAttachments();
refreshServer();
setInterval(() => {
  if (state.view === "experts" || !$("modelOverlay").classList.contains("hidden") ||
      state.experts.some(e => ["queued", "downloading"].includes(e.status))) refreshServer();
}, 1000);
setInterval(refreshServer, 30000);

refreshRequestLog();
setInterval(refreshRequestLog, 5000);
