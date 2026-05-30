function getSystemTheme() {
  // Browser prefers-color-scheme: default when the user hasn't toggled
  // explicitly. Defensive on non-browser contexts (tests, SSR) — falls
  // back to "dark" if matchMedia is unavailable.
  if (typeof window !== "undefined" && window.matchMedia) {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return "dark";
}

const emptyDetail = {
  id: null,
  title: "",
  deal_description: "",
  messages: [],
  versions: [],
  clarification_question: null,
  status: "in_progress",
  deal_type: null,
  latest_docx_url: null,
  updated_at: null,
};

const state = {
  cases: [],
  currentCaseId: null,
  detail: { ...emptyDetail },
  prompt: "",
  error: null,
  theme: getSystemTheme(),
  sidebarOpen: true,
  searchOpen: false,
  searchTerm: "",
  authChecked: false,
  currentUser: null,
  ws: null,
  wsReady: false,
  wsRetry: 0,
  // Billing UI state. `view` is the SPA "page" — 'chat' (default), 'plans'
  // (subscription picker), or 'return' (post-checkout polling screen).
  view: "chat",
  billing: null, // { plan: {...}, generations_used_this_month, monthly_generation_limit, expires_at, billing_enabled }
  plansCatalog: null, // [{code, title, price_rub, ...}]
  userMenuOpen: false,
  paymentInFlight: false,
  // Quota error overlay: { kind: 'quota_exceeded' | 'edit_not_allowed', message }
  quotaError: null,
  // Return-screen pending payment id (URL ?payment_id=...).
  pendingPaymentId: null,
  returnPolling: false,
  returnTimedOut: false,
  // Settings modal: { open, prefs, draft, saving, error } | null
  settings: null,
};

class UnauthorizedError extends Error {
  constructor() {
    super("Unauthorized");
    this.name = "UnauthorizedError";
  }
}

class PaymentRequiredError extends Error {
  constructor(detail) {
    const message = (detail && (detail.message || detail.error)) || "Доступ ограничен тарифом";
    super(message);
    this.name = "PaymentRequiredError";
    this.kind = (detail && detail.error) || "quota_exceeded";
    this.detail = detail || {};
  }
}

function isLoadingForCurrentCase() {
  return hasProcessingMessage(state.detail);
}

function hasProcessingMessage(detail) {
  if (!detail || !Array.isArray(detail.messages)) return false;
  return detail.messages.some((m) => m && m.status === "processing");
}

const PROCESSING_LOADER_TEXT = "Обрабатываю запрос";

const stageMessages = {
  classify_contract: "Определяю тип договора",
  retrieve_norms: "Ищу подходящие законодательные нормы",
  analyze_norms: "Анализирую законодательные нормы",
  generate_recommendations: "Готовлю юридические рекомендации",
  enrich_recommendations: "Уточняю ссылки на нормы",
  generate_contract: "Формулирую договор",
  edit_contract: "Применяю правки в договоре",
  validate_contract: "Проверяю итоговый документ",
  default: PROCESSING_LOADER_TEXT,
};

function stageLoaderText(stage) {
  if (!stage) return PROCESSING_LOADER_TEXT;
  return stageMessages[stage] || PROCESSING_LOADER_TEXT;
}

const iconPaths = {
  Search: '<circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path>',
  PanelLeft: '<rect width="18" height="18" x="3" y="3" rx="2"></rect><path d="M9 3v18"></path>',
  Scale: '<path d="M12 3v18"></path><path d="m19 8 3 8a5 5 0 0 1-6 0zV7"></path><path d="M3 7h1a17 17 0 0 0 8-2 17 17 0 0 0 8 2h1"></path><path d="m5 8 3 8a5 5 0 0 1-6 0zV7"></path><path d="M7 21h10"></path>',
  SquarePen: '<path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z"></path>',
  Moon: '<path d="M20.985 12.486a9 9 0 1 1-9.47-9.47c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.343-.215.825-.003.8.399Z"></path>',
  Sun: '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2"></path><path d="M12 20v2"></path><path d="m4.93 4.93 1.41 1.41"></path><path d="m17.66 17.66 1.41 1.41"></path><path d="M2 12h2"></path><path d="M20 12h2"></path><path d="m6.34 17.66-1.41 1.41"></path><path d="m19.07 4.93-1.41 1.41"></path>',
};

function icon(name) {
  return `<svg class="lucide-icon" aria-hidden="true" viewBox="0 0 24 24">${iconPaths[name] || ""}</svg>`;
}

function sidebarToggleIcon() {
  if (state.sidebarOpen) return icon("PanelLeft");
  return `<span class="sidebar-icon-default">${icon("Scale")}</span><span class="sidebar-icon-hover">${icon("PanelLeft")}</span>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fallbackMarkdownToHtml(text) {
  return escapeHtml(text || "")
    .replace(/^###\s+(.*)$/gm, "<h3>$1</h3>")
    .replace(/^##\s+(.*)$/gm, "<h2>$1</h2>")
    .replace(/^#\s+(.*)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br />");
}

function wrapMarkdown(text) {
  if (window.marked?.parse && window.DOMPurify?.sanitize) {
    window.marked.setOptions({ breaks: true, gfm: true });
    return window.DOMPurify.sanitize(window.marked.parse(text || ""));
  }
  return `<p>${fallbackMarkdownToHtml(text)}</p>`;
}

function clipText(text, limit = 1200) {
  const value = String(text || "");
  return value.length > limit ? `${value.slice(0, limit).trim()}...` : value;
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function statusLabel(value) {
  const labels = {
    in_progress: "В работе",
    completed: "Готово",
    error: "Требует внимания",
  };
  return labels[value] || "Новый";
}

function statusTone(value) {
  if (value === "completed") return "success";
  if (value === "error") return "danger";
  return "progress";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(status) {
  return [502, 503, 504].includes(status);
}

function friendlyHttpError(status, body) {
  if (isRetryableStatus(status)) {
    return "Сервис еще запускается или временно недоступен. Попробуйте обновить страницу через несколько секунд.";
  }
  if (body && typeof body === "object" && body.detail) {
    return String(body.detail);
  }
  if (typeof body === "string" && body.trim() && !body.trim().startsWith("<")) {
    return body;
  }
  return `Ошибка сервера: HTTP ${status}`;
}

async function api(path, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  const maxAttempts = method === "GET" ? 4 : 1;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    const response = await fetch(path, {
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      ...options,
    });
    if (response.status === 401) {
      throw new UnauthorizedError();
    }
    const type = response.headers.get("content-type") || "";
    if (response.ok) {
      return type.includes("application/json") ? response.json() : response.text();
    }
    const body = type.includes("application/json") ? await response.json().catch(() => null) : await response.text();
    if (response.status === 402) {
      // Server sends {"detail": {"error": "...", "message": "..."}}.
      const detail = body && typeof body === "object" ? body.detail || body : null;
      throw new PaymentRequiredError(detail);
    }
    if (attempt < maxAttempts && isRetryableStatus(response.status)) {
      await sleep(700 * attempt);
      continue;
    }
    throw new Error(friendlyHttpError(response.status, body));
  }
  throw new Error("Сервис временно недоступен.");
}

function handleUnauthorized() {
  state.currentUser = null;
  state.cases = [];
  state.currentCaseId = null;
  state.detail = { ...emptyDetail };
  closeWebSocket();
  renderSoon();
}

async function checkAuth() {
  try {
    state.currentUser = await api("/api/auth/me");
  } catch (error) {
    if (error instanceof UnauthorizedError) {
      state.currentUser = null;
    } else {
      state.error = error.message || String(error);
    }
  } finally {
    state.authChecked = true;
  }
  if (state.currentUser) {
    // Pull saved theme from the server so reloads keep the user's choice.
    // Fire-and-forget — applyTheme() inside re-renders if it differs.
    loadThemeFromServer();
  }
}

async function logout() {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } catch (error) {
    /* ignore */
  }
  handleUnauthorized();
}

function applyTheme() {
  document.documentElement.dataset.theme = state.theme;
}

// Debounced renderer. Server emits message_updated + case_version_added +
// case_updated + case_clarification_changed in succession after a contract
// finishes; each arrives in its own JS task on the loopback network, often
// 5-30 ms apart. rAF batches only within a single frame, so spaced-out
// events still produced visible double redraws. setTimeout with a small
// window (32 ms = ~2 frames) coalesces them into a single DOM pass — short
// enough that the user perceives the result as instant.
const _RENDER_BATCH_MS = 32;
let _renderTimer = null;
function renderSoon() {
  if (_renderTimer !== null) return;
  _renderTimer = setTimeout(() => {
    _renderTimer = null;
    render();
    scrollChatToBottom();
  }, _RENDER_BATCH_MS);
}

// Escape hatch for the few call sites that need a synchronous render
// before reading layout (e.g. focus management right after switching cases).
function renderNow() {
  if (_renderTimer !== null) {
    clearTimeout(_renderTimer);
    _renderTimer = null;
  }
  render();
  scrollChatToBottom();
}

function setState(patch) {
  Object.assign(state, patch);
  renderSoon();
}

function sortCases(cases) {
  return [...cases].sort((a, b) => new Date(b.updated_at || b.created_at || 0) - new Date(a.updated_at || a.created_at || 0));
}

function sortVersions(versions) {
  return [...(versions || [])].sort((a, b) => {
    if ((b.version_number || 0) !== (a.version_number || 0)) return (b.version_number || 0) - (a.version_number || 0);
    return new Date(b.created_at || 0) - new Date(a.created_at || 0);
  });
}

function filterCases(cases) {
  const query = state.searchTerm.trim().toLowerCase();
  if (!query) return cases;
  return cases.filter((item) => `${item.title || ""} ${item.deal_type || ""} ${item.status || ""}`.toLowerCase().includes(query));
}

// ---- URL routing ---------------------------------------------------------

function caseIdFromLocation() {
  const match = window.location.pathname.match(/^\/c\/([0-9a-f-]{36})\/?$/i);
  return match ? match[1] : null;
}

function viewFromLocation() {
  const path = window.location.pathname;
  if (path.startsWith("/billing/plans")) return "plans";
  if (path.startsWith("/billing/return")) return "return";
  return "chat";
}

function pushUrl(caseId) {
  const target = caseId ? `/c/${caseId}` : "/";
  if (window.location.pathname !== target) {
    window.history.pushState({}, "", target);
  }
}

function pushView(view, query = "") {
  const path = view === "plans" ? "/billing/plans" : view === "return" ? "/billing/return" : "/";
  const target = query ? `${path}${query}` : path;
  if (window.location.pathname + window.location.search !== target) {
    window.history.pushState({}, "", target);
  }
}

async function navigateToPlans() {
  state.view = "plans";
  state.userMenuOpen = false;
  pushView("plans");
  if (!state.plansCatalog) {
    try {
      state.plansCatalog = await api("/api/billing/plans");
    } catch (error) {
      showError(error);
    }
  }
  if (!state.billing) {
    await loadBilling().catch(() => null);
  }
  renderSoon();
}

window.addEventListener("popstate", () => {
  state.view = viewFromLocation();
  if (state.view === "plans") {
    if (!state.plansCatalog) api("/api/billing/plans").then((p) => { state.plansCatalog = p; renderSoon(); }).catch(() => null);
    renderSoon();
    return;
  }
  if (state.view === "return") {
    state.pendingPaymentId = new URLSearchParams(window.location.search).get("payment_id");
    pollSubscriptionAfterReturn();
    renderSoon();
    return;
  }
  const caseId = caseIdFromLocation();
  if (caseId) {
    if (caseId !== state.currentCaseId) {
      loadCase(caseId, false).then(renderSoon).catch(showError);
    }
  } else {
    state.currentCaseId = null;
    state.detail = { ...emptyDetail };
    renderSoon();
  }
});

// ---- WebSocket -----------------------------------------------------------

function closeWebSocket() {
  if (state.ws) {
    try {
      state.ws.close();
    } catch (e) {
      /* ignore */
    }
    state.ws = null;
    state.wsReady = false;
  }
}

function connectWebSocket() {
  if (!state.currentUser) return;
  closeWebSocket();
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${window.location.host}/api/ws`;
  const ws = new WebSocket(url);
  state.ws = ws;
  ws.addEventListener("open", () => {
    state.wsReady = true;
    state.wsRetry = 0;
    console.info("ws open");
  });
  ws.addEventListener("close", (ev) => {
    state.wsReady = false;
    state.ws = null;
    console.warn("ws closed code=", ev.code);
    if (ev.code === 4401) {
      handleUnauthorized();
      return;
    }
    state.wsRetry += 1;
    const delay = Math.min(15000, 500 * 2 ** Math.min(state.wsRetry, 5));
    setTimeout(() => connectWebSocket(), delay);
  });
  ws.addEventListener("error", (err) => {
    console.warn("ws error", err);
  });
  ws.addEventListener("message", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      handleWsEvent(data);
    } catch (e) {
      console.warn("bad ws frame", e);
    }
  });
}

function handleWsEvent(event) {
  switch (event.type) {
    case "case_created": {
      const summary = event.case;
      if (summary) {
        const without = state.cases.filter((c) => c.id !== summary.id);
        state.cases = sortCases([summary, ...without]);
        updateCaseListOnly();
      }
      break;
    }
    case "case_updated": {
      const summary = event.case;
      if (summary) {
        state.cases = sortCases(state.cases.map((c) => (c.id === summary.id ? { ...c, ...summary } : c)));
        if (state.currentCaseId === summary.id) {
          state.detail = { ...state.detail, status: summary.status, deal_type: summary.deal_type, latest_docx_url: summary.latest_docx_url };
        }
        updateCaseListOnly();
      }
      break;
    }
    case "case_stage_changed": {
      if (state.currentCaseId === event.case_id) {
        state.detail = { ...state.detail, processing_stage: event.stage };
        // Surgical update of every visible processing-message loader text.
        const stageText = stageLoaderText(event.stage);
        document.querySelectorAll(
          ".message__bubble--loading .typing-text"
        ).forEach((el) => {
          el.textContent = stageText;
        });
      }
      break;
    }
    case "case_clarification_changed": {
      // Fired by the backend after each run completes. Tells us whether
      // the agent is now waiting on another clarification or has moved on
      // — keeps the "Нужно уточнение" banner in sync without an F5.
      if (state.currentCaseId === event.case_id) {
        state.detail = {
          ...state.detail,
          clarification_needed: !!event.clarification_needed,
          clarification_question: event.clarification_question ?? null,
          processing_stage: event.processing_stage ?? state.detail.processing_stage,
        };
        renderSoon();
      }
      break;
    }
    case "case_title_changed": {
      state.cases = state.cases.map((c) => (c.id === event.case_id ? { ...c, title: event.title } : c));
      if (state.currentCaseId === event.case_id) {
        state.detail = { ...state.detail, title: event.title };
        renderSoon();
      } else {
        updateCaseListOnly();
      }
      break;
    }
    case "message_added": {
      if (state.currentCaseId === event.case_id) {
        const messages = state.detail.messages || [];
        if (!messages.some((m) => m.id === event.message.id)) {
          state.detail = { ...state.detail, messages: [...messages, event.message] };
          renderSoon();
        }
      } else {
        bumpCaseUpdatedAt(event.case_id);
      }
      break;
    }
    case "case_version_added": {
      if (state.currentCaseId === event.case_id) {
        const versions = state.detail.versions || [];
        if (!versions.some((v) => v.id === event.version.id)) {
          state.detail = { ...state.detail, versions: [...versions, event.version] };
          renderSoon();
        }
      } else {
        // For non-active cases, still bump updated_at so the sidebar reorders.
        bumpCaseUpdatedAt(event.case_id);
      }
      break;
    }
    case "message_updated": {
      if (state.currentCaseId === event.case_id) {
        const messages = (state.detail.messages || []).map((m) => (m.id === event.message.id ? event.message : m));
        state.detail = { ...state.detail, messages };
        renderSoon();
      } else {
        bumpCaseUpdatedAt(event.case_id);
      }
      // Sidebar may need to reflect "in progress" → "ready" indicator change.
      updateCaseListOnly();
      break;
    }
    case "subscription_updated": {
      // Webhook from YooKassa landed; refresh `/api/billing/me` to update
      // the plan label in the user menu and unlock features.
      loadBilling().then(renderSoon).catch(() => null);
      break;
    }
    default:
      console.debug("ws unknown event", event);
  }
}

function bumpCaseUpdatedAt(caseId) {
  const now = new Date().toISOString();
  state.cases = sortCases(state.cases.map((c) => (c.id === caseId ? { ...c, updated_at: now } : c)));
  updateCaseListOnly();
}

// ---- Billing -------------------------------------------------------------

async function loadBilling() {
  try {
    state.billing = await api("/api/billing/me");
  } catch (error) {
    if (error instanceof UnauthorizedError) {
      handleUnauthorized();
      return;
    }
    console.warn("loadBilling failed", error);
    state.billing = null;
  }
}

async function startCheckout(planCode) {
  if (state.paymentInFlight) return;
  state.paymentInFlight = true;
  state.error = null;
  renderSoon();
  try {
    const result = await api("/api/billing/checkout", {
      method: "POST",
      body: JSON.stringify({ plan_code: planCode }),
    });
    if (result && result.confirmation_url) {
      window.location.href = result.confirmation_url;
      return;
    }
    state.error = "Не удалось получить ссылку на оплату";
  } catch (error) {
    if (error instanceof UnauthorizedError) {
      handleUnauthorized();
      return;
    }
    state.error = error.message || String(error);
  } finally {
    state.paymentInFlight = false;
    renderSoon();
  }
}

let _returnPollTimer = null;

function pollSubscriptionAfterReturn() {
  if (_returnPollTimer) return;
  state.returnTimedOut = false;
  state.returnPolling = true;
  const startedAt = Date.now();
  const tick = async () => {
    await loadBilling();
    const sub = state.billing && state.billing.subscription_status;
    const planCode = state.billing && state.billing.plan && state.billing.plan.code;
    renderSoon();
    if (sub === "active" && planCode && planCode !== "free") {
      _returnPollTimer = null;
      state.returnPolling = false;
      window.history.replaceState({}, "", "/");
      state.view = "chat";
      state.pendingPaymentId = null;
      renderSoon();
      return;
    }
    if (Date.now() - startedAt > 30000) {
      _returnPollTimer = null;
      state.returnPolling = false;
      state.returnTimedOut = true;
      renderSoon();
      return;
    }
    _returnPollTimer = setTimeout(tick, 2000);
  };
  _returnPollTimer = setTimeout(tick, 0);
}

function retryReturnPoll() {
  _returnPollTimer = null;
  state.returnTimedOut = false;
  pollSubscriptionAfterReturn();
}

function dismissQuotaError() {
  state.quotaError = null;
  renderSoon();
}

function toggleUserMenu(force) {
  const next = typeof force === "boolean" ? force : !state.userMenuOpen;
  state.userMenuOpen = next;
  if (next) {
    _mountUserMenu();
    if (!state.billing) {
      loadBilling().then(() => {
        if (state.userMenuOpen) _mountUserMenu();
      }).catch(() => null);
    }
  } else {
    _unmountUserMenu();
  }
}

function _userBadgeWrap() {
  return document.querySelector(".user-badge-wrap");
}

function _mountUserMenu() {
  const wrap = _userBadgeWrap();
  if (!wrap) return;
  wrap.querySelector("#user-menu")?.remove();
  wrap.insertAdjacentHTML("afterbegin", renderUserMenu());
  _bindUserMenu(wrap.querySelector("#user-menu"));
}

function _unmountUserMenu() {
  _userBadgeWrap()?.querySelector("#user-menu")?.remove();
}

function _bindUserMenu(menu) {
  if (!menu) return;
  menu.querySelectorAll('[data-action]').forEach((btn) => {
    btn.addEventListener("click", () => {
      const action = btn.getAttribute("data-action");
      if (action === "logout") {
        logout().catch((err) => console.warn("logout failed", err));
      } else if (action === "upgrade") {
        navigateToPlans();
      } else if (action === "settings") {
        openSettings();
      }
    });
  });
}

document.addEventListener("click", (event) => {
  if (!state.userMenuOpen) return;
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (target.closest("#user-menu") || target.closest("#user-badge")) return;
  toggleUserMenu(false);
});

// ---- Settings modal -----------------------------------------------------

const DEFAULT_PREFS = {
  contract_generation_policy: "legal_only",
  ask_personal_data: true,
};

async function openSettings() {
  // Close the user menu without re-rendering the shell — direct DOM removal.
  state.userMenuOpen = false;
  document.querySelector(".user-menu-wrap #user-menu")?.remove();
  document.getElementById("user-menu")?.remove();

  state.settings = { open: true, prefs: null, draft: null, saving: false, error: "" };
  _mountSettingsModal();
  try {
    const prefs = await api("/api/preferences");
    state.settings.prefs = prefs;
    state.settings.draft = { ...prefs };
  } catch (error) {
    if (error instanceof UnauthorizedError) {
      handleUnauthorized();
      return;
    }
    state.settings.prefs = { ...DEFAULT_PREFS };
    state.settings.draft = { ...DEFAULT_PREFS };
    state.settings.error = error.message || String(error);
  }
  _mountSettingsModal();
}

function closeSettings() {
  state.settings = null;
  _unmountSettingsModal();
}

function _settingsRoot() {
  let root = document.getElementById("settings-root");
  if (!root) {
    root = document.createElement("div");
    root.id = "settings-root";
    document.body.appendChild(root);
  }
  return root;
}

function _mountSettingsModal() {
  const root = _settingsRoot();
  root.innerHTML = renderSettingsModal();
  _bindSettingsModalEvents(root);
}

function _unmountSettingsModal() {
  document.getElementById("settings-root")?.remove();
}

function _bindSettingsModalEvents(root) {
  root.querySelectorAll("[data-settings-dismiss]").forEach((el) => {
    el.addEventListener("click", (event) => {
      // Only dismiss when the click hits the element itself (the backdrop or
      // the close-X), not its children (the modal panel).
      if (event.target !== el) return;
      closeSettings();
    });
  });
  root.querySelectorAll("[data-settings-field]").forEach((field) => {
    const which = field.getAttribute("data-settings-field");
    field.addEventListener("change", () => {
      if (which === "policy") {
        patchSettingsDraft({ contract_generation_policy: field.value });
      } else if (which === "ask_personal_data") {
        patchSettingsDraft({ ask_personal_data: field.checked });
      }
    });
  });
  root.querySelector("#settings-save")?.addEventListener("click", saveSettings);
}

function patchSettingsDraft(patch) {
  if (!state.settings) return;
  // Mutate in place so we don't trigger a shell re-render (which would
  // collapse open <select>s, blink the page, and lose focus). The save
  // button's disabled state is refreshed surgically below.
  state.settings.draft = { ...(state.settings.draft || {}), ...patch };
  _refreshSettingsSaveButton();
}

function _refreshSettingsSaveButton() {
  const btn = document.getElementById("settings-save");
  if (!btn) return;
  const ctx = state.settings;
  const dirty = Boolean(
    ctx && ctx.prefs && ctx.draft && (
      ctx.prefs.contract_generation_policy !== ctx.draft.contract_generation_policy ||
      ctx.prefs.ask_personal_data !== ctx.draft.ask_personal_data
    )
  );
  btn.disabled = !dirty || Boolean(ctx?.saving);
  btn.textContent = ctx?.saving ? "Сохраняю..." : "Сохранить";
}

function _settingsDirty() {
  if (!state.settings || !state.settings.prefs || !state.settings.draft) return false;
  const a = state.settings.prefs;
  const b = state.settings.draft;
  return (
    a.contract_generation_policy !== b.contract_generation_policy ||
    a.ask_personal_data !== b.ask_personal_data
  );
}

async function saveSettings() {
  if (!state.settings || !_settingsDirty()) return;
  state.settings.saving = true;
  state.settings.error = "";
  _refreshSettingsSaveButton();
  _refreshSettingsErrorBanner();
  try {
    const updated = await api("/api/preferences", {
      method: "PUT",
      body: JSON.stringify(state.settings.draft),
    });
    state.settings.prefs = updated;
    state.settings.draft = { ...updated };
    state.settings.saving = false;
  } catch (error) {
    if (error instanceof UnauthorizedError) {
      handleUnauthorized();
      return;
    }
    state.settings.saving = false;
    state.settings.error = error.message || String(error);
  }
  _refreshSettingsSaveButton();
  _refreshSettingsErrorBanner();
}

function _refreshSettingsErrorBanner() {
  const root = document.getElementById("settings-root");
  if (!root) return;
  const panel = root.querySelector(".settings__panel");
  if (!panel) return;
  panel.querySelector(".settings__error")?.remove();
  const err = state.settings?.error;
  if (err) {
    const div = document.createElement("div");
    div.className = "error-banner settings__error";
    div.textContent = err;
    panel.appendChild(div);
  }
}

// ---- Cases / detail ------------------------------------------------------

async function refreshCases() {
  state.cases = sortCases(await api("/api/cases"));
  renderSoon();
}

async function loadCase(caseId, withPushState = true) {
  const detail = await api(`/api/cases/${caseId}`);
  state.currentCaseId = caseId;
  state.detail = detail;
  state.error = null;
  if (withPushState) pushUrl(caseId);
}

function newCase() {
  state.currentCaseId = null;
  state.detail = { ...emptyDetail };
  state.prompt = "";
  state.error = null;
  pushUrl(null);
  renderSoon();
}

async function sendPrompt() {
  const prompt = state.prompt.trim();
  if (!prompt) return;
  if (isLoadingForCurrentCase()) return;

  state.prompt = "";
  state.error = null;
  renderSoon();

  try {
    const response = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ prompt, case_id: state.currentCaseId }),
    });

    const { case_id, user_message, assistant_message } = response;

    // For a brand-new case, switch the URL/detail to the freshly created one.
    if (!state.currentCaseId || state.currentCaseId === case_id) {
      if (state.currentCaseId !== case_id) {
        state.currentCaseId = case_id;
        pushUrl(case_id);
      }
      const messages = state.detail.messages || [];
      const merged = mergeMessages(messages, [user_message, assistant_message]);
      // Reset stage so the loader starts at the default text and only switches
      // once we get the first `case_stage_changed` event from the server.
      state.detail = { ...state.detail, id: case_id, messages: merged, processing_stage: null };
      renderSoon();
    }

    // Sidebar: refresh the list so a new chat appears at top with its title.
    state.cases = sortCases(await api("/api/cases"));
    if (state.currentCaseId === case_id) {
      renderSoon();
    } else {
      updateCaseListOnly();
    }
  } catch (error) {
    if (error instanceof UnauthorizedError) {
      handleUnauthorized();
      return;
    }
    if (error instanceof PaymentRequiredError) {
      state.prompt = prompt;
      state.quotaError = { kind: error.kind, message: error.message };
      renderSoon();
      return;
    }
    state.prompt = prompt;
    state.error = error.message || String(error);
    renderSoon();
  }
}

function mergeMessages(existing, incoming) {
  const byId = new Map(existing.map((m) => [m.id, m]));
  for (const m of incoming) {
    if (m && m.id) byId.set(m.id, m);
  }
  return [...byId.values()].sort((a, b) => new Date(a.created_at || 0) - new Date(b.created_at || 0));
}

function toggleTheme() {
  state.theme = state.theme === "dark" ? "light" : "dark";
  applyTheme();
  renderSoon();
  // Persist asynchronously — never block the UI on the network. Failures
  // are tolerable; theme is in-state-only until next persist succeeds.
  persistTheme(state.theme);
}

let _themePersistInflight = null;
function persistTheme(theme) {
  if (!state.currentUser) return;  // not logged in; nothing to persist against
  if (_themePersistInflight === theme) return;  // dedup rapid toggles
  _themePersistInflight = theme;
  fetch("/api/preferences", {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ theme }),
  })
    .catch(() => {})
    .finally(() => {
      if (_themePersistInflight === theme) _themePersistInflight = null;
    });
}

async function loadThemeFromServer() {
  try {
    const res = await fetch("/api/preferences", { credentials: "include" });
    if (!res.ok) return;
    const data = await res.json();
    // Server returns null when the user has never toggled — keep the
    // system-derived theme we initialized with. Only override on an
    // explicit "dark"/"light" coming from the DB.
    if (data && (data.theme === "dark" || data.theme === "light")) {
      state.theme = data.theme;
      applyTheme();
    }
  } catch {}
}

function scrollChatToBottom() {
  const list = document.getElementById("chat-list");
  if (list) list.scrollTop = list.scrollHeight;
}

function updateCaseListOnly() {
  const container = document.getElementById("chat-history");
  if (container) {
    container.innerHTML = renderCaseList();
    bindCaseButtons();
  }
}

function renderCaseList() {
  const cases = filterCases(sortCases(state.cases));
  if (cases.length === 0) {
    return `<div class="empty-mini">${state.searchTerm ? "Ничего не найдено" : "Чатов пока нет"}</div>`;
  }
  return cases.map((item) => `
    <button class="chat-card ${state.currentCaseId === item.id ? "chat-card--active" : ""}" data-case-id="${item.id}">
      <span class="chat-card__title">${escapeHtml(item.title || "Новый чат")}</span>
      <span class="chat-card__meta">
        <span class="status-dot status-dot--${statusTone(item.status)}"></span>
        ${statusLabel(item.status)}${item.deal_type ? ` · ${escapeHtml(item.deal_type)}` : ""}
      </span>
    </button>
  `).join("");
}

function renderMessage(message) {
  if (message.status === "processing") {
    const stageText = stageLoaderText(state.detail.processing_stage);
    return `
      <article class="message message--${escapeHtml(message.role)}" data-message-id="${escapeHtml(message.id)}">
        <div class="message__bubble message__bubble--loading">
          <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
          <span class="typing-text">${escapeHtml(stageText)}</span>
        </div>
      </article>
    `;
  }
  if (message.status === "error") {
    return `
      <article class="message message--${escapeHtml(message.role)} message--error" data-message-id="${escapeHtml(message.id)}">
        <div class="message__bubble message__bubble--error">
          <div class="message__content markdown-body">${wrapMarkdown(message.content || "Ошибка обработки.")}</div>
          <div class="message__time">${formatDate(message.updated_at || message.created_at)}</div>
        </div>
      </article>
    `;
  }
  return `
    <article class="message message--${escapeHtml(message.role)}" data-message-id="${escapeHtml(message.id)}">
      <div class="message__bubble">
        <div class="message__content markdown-body">${wrapMarkdown(message.content)}</div>
        <div class="message__time">${formatDate(message.created_at)}</div>
      </div>
    </article>
  `;
}

function renderMessages() {
  const messages = [...(state.detail.messages || [])].sort(
    (a, b) => new Date(a.created_at || 0) - new Date(b.created_at || 0),
  );
  if (messages.length === 0) {
    return `
      <div class="welcome-card">
        <p class="eyebrow">PactumAI</p>
        <h2>Опишите сделку простыми словами</h2>
        <p>Я помогу проверить условия, собрать рекомендации и подготовить договор.</p>
      </div>
    `;
  }
  return messages.map(renderMessage).join("");
}

function renderVersions() {
  const versions = sortVersions(state.detail.versions);
  if (versions.length === 0) {
    return `
      <div class="contract-empty">
        <div class="contract-empty__icon">DOCX</div>
        <h3>Договоров пока нет</h3>
        <p>Когда агент подготовит документ, он появится здесь.</p>
      </div>
    `;
  }
  return versions.map((version) => `
    <article class="contract-card">
      <div class="contract-card__top">
        <div>
          <strong>Версия ${version.version_number}</strong>
          <span>${formatDate(version.created_at)}</span>
        </div>
        ${version.docx_url ? `<a class="download-button" href="${version.docx_url}" target="_blank" rel="noreferrer">Скачать</a>` : ""}
      </div>
      <div class="contract-preview markdown-body">${wrapMarkdown(clipText(version.content_md || ""))}</div>
    </article>
  `).join("");
}

function bindEvents() {
  bindCaseButtons();
  document.getElementById("new-chat-button")?.addEventListener("click", newCase);
  document.getElementById("search-button")?.addEventListener("click", () => {
    if (!state.sidebarOpen) {
      setState({ sidebarOpen: true, searchOpen: true });
    } else {
      setState({ searchOpen: !state.searchOpen });
    }
  });
  document.getElementById("sidebar-toggle")?.addEventListener("click", () => setState({ sidebarOpen: !state.sidebarOpen }));
  document.getElementById("theme-toggle")?.addEventListener("click", toggleTheme);
  const search = document.getElementById("chat-search");
  if (search) {
    search.value = state.searchTerm;
    search.addEventListener("input", (event) => {
      state.searchTerm = event.target.value;
      updateCaseListOnly();
    });
  }
  const prompt = document.getElementById("prompt-input");
  if (prompt) {
    prompt.value = state.prompt;
    prompt.addEventListener("input", (event) => {
      state.prompt = event.target.value;
    });
    prompt.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        sendPrompt();
      }
    });
  }
  document.getElementById("chat-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    sendPrompt();
  });
  document.getElementById("user-badge")?.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleUserMenu();
  });
  // The user-menu lives in its own mount inside .user-badge-wrap and has its
  // own handlers attached at mount time (see _bindUserMenu). After every
  // shell render we have to re-mount it if it was open, otherwise the new
  // .user-badge-wrap node won't have the menu.
  if (state.userMenuOpen) _mountUserMenu();
  document.querySelectorAll('[data-modal-dismiss]').forEach((el) => {
    el.addEventListener("click", (event) => {
      if (event.target !== el) return;
      dismissQuotaError();
    });
  });
  document.querySelectorAll('[data-modal-upgrade]').forEach((btn) => {
    btn.addEventListener("click", () => {
      dismissQuotaError();
      navigateToPlans();
    });
  });
  // Settings-modal events are bound separately via _bindSettingsModalEvents
  // when the modal is mounted; they don't go through bindEvents() since the
  // modal lives outside #app.
  bindResizeHandles();
}

const RESIZE_LIMITS = {
  left:  { min: 240, max: 520, varName: "--sidebar-width",     storageKey: "pactumai.sidebarWidth" },
  right: { min: 260, max: 640, varName: "--right-panel-width", storageKey: "pactumai.rightPanelWidth" },
};

function restorePanelWidths() {
  for (const cfg of Object.values(RESIZE_LIMITS)) {
    const saved = parseInt(localStorage.getItem(cfg.storageKey), 10);
    if (Number.isFinite(saved) && saved >= cfg.min && saved <= cfg.max) {
      document.documentElement.style.setProperty(cfg.varName, saved + "px");
    }
  }
}

const COLLAPSE_THRESHOLD = 120; // Драг левой ручки левее min на столько — сворачиваем сайдбар.
const EXPAND_THRESHOLD = 10;    // Драг правой границы свёрнутого сайдбара на столько вправо — раскрываем.
const COLLAPSED_VISUAL_WIDTH = 72;

function bindResizeHandles() {
  document.querySelectorAll(".resize-handle").forEach((handle) => {
    const side = handle.dataset.resize;
    const cfg = RESIZE_LIMITS[side];
    if (!cfg) return;
    handle.addEventListener("mousedown", (event) => {
      event.preventDefault();
      const startX = event.clientX;
      const startedClosed = side === "left" && !state.sidebarOpen;
      const storedWidth = parseInt(
        getComputedStyle(document.documentElement).getPropertyValue(cfg.varName).trim(),
        10
      ) || (side === "left" ? 310 : 360);
      const startWidth = startedClosed ? COLLAPSED_VISUAL_WIDTH : storedWidth;
      let opened = !startedClosed;

      document.body.classList.add("resizing");
      handle.classList.add("resize-handle--active");

      const cleanup = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.body.classList.remove("resizing");
        handle.classList.remove("resize-handle--active");
      };

      const onMove = (ev) => {
        const delta = side === "left" ? ev.clientX - startX : startX - ev.clientX;
        const raw = startWidth + delta;
        if (side === "left") {
          if (!opened) {
            if (delta >= EXPAND_THRESHOLD) {
              opened = true;
              const next = Math.max(cfg.min, Math.min(cfg.max, raw));
              document.documentElement.style.setProperty(cfg.varName, next + "px");
              setState({ sidebarOpen: true });
            }
            return;
          }
          if (raw < cfg.min - COLLAPSE_THRESHOLD) {
            opened = false;
            setState({ sidebarOpen: false });
            return;
          }
        }
        const next = Math.max(cfg.min, Math.min(cfg.max, raw));
        document.documentElement.style.setProperty(cfg.varName, next + "px");
      };
      const onUp = () => {
        cleanup();
        const final = parseInt(
          getComputedStyle(document.documentElement).getPropertyValue(cfg.varName).trim(),
          10
        );
        if (Number.isFinite(final)) localStorage.setItem(cfg.storageKey, String(final));
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
  });
}

function bindCaseButtons() {
  document.querySelectorAll("[data-case-id]").forEach((button) => {
    button.addEventListener("click", () => {
      loadCase(button.dataset.caseId).then(renderSoon).catch(showError);
    });
  });
}

function showError(error) {
  if (error instanceof UnauthorizedError) {
    handleUnauthorized();
    return;
  }
  setState({ error: error.message || String(error) });
}

function googleLogoSvg() {
  return `
    <svg class="g-logo" width="20" height="20" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#FFC107" d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z"/>
      <path fill="#FF3D00" d="M6.306 14.691l6.571 4.819C14.655 15.108 18.961 12 24 12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 16.318 4 9.656 8.337 6.306 14.691z"/>
      <path fill="#4CAF50" d="M24 44c5.166 0 9.86-1.977 13.409-5.192l-6.19-5.238C29.211 35.091 26.715 36 24 36c-5.202 0-9.619-3.317-11.283-7.946l-6.522 5.025C9.505 39.556 16.227 44 24 44z"/>
      <path fill="#1976D2" d="M43.611 20.083H42V20H24v8h11.303c-.792 2.237-2.231 4.166-4.087 5.571.001-.001.002-.001.003-.002l6.19 5.238C36.971 39.205 44 34 44 24c0-1.341-.138-2.65-.389-3.917z"/>
    </svg>
  `;
}

function yandexLogoSvg() {
  return `
    <svg class="ya-logo" width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="12" fill="#FC3F1D"/>
      <path fill="#fff" d="M13.32 6.4h-1.1C10.56 6.4 9.6 7.27 9.6 8.72c0 1.26.57 1.97 1.68 2.72l.93.63-2.7 4.33H7.8l2.48-3.97C8.9 11.5 7.98 10.4 7.98 8.68c0-2.1 1.45-3.48 3.68-3.48h2.86V16.4H13.3V6.4z"/>
    </svg>
  `;
}

function authThemeIcon() {
  return state.theme === "dark" ? icon("Moon") : icon("Sun");
}

function renderLoginScreen() {
  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="auth-aurora"></div>
    <button id="auth-theme-toggle" class="auth-theme-toggle" type="button" aria-label="Переключить тему">${authThemeIcon()}</button>
    <div class="auth-page">
      <section class="auth-hero">
        <div class="auth-hero__left">
          <span class="auth-eyebrow"><span class="auth-eyebrow__dot"></span>PactumAI · юридический ИИ</span>
          <h1 class="auth-brand">PactumAI</h1>
          <h2 class="auth-headline">От описания сделки — к готовому проекту договора</h2>
          <p class="auth-lead">Расскажите, о чём договор, простым языком. PactumAI определит правовую логику сделки, подберёт нормы, подготовит проект договора и выдаст рекомендации.</p>
          <div class="auth-cta">
            <div class="auth-buttons">
              <a class="google-button" href="/api/auth/google/login">${googleLogoSvg()}<span>Войти через Google</span></a>
              <a class="yandex-button" href="/api/auth/yandex/login">${yandexLogoSvg()}<span>Войти через Яндекс</span></a>
            </div>
          </div>
        </div>
        <div class="auth-hero__right">
          <div class="auth-preview">
            <div class="auth-preview__shine"></div>
            <div class="auth-preview__msg auth-preview__msg--user">
              <div class="auth-preview__bubble auth-preview__bubble--user">Хочу подарить другу свою старую книгу. Передам прямо сейчас из рук в руки, безвозмездно.</div>
            </div>
            <div class="auth-preview__msg auth-preview__msg--ai">
              <div class="auth-preview__avatar">⚖️</div>
              <div class="auth-preview__bubble auth-preview__bubble--ai">
                <div class="auth-preview__title">Анализ сделки</div>
                <ul class="auth-preview__list">
                  <li>Тип: <strong>договор дарения движимого имущества</strong> <span class="auth-preview__cite">[ГК РФ, ст. 572]</span></li>
                  <li>Письменная форма: <strong>не требуется</strong> при передаче в момент заключения <span class="auth-preview__cite">[ГК РФ, ст. 574]</span></li>
                  <li>Существенные условия: безвозмездность, конкретный предмет, согласие одаряемого</li>
                  <li>Риски: запрет дарения подарков, кроме обычных, между некоторыми лицами <span class="auth-preview__cite">[ГК РФ, ст. 575]</span></li>
                </ul>
                <div class="auth-preview__chips">
                  <span class="auth-preview__chip auth-preview__chip--docx"><span class="auth-preview__chip-icon">📄</span> dogovor_dareniya.docx</span>
                  <span class="auth-preview__chip">+ рекомендации к подписанию</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      <footer class="auth-footer">
        <span class="auth-footer__brand"><span class="auth-footer__logo">PactumAI</span> · Powered by AI</span>
      </footer>
    </div>
  `;
  document.getElementById("auth-theme-toggle")?.addEventListener("click", toggleTheme);
}

function renderAuthLoading() {
  const app = document.getElementById("app");
  app.innerHTML = `<div class="auth-screen"><div class="auth-card"><p>Загрузка...</p></div></div>`;
}

function render() {
  applyTheme();
  if (!state.authChecked) {
    renderAuthLoading();
    return;
  }
  if (!state.currentUser) {
    renderLoginScreen();
    return;
  }
  if (state.view === "plans") {
    renderPlansScreen();
    return;
  }
  if (state.view === "return") {
    renderReturnScreen();
    return;
  }
  const app = document.getElementById("app");
  const detail = state.detail || emptyDetail;
  const hasCase = Boolean(state.currentCaseId);
  const loading = isLoadingForCurrentCase();
  const prevHistoryScroll = document.getElementById("chat-history")?.scrollTop ?? 0;
  app.innerHTML = `
    <div class="shell ${state.sidebarOpen ? "" : "shell--sidebar-closed"}">
      <aside class="sidebar">
        <div class="sidebar__top">
          <div class="brand">
            <div class="brand__text">
              <strong>PactumAI</strong>
            </div>
          </div>
          <button id="sidebar-toggle" class="icon-button icon-button--close" type="button" aria-label="${state.sidebarOpen ? "Закрыть боковую панель" : "Открыть боковую панель"}">
            ${sidebarToggleIcon()}
            <span class="tooltip">${state.sidebarOpen ? "Закрыть боковую панель" : "Открыть боковую панель"}</span>
          </button>
        </div>

        <div class="sidebar__actions">
          <button id="new-chat-button" class="side-action side-action--primary" type="button">${icon("SquarePen")}<span>Новый чат</span></button>
          <button id="search-button" class="side-action" type="button">${icon("Search")}<span>Поиск в чатах</span></button>
        </div>

        ${state.searchOpen ? `<input id="chat-search" class="chat-search" placeholder="Название, тип, статус..." />` : ""}

        <div class="history-heading">Недавние</div>
        <div id="chat-history" class="chat-history">
          ${renderCaseList()}
        </div>

        ${state.currentUser ? `
          <div class="user-badge-wrap">
            <button id="user-badge" class="user-badge user-badge--clickable" type="button">
              ${state.currentUser.picture_url
                ? `<img class="user-badge__avatar" src="${escapeHtml(state.currentUser.picture_url)}" alt="" referrerpolicy="no-referrer" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'user-badge__avatar user-badge__avatar--fallback',textContent:${JSON.stringify((state.currentUser.name || state.currentUser.email || "?").trim().charAt(0).toUpperCase())}}))" />`
                : `<div class="user-badge__avatar user-badge__avatar--fallback">${escapeHtml((state.currentUser.name || state.currentUser.email || "?").trim().charAt(0).toUpperCase())}</div>`}
              <div class="user-badge__info">
                <div class="user-badge__name">${escapeHtml(state.currentUser.name || state.currentUser.email)}</div>
                <div class="user-badge__email">${escapeHtml(billingPlanLabel())}</div>
              </div>
            </button>
          </div>
        ` : ""}
      </aside>

      <div class="resize-handle resize-handle--left" data-resize="left" aria-hidden="true"></div>

      <main class="chat-column">
        <header class="chat-header">
          <div>
            <p class="eyebrow">${hasCase ? statusLabel(detail.status) : "Новый чат"}</p>
            <h1>${escapeHtml(detail.title || "Новый договор")}</h1>
          </div>
          <button id="theme-toggle" class="theme-toggle" type="button" aria-label="Переключить тему">
            ${state.theme === "dark" ? icon("Moon") : icon("Sun")}
          </button>
        </header>

        ${state.error ? `<div class="error-banner">${escapeHtml(state.error)}</div>` : ""}
        ${detail.clarification_question ? `<div class="notice"><strong>Нужно уточнение</strong><span>${escapeHtml(detail.clarification_question)}</span></div>` : ""}

        <section id="chat-list" class="chat-list">
          ${renderMessages()}
        </section>

        <form id="chat-form" class="composer">
          <textarea id="prompt-input" placeholder="Напишите, какой договор нужно подготовить..." rows="3" ${loading ? "disabled" : ""}></textarea>
          <div class="composer__bottom">
            <span>Ctrl + Enter для отправки</span>
            <button class="send-button" type="submit" ${loading ? "disabled" : ""}>${loading ? "Ждем ответ..." : "Отправить"}</button>
          </div>
        </form>
        <p class="composer__disclaimer">PactumAI может допускать ошибки. Рекомендуем проконсультироваться с профессиональным юристом.</p>
      </main>

      <div class="resize-handle resize-handle--right" data-resize="right" aria-hidden="true"></div>

      <aside class="contracts-panel">
        <div class="contracts-panel__header">
          <div>
            <p class="eyebrow">Документы</p>
            <h2>Договоры</h2>
          </div>
          <span class="contract-count">${sortVersions(detail.versions).length}</span>
        </div>
        <div class="contracts-list">
          ${renderVersions()}
        </div>
      </aside>
    </div>
    ${state.quotaError ? renderQuotaModal() : ""}
  `;
  bindEvents();
  const historyEl = document.getElementById("chat-history");
  if (historyEl && prevHistoryScroll > 0) historyEl.scrollTop = prevHistoryScroll;
  requestAnimationFrame(scrollChatToBottom);
}

// ---- Billing UI fragments ------------------------------------------------

function billingPlanLabel() {
  if (!state.currentUser) return "";
  if (!state.billing) return state.currentUser.email || "";
  const plan = state.billing.plan;
  if (!plan) return state.currentUser.email || "";
  if (plan.code === "free") {
    const used = state.billing.generations_used_this_month ?? 0;
    const cap = plan.monthly_generation_limit ?? 0;
    return `${plan.title} · ${used}/${cap} в месяц`;
  }
  return plan.title;
}

function renderUserMenu() {
  const billing = state.billing;
  const planTitle = billing && billing.plan ? billing.plan.title : "Загружаем...";
  const used = billing ? (billing.generations_used_this_month ?? 0) : 0;
  const cap = billing && billing.monthly_generation_limit;
  const usageLine = billing && cap != null
    ? `<div class="user-menu__usage">${used} / ${cap} генераций в этом месяце</div>`
    : (billing ? `<div class="user-menu__usage">Без лимита генераций</div>` : "");
  const expiresLine = billing && billing.expires_at
    ? `<div class="user-menu__expires">Действует до ${escapeHtml(formatDate(billing.expires_at))}</div>`
    : "";
  return `
    <div id="user-menu" class="user-menu" role="menu">
      <div class="user-menu__plan">${escapeHtml(planTitle)}</div>
      ${usageLine}
      ${expiresLine}
      <div class="user-menu__divider"></div>
      <button class="user-menu__item" data-action="upgrade" type="button">Обновить план</button>
      <button class="user-menu__item" data-action="settings" type="button">Настройки</button>
      <button class="user-menu__item user-menu__item--danger" data-action="logout" type="button">Выйти</button>
    </div>
  `;
}

function formatDate(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("ru-RU", { day: "2-digit", month: "long", year: "numeric" });
  } catch (e) {
    return iso;
  }
}

function renderSettingsModal() {
  const ctx = state.settings;
  if (!ctx || !ctx.open) return "";
  const draft = ctx.draft || DEFAULT_PREFS;
  const policy = draft.contract_generation_policy || "legal_only";
  const ask = draft.ask_personal_data !== false;
  const loading = !ctx.prefs;
  const dirty = ctx.prefs && (
    ctx.prefs.contract_generation_policy !== policy || ctx.prefs.ask_personal_data !== ask
  );
  return `
    <div class="modal-backdrop" data-settings-dismiss>
      <div class="modal modal--settings" role="dialog" aria-modal="true">
        <header class="settings__header">
          <h2>Настройки</h2>
          <button type="button" class="settings__close" data-settings-dismiss aria-label="Закрыть">×</button>
        </header>
        <nav class="settings__tabs" role="tablist">
          <button type="button" class="settings__tab settings__tab--active" role="tab">Предпочтения</button>
        </nav>
        <section class="settings__panel">
          ${loading ? `<p class="muted">Загрузка...</p>` : `
            <div class="settings__field">
              <label class="settings__label" for="pref-policy">Генерация письменного договора</label>
              <select id="pref-policy" data-settings-field="policy">
                <option value="legal_only" ${policy === "legal_only" ? "selected" : ""}>Не генерировать, если не обязательна по закону</option>
                <option value="always" ${policy === "always" ? "selected" : ""}>Всегда генерировать</option>
                <option value="always_ask" ${policy === "always_ask" ? "selected" : ""}>Всегда спрашивать</option>
              </select>
              <p class="settings__hint">Если по закону договор обязан быть письменным, он будет сформирован независимо от этого выбора.</p>
            </div>
            <div class="settings__field settings__field--check">
              <label class="settings__check">
                <input type="checkbox" data-settings-field="ask_personal_data" ${ask ? "checked" : ""} />
                <span>Спрашивать мои личные данные (ФИО, паспорт, ИНН, адрес и пр.)</span>
              </label>
              <p class="settings__hint">Если выключено, агент будет составлять договор с заглушками вместо реальных персональных данных.</p>
            </div>
          `}
          ${ctx.error ? `<div class="error-banner settings__error">${escapeHtml(ctx.error)}</div>` : ""}
        </section>
        <footer class="modal__actions settings__footer">
          <button type="button" class="modal__cancel" data-settings-dismiss>Закрыть</button>
          <button type="button" class="modal__primary" id="settings-save" ${(!dirty || ctx.saving) ? "disabled" : ""}>
            ${ctx.saving ? "Сохраняю..." : "Сохранить"}
          </button>
        </footer>
      </div>
    </div>
  `;
}

function renderQuotaModal() {
  const err = state.quotaError;
  if (!err) return "";
  const title = err.kind === "edit_not_allowed"
    ? "Правка договора недоступна"
    : "Бесплатный лимит исчерпан";
  const body = err.kind === "edit_not_allowed"
    ? "На бесплатном тарифе можно сгенерировать договор и задавать вопросы по нему, но не редактировать. Обновите подписку, чтобы продолжить."
    : (err.message || "Вы использовали все 10 бесплатных генераций в этом месяце. Чтобы продолжить, оформите подписку.");
  return `
    <div class="modal-backdrop" data-modal-dismiss>
      <div class="modal" role="dialog" aria-modal="true">
        <h2>${escapeHtml(title)}</h2>
        <p>${escapeHtml(body)}</p>
        <div class="modal__actions">
          <button class="modal__cancel" type="button" data-modal-dismiss>Закрыть</button>
          <button class="modal__primary" type="button" data-modal-upgrade>Купить подписку</button>
        </div>
      </div>
    </div>
  `;
}

function renderPlansScreen() {
  const app = document.getElementById("app");
  const plans = state.plansCatalog || [];
  const currentCode = state.billing && state.billing.plan ? state.billing.plan.code : "free";
  const billingEnabled = state.billing ? state.billing.billing_enabled : true;
  app.innerHTML = `
    <div class="plans-shell">
      <header class="plans-header">
        <button id="plans-back" class="plans-back" type="button">← Назад</button>
        <h1>Тарифы PactumAI</h1>
        <p>Выберите подписку, которая подходит вам. Платежи проводятся через ЮKassa.</p>
      </header>
      ${!billingEnabled ? `<div class="plans-disabled-banner">Платежи временно недоступны. Обратитесь к администратору.</div>` : ""}
      <section class="plan-grid">
        ${plans.map((p) => renderPlanCard(p, currentCode, billingEnabled)).join("")}
      </section>
      ${state.error ? `<div class="error-banner plans-error">${escapeHtml(state.error)}</div>` : ""}
    </div>
  `;
  document.getElementById("plans-back")?.addEventListener("click", () => {
    state.view = "chat";
    pushView("chat");
    renderSoon();
  });
  document.querySelectorAll("[data-buy-plan]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const code = btn.getAttribute("data-buy-plan");
      if (code) startCheckout(code);
    });
  });
}

function planPeriodLabel(plan) {
  if (plan.code === "free") return "";
  const d = plan.duration_days;
  if (!d) return "";
  if (d === 30) return "/ месяц";
  if (d === 365) return "/ год";
  if (d === 730) return "/ 2 года";
  if (d % 30 === 0) return `/ ${d / 30} мес.`;
  return `/ ${d} дней`;
}

function planPriceLabel(plan) {
  if (plan.code === "free") return "Бесплатно";
  return `${Math.round(plan.price_rub).toLocaleString("ru-RU")} ₽`;
}

function renderPlanCard(plan, currentCode, billingEnabled) {
  const isCurrent = plan.code === currentCode;
  const isFree = plan.code === "free";
  const description = wrapMarkdown(plan.description_md || "");
  const priceLabel = planPriceLabel(plan);
  const periodLabel = planPeriodLabel(plan);
  const disabled = isFree || isCurrent || !billingEnabled || state.paymentInFlight;
  const cta = isCurrent ? "Ваш план" : (isFree ? "Включён по умолчанию" : (state.paymentInFlight ? "..." : "Купить"));
  return `
    <article class="plan-card ${isCurrent ? "plan-card--current" : ""}">
      ${isCurrent ? `<div class="plan-card__badge plan-card__badge--current">Текущий</div>` : ""}
      <h2>${escapeHtml(plan.title)}</h2>
      <div class="plan-card__price">${priceLabel}<span>${escapeHtml(periodLabel)}</span></div>
      <div class="plan-card__description">${description}</div>
      <button class="plan-card__cta" type="button" data-buy-plan="${plan.code}" ${disabled ? "disabled" : ""}>${cta}</button>
    </article>
  `;
}

function renderReturnScreen() {
  const app = document.getElementById("app");
  const planCode = state.billing && state.billing.plan && state.billing.plan.code;
  const isReady = planCode && planCode !== "free" && state.billing.subscription_status === "active";
  const timedOut = state.returnTimedOut;

  let title;
  let body;
  let primaryLabel;
  let primaryAction;
  let secondaryHtml = "";

  if (isReady) {
    title = "Подписка активирована";
    body = `Вы подключили тариф «${escapeHtml(state.billing.plan.title)}». Можно вернуться к чату.`;
    primaryLabel = "К чату";
    primaryAction = "home";
  } else if (timedOut) {
    title = "Подтверждение оплаты задерживается";
    body =
      "Платёжный сервис ещё не подтвердил оплату. Это нормально для тестового режима — иногда уведомление приходит с задержкой. Можно подождать или вернуться в чат и проверить план позже.";
    primaryLabel = "Проверить ещё раз";
    primaryAction = "retry";
    secondaryHtml = `<button id="return-home" class="modal__cancel" type="button">Вернуться в чат</button>`;
  } else {
    title = "Проверяем оплату...";
    body = "Это занимает до полуминуты. Не закрывайте страницу.";
    primaryLabel = "Я подожду...";
    primaryAction = "home";
  }

  app.innerHTML = `
    <div class="return-shell">
      <div class="return-card">
        <h1>${escapeHtml(title)}</h1>
        <p>${escapeHtml(body)}</p>
        <div class="modal__actions" style="justify-content: center;">
          <button id="return-primary" class="modal__primary" type="button">${escapeHtml(primaryLabel)}</button>
          ${secondaryHtml}
        </div>
      </div>
    </div>
  `;
  document.getElementById("return-primary")?.addEventListener("click", () => {
    if (primaryAction === "retry") {
      retryReturnPoll();
    } else {
      state.view = "chat";
      pushView("chat");
      renderSoon();
    }
  });
  document.getElementById("return-home")?.addEventListener("click", () => {
    state.view = "chat";
    pushView("chat");
    renderSoon();
  });
}

async function bootstrap() {
  applyTheme();
  restorePanelWidths();
  render();
  await checkAuth();
  if (!state.currentUser) {
    render();
    return;
  }
  state.view = viewFromLocation();
  loadBilling().catch(() => null);
  if (state.view === "plans") {
    api("/api/billing/plans")
      .then((p) => { state.plansCatalog = p; renderSoon(); })
      .catch(() => null);
  } else if (state.view === "return") {
    state.pendingPaymentId = new URLSearchParams(window.location.search).get("payment_id");
    pollSubscriptionAfterReturn();
  }
  try {
    await refreshCases();
    const fromUrl = caseIdFromLocation();
    if (fromUrl) {
      try {
        await loadCase(fromUrl, false);
      } catch (e) {
        console.warn("could not load case from URL", e);
        pushUrl(null);
      }
    }
    connectWebSocket();
  } catch (error) {
    showError(error);
  }
  render();
}

bootstrap();
