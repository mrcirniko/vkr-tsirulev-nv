/* Vanilla-JS admin panel for managing the NPA corpus.
 *
 * Layout mirrors app.js (chat UI): left sidebar with brand + navigation
 * + admin badge, right panel renders the active section.
 *
 * Sections:
 *   - upload : drag-drop area + upload button (multi-file TXT)
 *   - sources: list of uploaded NPAs with index/delete actions
 *
 * Auth: POST /api/admin/login on the same Starlette session as the user
 * cookie. /api/admin/ws keeps live progress.
 */

const STATUS_LABELS = {
  uploaded: "Загружен",
  chunking: "Чанкинг…",
  ready: "Готов к индексации",
  indexing: "Индексация…",
  indexed: "Проиндексирован",
  failed: "Ошибка",
};

const SOURCE_GROUP_LABELS = {
  general: "Общие нормы",
  primal: "Специальные нормы",
  secondary: "Дополнительные нормы",
};

function getSystemTheme() {
  // Browser prefers-color-scheme: default when the admin hasn't toggled
  // explicitly. Falls back to "dark" outside a browser context.
  if (typeof window !== "undefined" && window.matchMedia) {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  return "dark";
}

const state = {
  view: "loading", // 'loading' | 'login' | 'app'
  section: "upload",
  theme: getSystemTheme(),
  admin: null,
  loginUsername: "",
  loginPassword: "",
  loginError: "",
  loginBusy: false,
  sources: [],
  uploadFiles: [],
  uploadExtractRefs: false,
  uploadProgress: 0,
  uploadBusy: false,
  uploadResults: [],
  uploadError: "",
  progressMap: {},
  selectedIds: new Set(),
  bulkSourceGroup: "primal",
  bulkRecreate: false,
  // True: перед загрузкой удалить старые чанки этого источника (safe
  // re-index, default). False: добавить новые чанки в коллекцию рядом
  // со старыми (полезно для side-by-side версий или append-режима).
  bulkDeleteExisting: true,
  modal: null,
  bulkBusy: false,
  ws: null,
  wsRetry: 0,
  plans: [],
  plansLoading: false,
  plansSavingCode: null,
  plansError: "",
  plansDirty: {}, // { [code]: { title, price_rub, ... } }
  plansPreviewOpen: false,
  dealTypes: null,            // { name: description, ... } from backend
  dealTypesText: "",          // textarea content (string)
  dealTypesLoading: false,
  dealTypesSaving: false,
  dealTypesError: "",
  dealTypesDirty: false,
  // Admin management (owner only)
  adminUsers: [],
  adminUsersLoading: false,
  adminUsersError: "",
  newAdminUsername: "",
  newAdminBusy: false,
  newAdminError: "",
  newAdminCreated: null, // { id, username, password } — shown once
};

class AdminUnauthorized extends Error {}

async function api(path, options = {}) {
  const isFormData = options.body instanceof FormData;
  const response = await fetch(path, {
    credentials: "include",
    ...options,
    headers: {
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
  });
  if (response.status === 401) throw new AdminUnauthorized();
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  const ct = response.headers.get("content-type") || "";
  return ct.includes("application/json") ? response.json() : response.text();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// Promise-based confirm dialog styled with the admin's own .modal classes.
// Replaces `window.confirm(...)` so we don't get the native ugly browser
// chrome. Resolves true on OK, false on Cancel / Escape / backdrop click.
function customConfirm(message, opts = {}) {
  return new Promise((resolve) => {
    const title = opts.title || "Подтверждение";
    const okLabel = opts.okLabel || "OK";
    const cancelLabel = opts.cancelLabel || "Отмена";
    const danger = !!opts.danger;

    const backdrop = document.createElement("div");
    backdrop.className = "modal-backdrop";
    backdrop.innerHTML = `
      <div class="modal" role="dialog" aria-modal="true" aria-labelledby="cc-title">
        <h2 id="cc-title">${escapeHtml(title)}</h2>
        <div class="modal__body" style="white-space: pre-wrap; line-height: 1.5;">${escapeHtml(message)}</div>
        <div class="modal__actions">
          <button type="button" class="btn" data-action="cancel">${escapeHtml(cancelLabel)}</button>
          <button type="button" class="btn ${danger ? "btn--danger" : "btn--primary"}" data-action="ok">${escapeHtml(okLabel)}</button>
        </div>
      </div>
    `;

    function close(result) {
      document.removeEventListener("keydown", onKey);
      backdrop.remove();
      resolve(result);
    }

    function onKey(e) {
      if (e.key === "Escape") close(false);
      else if (e.key === "Enter") close(true);
    }

    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) close(false);
    });
    backdrop.querySelector('[data-action="ok"]').addEventListener("click", () => close(true));
    backdrop.querySelector('[data-action="cancel"]').addEventListener("click", () => close(false));

    document.body.appendChild(backdrop);
    document.addEventListener("keydown", onKey);
    backdrop.querySelector('[data-action="ok"]').focus();
  });
}


function setState(patch) {
  Object.assign(state, patch);
  render();
}

let _adminThemePersistInflight = null;
function persistAdminTheme(theme) {
  if (!state.admin) return;
  if (_adminThemePersistInflight === theme) return;
  _adminThemePersistInflight = theme;
  fetch("/api/admin/theme", {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ theme }),
  })
    .catch(() => {})
    .finally(() => {
      if (_adminThemePersistInflight === theme) _adminThemePersistInflight = null;
    });
}

async function loadAdminThemeFromServer() {
  try {
    const res = await fetch("/api/admin/theme", { credentials: "include" });
    if (!res.ok) return;
    const data = await res.json();
    // null theme = админ ещё не выбирал → не оверрайдим системную.
    if (data && (data.theme === "dark" || data.theme === "light")) {
      state.theme = data.theme;
      applyTheme();
    }
  } catch {}
}

function applyTheme() {
  document.documentElement.dataset.theme = state.theme;
}

function toggleTheme() {
  state.theme = state.theme === "dark" ? "light" : "dark";
  applyTheme();
  persistAdminTheme(state.theme);
  render();
}

const iconPaths = {
  Upload: '<path d="M12 3v12"></path><path d="m6 9 6-6 6 6"></path><path d="M5 21h14"></path>',
  List: '<line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line>',
  Logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line>',
  Scale: '<path d="M12 3v18"></path><path d="m19 8 3 8a5 5 0 0 1-6 0zV7"></path><path d="M3 7h1a17 17 0 0 0 8-2 17 17 0 0 0 8 2h1"></path><path d="m5 8 3 8a5 5 0 0 1-6 0zV7"></path><path d="M7 21h10"></path>',
  Trash: '<path d="M3 6h18"></path><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"></path><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>',
  Download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line>',
  Play: '<polygon points="5 3 19 12 5 21 5 3"></polygon>',
  Refresh: '<polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>',
  DollarSign: '<line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>',
  FileText: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline>',
  Eye: '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle>',
  X: '<line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line>',
  GripVertical: '<circle cx="9" cy="5" r="1"></circle><circle cx="9" cy="12" r="1"></circle><circle cx="9" cy="19" r="1"></circle><circle cx="15" cy="5" r="1"></circle><circle cx="15" cy="12" r="1"></circle><circle cx="15" cy="19" r="1"></circle>',
  Moon: '<path d="M20.985 12.486a9 9 0 1 1-9.47-9.47c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.343-.215.825-.003.8.399Z"></path>',
  Sun: '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2"></path><path d="M12 20v2"></path><path d="m4.93 4.93 1.41 1.41"></path><path d="m17.66 17.66 1.41 1.41"></path><path d="M2 12h2"></path><path d="M20 12h2"></path><path d="m6.34 17.66-1.41 1.41"></path><path d="m19.07 4.93-1.41 1.41"></path>',
};

function icon(name) {
  return `<svg class="lucide-icon" aria-hidden="true" viewBox="0 0 24 24">${iconPaths[name] || ""}</svg>`;
}

// ---- Bootstrap ----

async function bootstrap() {
  try {
    const me = await api("/api/admin/me");
    setState({ view: "app", admin: me });
    if (window.location.pathname === "/admin/login") {
      window.history.replaceState({}, "", "/admin");
    }
    // Pull saved theme so reload keeps the admin's choice.
    loadAdminThemeFromServer();
    await loadSources();
    connectWS();
  } catch (err) {
    if (err instanceof AdminUnauthorized) {
      setState({ view: "login" });
      if (window.location.pathname !== "/admin/login") {
        window.history.replaceState({}, "", "/admin/login");
      }
    } else {
      setState({ view: "login", loginError: err.message });
    }
  }
}

async function loadAdminUsers() {
  setState({ adminUsersLoading: true, adminUsersError: "" });
  try {
    const list = await api("/api/admin/admins");
    setState({ adminUsers: list, adminUsersLoading: false });
  } catch (err) {
    if (err instanceof AdminUnauthorized) { goToLogin(); return; }
    setState({ adminUsersLoading: false, adminUsersError: err.message || String(err) });
  }
}

async function createAdminUser() {
  const username = state.newAdminUsername.trim();
  if (!username) return;
  setState({ newAdminBusy: true, newAdminError: "", newAdminCreated: null });
  try {
    const result = await api("/api/admin/admins", {
      method: "POST",
      body: JSON.stringify({ username }),
    });
    const adminUsers = [...state.adminUsers, { id: result.id, username: result.username, created_at: result.created_at }];
    setState({ adminUsers, newAdminUsername: "", newAdminBusy: false, newAdminCreated: result });
  } catch (err) {
    if (err instanceof AdminUnauthorized) { goToLogin(); return; }
    setState({ newAdminBusy: false, newAdminError: err.message || String(err) });
  }
}

async function deleteAdminUser(adminId, username) {
  if (!(await customConfirm(`Удалить администратора «${username}»?`, { title: "Удаление администратора", okLabel: "Удалить", danger: true }))) return;
  try {
    await api(`/api/admin/admins/${adminId}`, { method: "DELETE" });
    setState({ adminUsers: state.adminUsers.filter((a) => a.id !== adminId) });
  } catch (err) {
    if (err instanceof AdminUnauthorized) { goToLogin(); return; }
    setState({ adminUsersError: err.message || String(err) });
  }
}

function downloadPasswordFile(username, password) {
  const content = `Администратор: ${username}\nПароль: ${password}\n\nСохраните пароль в надёжном месте — он больше не будет показан.`;
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `admin_${username}_credentials.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

async function loadSources() {
  try {
    const list = await api("/api/admin/sources");
    setState({ sources: list });
  } catch (err) {
    if (err instanceof AdminUnauthorized) goToLogin();
  }
}

async function loadPlans() {
  setState({ plansLoading: true, plansError: "" });
  try {
    const list = await api("/api/admin/plans");
    setState({ plans: list, plansLoading: false, plansDirty: {} });
  } catch (err) {
    if (err instanceof AdminUnauthorized) {
      goToLogin();
      return;
    }
    setState({ plansLoading: false, plansError: err.message || String(err) });
  }
}

async function loadDealTypes() {
  setState({ dealTypesLoading: true, dealTypesError: "" });
  try {
    const data = await api("/api/admin/deal-types");
    setState({
      dealTypes: data,
      dealTypesText: JSON.stringify(data, null, 2),
      dealTypesLoading: false,
      dealTypesDirty: false,
    });
  } catch (err) {
    if (err instanceof AdminUnauthorized) {
      goToLogin();
      return;
    }
    setState({ dealTypesLoading: false, dealTypesError: err.message || String(err) });
  }
}

async function saveDealTypes() {
  let parsed;
  try {
    parsed = JSON.parse(state.dealTypesText);
  } catch (err) {
    setState({ dealTypesError: `Невалидный JSON: ${err.message}` });
    return;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    setState({ dealTypesError: "Корневое значение должно быть объектом {имя: описание}" });
    return;
  }
  for (const [k, v] of Object.entries(parsed)) {
    if (typeof k !== "string" || !k.trim() || typeof v !== "string") {
      setState({ dealTypesError: `Неверная запись для ключа "${k}": ключи и значения должны быть строками` });
      return;
    }
  }
  if (Object.keys(parsed).length === 0) {
    setState({ dealTypesError: "Каталог не может быть пустым" });
    return;
  }
  setState({ dealTypesSaving: true, dealTypesError: "" });
  try {
    const updated = await api("/api/admin/deal-types", {
      method: "PUT",
      body: JSON.stringify(parsed),
    });
    setState({
      dealTypes: updated,
      dealTypesText: JSON.stringify(updated, null, 2),
      dealTypesDirty: false,
      dealTypesSaving: false,
    });
  } catch (err) {
    if (err instanceof AdminUnauthorized) {
      goToLogin();
      return;
    }
    setState({ dealTypesSaving: false, dealTypesError: err.message || String(err) });
  }
}

async function savePlan(code) {
  const dirty = state.plansDirty[code];
  if (!dirty) return;
  setState({ plansSavingCode: code, plansError: "" });
  try {
    const updated = await api(`/api/admin/plans/${code}`, {
      method: "PATCH",
      body: JSON.stringify(dirty),
    });
    const plans = state.plans.map((p) => (p.code === code ? updated : p));
    const dirtyMap = { ...state.plansDirty };
    delete dirtyMap[code];
    setState({ plans, plansDirty: dirtyMap, plansSavingCode: null });
  } catch (err) {
    if (err instanceof AdminUnauthorized) {
      goToLogin();
      return;
    }
    setState({ plansSavingCode: null, plansError: err.message || String(err) });
  }
}

function patchPlanDraft(code, patch) {
  // Mutate in place so we don't lose <input> focus by triggering a full
  // setState→render cycle on every keystroke. Save/reset button state is
  // refreshed surgically by `_refreshPlanCardActions`.
  const next = { ...(state.plansDirty[code] || {}), ...patch };
  state.plansDirty = { ...state.plansDirty, [code]: next };
  _refreshPlanCardActions(code);
}

function _refreshPlanCardActions(code) {
  const card = document.querySelector(`.admin-plan-card[data-plan="${CSS.escape(code)}"]`);
  if (!card) return;
  const saveBtn = card.querySelector(`[data-save="${CSS.escape(code)}"]`);
  if (saveBtn) {
    const dirty = Boolean(state.plansDirty[code]);
    const saving = state.plansSavingCode === code;
    saveBtn.disabled = !(dirty && !saving);
    saveBtn.textContent = saving ? "Сохраняю..." : "Сохранить";
  }
  const actions = card.querySelector(".admin-plan-card__actions");
  if (!actions) return;
  let resetBtn = actions.querySelector(`[data-reset="${CSS.escape(code)}"]`);
  const dirty = Boolean(state.plansDirty[code]);
  if (dirty && !resetBtn) {
    resetBtn = document.createElement("button");
    resetBtn.type = "button";
    resetBtn.className = "btn btn--ghost";
    resetBtn.setAttribute("data-reset", code);
    resetBtn.textContent = "Отменить";
    resetBtn.addEventListener("click", () => {
      const next = { ...state.plansDirty };
      delete next[code];
      setState({ plansDirty: next });
    });
    saveBtn?.insertAdjacentElement("afterend", resetBtn);
  } else if (!dirty && resetBtn) {
    resetBtn.remove();
  }
}

function planDraftValue(plan, field) {
  const dirty = state.plansDirty[plan.code];
  if (dirty && Object.prototype.hasOwnProperty.call(dirty, field)) return dirty[field];
  return plan[field];
}

function goToLogin() {
  state.ws?.close();
  setState({ view: "login", admin: null, sources: [], progressMap: {} });
  window.history.replaceState({}, "", "/admin/login");
}

// ---- WebSocket ----

function connectWS() {
  if (state.ws) return;
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${window.location.host}/api/admin/ws`);
  state.ws = ws;
  ws.onopen = () => { state.wsRetry = 0; };
  ws.onmessage = (event) => {
    let evt;
    try { evt = JSON.parse(event.data); } catch { return; }
    if (evt.type === "admin_npa_progress_snapshot") {
      // Sent by the backend right after WS accept — rebuilds the in-memory
      // progressMap for any indexing/chunking job that was already in
      // flight when the page reloaded.
      const next = {};
      for (const item of evt.items || []) {
        if (!item || !item.npa_id) continue;
        next[item.npa_id] = {
          processed: item.processed ?? 0,
          total: item.total ?? 0,
          stage: item.stage || item.status || "",
        };
      }
      state.progressMap = next;
      render();
    } else if (evt.type === "admin_npa_progress") {
      state.progressMap[evt.npa_id] = {
        processed: evt.processed,
        total: evt.total,
        stage: evt.stage,
      };
      render();
    } else if (evt.type === "admin_npa_status") {
      if (evt.status === "indexed") {
        // Pin the bar at 100% for a moment before removing — gives the
        // operator time to see completion on fast files.
        const id = evt.npa_id;
        const cur = state.progressMap[id];
        if (cur && cur.total) {
          state.progressMap[id] = { ...cur, processed: cur.total };
          render();
          setTimeout(() => {
            const stale = state.progressMap[id];
            if (stale && stale.processed === stale.total) {
              delete state.progressMap[id];
              render();
            }
          }, 1500);
        }
      } else if (["ready", "failed"].includes(evt.status)) {
        delete state.progressMap[evt.npa_id];
      }
      loadSources();
    } else if (evt.type === "admin_npa_failed") {
      delete state.progressMap[evt.npa_id];
      loadSources();
    }
  };
  ws.onclose = (event) => {
    state.ws = null;
    if (event.code === 4401) { goToLogin(); return; }
    if (state.view === "app") {
      const delay = Math.min(1000 * Math.pow(2, state.wsRetry++), 15000);
      setTimeout(connectWS, delay);
    }
  };
  ws.onerror = () => {};
}

// ---- Login ----

async function submitLogin(event) {
  event.preventDefault();
  setState({ loginBusy: true, loginError: "" });
  try {
    await api("/api/admin/login", {
      method: "POST",
      body: JSON.stringify({
        username: state.loginUsername,
        password: state.loginPassword,
      }),
    });
    setState({ loginBusy: false, loginPassword: "" });
    bootstrap();
  } catch (err) {
    setState({
      loginBusy: false,
      loginError: err instanceof AdminUnauthorized ? "Неверные учетные данные" : err.message,
    });
  }
}

function renderLogin() {
  return `
    <div class="admin-login-page">
      <button id="admin-theme-toggle" class="theme-toggle" type="button" aria-label="Переключить тему">
        ${state.theme === "dark" ? icon("Moon") : icon("Sun")}
      </button>
      <form id="admin-login-form" class="admin-login-card">
        <h1>Вход</h1>
        <label class="admin-field">
          <span>Логин</span>
          <input type="text" id="admin-login-username"
            value="${escapeHtml(state.loginUsername)}"
            autocomplete="username" autofocus required />
        </label>
        <label class="admin-field">
          <span>Пароль</span>
          <input type="password" id="admin-login-password"
            value="${escapeHtml(state.loginPassword)}"
            autocomplete="current-password" required />
        </label>
        ${state.loginError ? `<div class="admin-error">${escapeHtml(state.loginError)}</div>` : ""}
        <button type="submit" class="btn btn--primary btn--wide" ${state.loginBusy ? "disabled" : ""}>
          ${state.loginBusy ? "Входим…" : "Войти"}
        </button>
      </form>
    </div>
  `;
}

function bindLogin(root) {
  const form = root.querySelector("#admin-login-form");
  const u = root.querySelector("#admin-login-username");
  const p = root.querySelector("#admin-login-password");
  u?.addEventListener("input", (e) => { state.loginUsername = e.target.value; });
  p?.addEventListener("input", (e) => { state.loginPassword = e.target.value; });
  form?.addEventListener("submit", submitLogin);
  root.querySelector("#admin-theme-toggle")?.addEventListener("click", toggleTheme);
}

// ---- Upload section ----

function pickFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const txtOnly = Array.from(fileList).filter((f) => f.name.toLowerCase().endsWith(".txt"));
  if (!txtOnly.length) {
    setState({ uploadError: "Можно загружать только .txt файлы" });
    return;
  }
  setState({
    uploadFiles: [...state.uploadFiles, ...txtOnly],
    uploadError: "",
  });
}

function removeFileAt(index) {
  const next = state.uploadFiles.slice();
  next.splice(index, 1);
  setState({ uploadFiles: next });
}

async function submitUpload(event) {
  event.preventDefault();
  if (!state.uploadFiles.length || state.uploadBusy) return;
  setState({ uploadBusy: true, uploadProgress: 0, uploadError: "", uploadResults: [] });

  const form = new FormData();
  for (const f of state.uploadFiles) form.append("files", f);
  form.append("extract_references", state.uploadExtractRefs ? "true" : "false");

  try {
    const result = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/admin/upload");
      xhr.withCredentials = true;
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          state.uploadProgress = Math.round((e.loaded / e.total) * 100);
          render();
        }
      };
      xhr.onload = () => {
        if (xhr.status === 401) reject(new AdminUnauthorized());
        else if (xhr.status >= 400) reject(new Error(xhr.responseText || `HTTP ${xhr.status}`));
        else {
          try { resolve(JSON.parse(xhr.responseText)); }
          catch { reject(new Error("Bad JSON in response")); }
        }
      };
      xhr.onerror = () => reject(new Error("Сетевая ошибка"));
      xhr.send(form);
    });
    setState({
      uploadBusy: false,
      uploadFiles: [],
      uploadProgress: 100,
      uploadResults: result.results || [],
    });
    loadSources();
  } catch (err) {
    if (err instanceof AdminUnauthorized) { goToLogin(); return; }
    setState({ uploadBusy: false, uploadError: err.message });
  }
}

function renderUploadSection() {
  const fileItems = state.uploadFiles.length
    ? state.uploadFiles.map((f, idx) => `
        <li class="upload-file">
          <span class="upload-file__name">${escapeHtml(f.name)}</span>
          <span class="upload-file__size">${(f.size / 1024).toFixed(1)} КБ</span>
          <button type="button" class="icon-button" data-action="remove-file" data-idx="${idx}" title="Убрать">${icon("Trash")}</button>
        </li>`).join("")
    : `<li class="upload-empty">Файлы не выбраны</li>`;

  const results = state.uploadResults.length ? `
    <div class="upload-results">
      <h3>Результат загрузки</h3>
      <ul class="upload-results__list">
        ${state.uploadResults.map((r) => `
          <li class="upload-result upload-result--${r.error ? "fail" : "ok"}">
            <span class="upload-result__name">${escapeHtml(r.filename)}</span>
            ${r.error
              ? `<span class="upload-result__msg">${escapeHtml(r.error)}</span>`
              : `<span class="upload-result__msg">Источник: <strong>${escapeHtml(r.npa.source_name)}</strong> · ${r.chunks_preview_count} чанков</span>`}
          </li>`).join("")}
      </ul>
    </div>` : "";

  return `
    <section class="admin-section">
      <header class="admin-section__head">
        <div>
          <h1>Загрузка НПА</h1>
          <p class="muted">Перетащите .txt файлы или выберите несколько сразу. Имя источника = имя файла без расширения.</p>
        </div>
      </header>

      <div class="admin-card">
        <div class="upload-drop" id="upload-drop">
          <input id="upload-input" type="file" accept=".txt" multiple hidden />
          <div class="upload-drop__icon">${icon("Upload")}</div>
          <div class="upload-drop__title">Перетащите файлы сюда</div>
          <div class="upload-drop__hint">или кликните, чтобы выбрать (поддерживается множественный выбор)</div>
        </div>

        <ul class="upload-list">${fileItems}</ul>

        <label class="admin-checkbox">
          <input type="checkbox" id="upload-extract-refs" ${state.uploadExtractRefs ? "checked" : ""} />
          <span>
            <strong>Обнаруживать ссылки между статьями</strong>
            <small class="muted">
              Помогает агенту находить связанные нормы при ответе на вопросы.
              Сильно замедляет загрузку — несколько минут на файл.
              Без галки ссылки в чанках не сохраняются.
            </small>
          </span>
        </label>

        ${state.uploadError ? `<div class="admin-error">${escapeHtml(state.uploadError)}</div>` : ""}
        ${state.uploadBusy ? (state.uploadProgress < 100 ? `
          <div class="progress">
            <div class="progress__label">
              <span>Передаём файлы…</span>
              <span class="progress__count">${state.uploadProgress}%</span>
            </div>
            <div class="progress__track">
              <div class="progress__fill" style="width:${state.uploadProgress}%"></div>
            </div>
          </div>` : `
          <div class="progress progress--indeterminate">
            <div class="progress__label">
              <span>Обрабатываем на сервере… это может занять несколько минут</span>
            </div>
            <div class="progress__track">
              <div class="progress__fill"></div>
            </div>
          </div>`) : ""}

        <div class="admin-card__actions">
          <button type="button" class="btn btn--primary" id="upload-submit"
            ${(!state.uploadFiles.length || state.uploadBusy) ? "disabled" : ""}>
            ${state.uploadBusy ? "Загружаем…" : `Загрузить (${state.uploadFiles.length})`}
          </button>
        </div>

        ${results}
      </div>
    </section>
  `;
}

function bindUploadSection(root) {
  const drop = root.querySelector("#upload-drop");
  const input = root.querySelector("#upload-input");
  const submit = root.querySelector("#upload-submit");

  drop?.addEventListener("click", () => input?.click());
  drop?.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("upload-drop--hover"); });
  drop?.addEventListener("dragleave", () => drop.classList.remove("upload-drop--hover"));
  drop?.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("upload-drop--hover");
    pickFiles(e.dataTransfer.files);
  });
  input?.addEventListener("change", (e) => {
    pickFiles(e.target.files);
    e.target.value = "";
  });
  root.querySelectorAll('[data-action="remove-file"]').forEach((btn) => {
    btn.addEventListener("click", () => removeFileAt(parseInt(btn.getAttribute("data-idx"), 10)));
  });
  root.querySelector("#upload-extract-refs")?.addEventListener("change", (e) => {
    state.uploadExtractRefs = e.target.checked;
  });
  submit?.addEventListener("click", submitUpload);
}

// ---- Sources section ----

function selectableSourceIds() {
  return state.sources
    .filter((s) => s.status !== "chunking" && s.status !== "indexing")
    .map((s) => s.id);
}

function toggleSelected(id) {
  if (state.selectedIds.has(id)) state.selectedIds.delete(id);
  else state.selectedIds.add(id);
  render();
}

function selectAll(checked) {
  if (checked) state.selectedIds = new Set(selectableSourceIds());
  else state.selectedIds = new Set();
  render();
}

function clearSelection() {
  state.selectedIds = new Set();
}

async function bulkDelete() {
  const ids = Array.from(state.selectedIds);
  if (!ids.length) return;
  if (!(await customConfirm(
    `Удалить выбранные источники: ${ids.length} шт.?\nТочки в индексе не очищаются автоматически.`,
    { title: "Удаление источников", okLabel: "Удалить", danger: true }
  ))) return;
  state.bulkBusy = true;
  render();
  try {
    for (const id of ids) {
      try {
        await api(`/api/admin/sources/${id}`, { method: "DELETE" });
      } catch (err) {
        if (err instanceof AdminUnauthorized) { goToLogin(); return; }
        // continue with the rest, but show the failure at the end
        console.warn(`delete ${id} failed`, err);
      }
    }
    clearSelection();
    await loadSources();
  } finally {
    state.bulkBusy = false;
    render();
  }
}

async function startBulkIndex() {
  if (!state.selectedIds.size || state.bulkBusy) return;
  const npas = state.sources.filter((s) => state.selectedIds.has(s.id));
  const groupLabel = SOURCE_GROUP_LABELS[state.bulkSourceGroup] || state.bulkSourceGroup;
  let confirmMsg = `Индексировать ${npas.length} источник(ов) в «${groupLabel}»?`;
  if (state.bulkRecreate) {
    confirmMsg += "\n\n⚠️ ВНИМАНИЕ: «Очистить индекс» удалит ВСЕ документы этого типа из коллекции — включая источники, не выбранные сейчас.";
  } else if (state.bulkDeleteExisting) {
    confirmMsg += "\n\n⚠️ Старые чанки выбранных источников будут удалены из индекса перед загрузкой новых.";
  }
  // Danger-styling: красная кнопка только когда стираем — recreate (wipe
  // всей коллекции) или delete_existing (удаление чанков выбранных
  // источников). Append-режим — обычная синяя кнопка.
  const isDanger = state.bulkRecreate || state.bulkDeleteExisting;
  let okLabel = "Индексировать";
  if (state.bulkRecreate) okLabel = "Очистить коллекцию и индексировать";
  else if (state.bulkDeleteExisting) okLabel = "Перезаписать и индексировать";
  if (!(await customConfirm(confirmMsg, {
    title: "Индексация источников",
    okLabel,
    danger: isDanger,
  }))) return;

  state.bulkBusy = true;
  render();
  try {
    for (let i = 0; i < npas.length; i++) {
      const npa = npas[i];
      try {
        await api(`/api/admin/sources/${npa.id}/index`, {
          method: "POST",
          body: JSON.stringify({
            source_group: state.bulkSourceGroup,
            // Only the FIRST source of a bulk run respects recreate=true so
            // we don't wipe the freshly indexed sources we just inserted.
            recreate_collection: state.bulkRecreate && i === 0,
            delete_existing_for_source: state.bulkDeleteExisting,
          }),
        });
      } catch (err) {
        if (err instanceof AdminUnauthorized) { goToLogin(); return; }
        console.warn(`index ${npa.id} failed`, err);
      }
    }
    clearSelection();
    await loadSources();
  } finally {
    state.bulkBusy = false;
    render();
  }
}

function renderSourcesSection() {
  if (!state.sources.length) {
    return `
      <section class="admin-section">
        <header class="admin-section__head">
          <div>
            <h1>Загруженные НПА</h1>
            <p class="muted">Здесь появляются файлы после загрузки.</p>
          </div>
          <button type="button" class="icon-button" id="sources-refresh" title="Обновить">${icon("Refresh")}</button>
        </header>
        <div class="admin-card admin-card--empty">
          <div class="empty-state">
            <div class="empty-state__icon">${icon("List")}</div>
            <p>Пока нет загруженных источников.</p>
            <p class="muted">Перейдите в раздел «Загрузка», чтобы добавить файлы.</p>
          </div>
        </div>
      </section>
    `;
  }

  const selectable = selectableSourceIds();
  const allSelected = selectable.length > 0 && selectable.every((id) => state.selectedIds.has(id));
  const someSelected = state.selectedIds.size > 0 && !allSelected;

  const rows = state.sources.map((npa) => {
    const progress = state.progressMap[npa.id];
    const inJob = npa.status === "chunking" || npa.status === "indexing";
    const checked = state.selectedIds.has(npa.id);
    return `
      <li class="source-row ${inJob ? "source-row--locked" : ""} ${checked ? "source-row--checked" : ""}">
        <label class="source-row__check">
          <input type="checkbox" data-action="toggle" data-id="${escapeHtml(npa.id)}"
            ${checked ? "checked" : ""} ${inJob ? "disabled" : ""} />
        </label>
        <div class="source-row__main">
          <div class="source-row__top">
            <h3 class="source-row__title">${escapeHtml(npa.source_name)}</h3>
            <span class="status-pill status-pill--${npa.status}">${STATUS_LABELS[npa.status] || npa.status}</span>
          </div>
          <div class="source-row__meta">
            <span title="${escapeHtml(npa.original_filename)}">${escapeHtml(npa.original_filename)}</span>
            <span>·</span>
            <span>${npa.chunks_count ?? "—"} чанков</span>
            <span>·</span>
            <span>загружен ${fmtDate(npa.created_at)}</span>
            ${npa.last_indexed_at
              ? `<span>·</span><span>в индексе: ${escapeHtml(SOURCE_GROUP_LABELS[_groupFromCollection(npa.last_indexed_collection)] || npa.last_indexed_collection)}</span>`
              : ""}
          </div>
          ${progress && progress.total > 0
            ? (() => {
                const pct = Math.round((progress.processed / progress.total) * 100);
                return `<div class="progress">
                  <div class="progress__label">
                    <span>${progress.processed === 0 ? "Подготовка…" : "Индексирую"}</span>
                    <span class="progress__count">${progress.processed} / ${progress.total}</span>
                  </div>
                  <div class="progress__track">
                    <div class="progress__fill" style="width:${pct}%"></div>
                  </div>
                </div>`;
              })()
            : ""}
          ${npa.error_text ? `<div class="source-row__error">${escapeHtml(npa.error_text)}</div>` : ""}
        </div>
        <div class="source-row__actions">
          ${npa.has_chunks_json
            ? `<a class="btn btn--small" href="/api/admin/sources/${escapeHtml(npa.id)}/chunks.json" title="Скачать JSON чанков">${icon("Download")}<span>JSON</span></a>`
            : ""}
          <a class="btn btn--small" href="/api/admin/sources/${escapeHtml(npa.id)}/raw.txt" title="Скачать исходный TXT">${icon("Download")}<span>TXT</span></a>
        </div>
      </li>
    `;
  }).join("");

  const groupOptions = Object.entries(SOURCE_GROUP_LABELS)
    .map(([k, label]) => `<option value="${k}" ${state.bulkSourceGroup === k ? "selected" : ""}>${escapeHtml(label)}</option>`)
    .join("");

  const bulkBar = state.selectedIds.size ? `
    <div class="bulk-bar">
      <span class="bulk-bar__count">Выбрано: <strong>${state.selectedIds.size}</strong></span>

      <label class="bulk-bar__field">
        <span>Куда индексировать:</span>
        <select id="bulk-group" ${state.bulkBusy ? "disabled" : ""}>${groupOptions}</select>
      </label>

      <label class="bulk-bar__check" title="Удалит ВСЕ остальные документы выбранного типа">
        <input type="checkbox" id="bulk-recreate" ${state.bulkRecreate ? "checked" : ""} ${state.bulkBusy ? "disabled" : ""} />
        <span>Очистить индекс</span>
      </label>

      <label class="bulk-bar__check" title="По умолчанию удаляет старые чанки источника перед загрузкой. Сними чтобы новые чанки добавились РЯДОМ со старыми (append-режим — будут дубли для статей, которые есть в обеих версиях).">
        <input type="checkbox" id="bulk-delete-existing" ${state.bulkDeleteExisting ? "checked" : ""} ${state.bulkBusy || state.bulkRecreate ? "disabled" : ""} />
        <span>Очистить все данные источника</span>
      </label>

      <div class="bulk-bar__actions">
        <button type="button" class="btn" id="bulk-clear" ${state.bulkBusy ? "disabled" : ""}>Снять выделение</button>
        <button type="button" class="btn btn--danger" id="bulk-delete" ${state.bulkBusy ? "disabled" : ""}>Удалить</button>
        <button type="button" class="btn btn--primary" id="bulk-index" ${state.bulkBusy ? "disabled" : ""}>Индексировать</button>
      </div>
    </div>
  ` : "";

  return `
    <section class="admin-section">
      <header class="admin-section__head">
        <div>
          <h1>Загруженные НПА</h1>
          <p class="muted">${state.sources.length} источник(ов). Отметьте нужные галочками и выберите действие.</p>
        </div>
        <button type="button" class="icon-button" id="sources-refresh" title="Обновить">${icon("Refresh")}</button>
      </header>

      <div class="source-list-head">
        <label class="source-row__check">
          <input type="checkbox" id="select-all"
            ${allSelected ? "checked" : ""}
            ${someSelected ? "data-indeterminate=true" : ""}
            ${selectable.length === 0 ? "disabled" : ""} />
        </label>
        <span class="source-list-head__label">${allSelected ? "Снять выделение" : "Выбрать все"}</span>
      </div>

      <ul class="source-list">${rows}</ul>

      ${bulkBar}
    </section>
  `;
}

// Reverse-map a collection name back to a group label for the meta line.
// Stored as `npa_general` / `npa_primal` / `npa_secondary` by default.
function _groupFromCollection(collection) {
  if (!collection) return "";
  if (collection.endsWith("general"))   return "general";
  if (collection.endsWith("primal"))    return "primal";
  if (collection.endsWith("secondary")) return "secondary";
  return "";
}

function renderPlansSection() {
  if (state.plansLoading) {
    return `<section class="admin-section"><h2>Тарифы</h2><p class="muted">Загрузка...</p></section>`;
  }
  const plans = state.plans;
  return `
    <section class="admin-section admin-plans-section">
      <header class="admin-section__head admin-plans__header">
        <div>
          <h1>Тарифы</h1>
          <p class="muted">Перетащите карточку, чтобы поменять порядок. Изменения сохраняются автоматически.</p>
        </div>
        <div class="admin-plans__controls">
          <button type="button" class="btn btn--ghost" id="plans-preview" title="Посмотреть, как тарифы видит пользователь">${icon("Eye")}<span>Как видит пользователь</span></button>
          <button type="button" class="icon-button" id="plans-refresh" title="Обновить">${icon("Refresh")}</button>
        </div>
      </header>
      ${state.plansError ? `<div class="admin-error">${escapeHtml(state.plansError)}</div>` : ""}
      <div class="admin-plan-grid" id="admin-plan-grid">
        ${plans.map((p) => renderPlanCardAdmin(p)).join("")}
      </div>
      <button type="button" class="admin-plan-add" id="plans-add" title="Добавить тариф">+ Добавить тариф</button>
    </section>
    ${state.plansPreviewOpen ? renderUserPreviewModal(plans) : ""}
  `;
}

function renderPlanCardAdmin(plan) {
  const dirty = Boolean(state.plansDirty[plan.code]);
  const saving = state.plansSavingCode === plan.code;
  const title = planDraftValue(plan, "title");
  const price = planDraftValue(plan, "price_rub");
  const duration = planDraftValue(plan, "duration_days");
  const limit = planDraftValue(plan, "monthly_generation_limit");
  const allow = planDraftValue(plan, "allow_edit");
  const isActive = planDraftValue(plan, "is_active");
  const description = planDraftValue(plan, "description_md") || "";
  const isFree = plan.code === "free";
  return `
    <article class="admin-plan-card" data-plan="${escapeHtml(plan.code)}">
      <header class="admin-plan-card__head">
        <div class="admin-plan-card__code">
          <code>${escapeHtml(plan.code)}</code>
          ${isFree ? `<span class="admin-plan-card__badge">FREE</span>` : ""}
        </div>
        <span class="admin-plan-card__drag-handle" title="Перетащить для изменения порядка">${icon("GripVertical")}</span>
      </header>
      <label class="admin-plan-card__field">Название
        <input type="text" data-field="title" value="${escapeHtml(title || "")}" />
      </label>
      <label class="admin-plan-card__field">Цена (₽)
        <input type="number" min="0" step="0.01" data-field="price_rub" value="${price ?? 0}" ${isFree ? "disabled" : ""} />
      </label>
      <label class="admin-plan-card__field">Длительность (дней; пусто = бессрочно)
        <input type="number" min="0" data-field="duration_days" value="${duration ?? ""}" ${isFree ? "disabled" : ""} />
      </label>
      <label class="admin-plan-card__field">Лимит генераций / мес (пусто = ∞)
        <input type="number" min="0" data-field="monthly_generation_limit" value="${limit ?? ""}" />
      </label>
      <label class="admin-plan-card__check">
        <input type="checkbox" data-field="allow_edit" ${allow ? "checked" : ""} />
        Правка договора
      </label>
      <label class="admin-plan-card__check">
        <input type="checkbox" data-field="is_active" ${isActive ? "checked" : ""} />
        Виден пользователям
      </label>
      <label class="admin-plan-card__field admin-plan-card__field--grow">Описание (Markdown)
        <textarea data-field="description_md" rows="6" placeholder="- Пункт 1&#10;- **Жирный** пункт 2">${escapeHtml(description)}</textarea>
      </label>
      <div class="admin-plan-card__actions">
        <button type="button" class="btn btn--primary" data-save="${escapeHtml(plan.code)}" ${dirty && !saving ? "" : "disabled"}>
          ${saving ? "Сохраняю..." : "Сохранить"}
        </button>
        ${dirty ? `<button type="button" class="btn btn--ghost" data-reset="${escapeHtml(plan.code)}">Отменить</button>` : ""}
        ${isFree ? "" : `<button type="button" class="btn btn--danger" data-delete="${escapeHtml(plan.code)}">Удалить</button>`}
      </div>
    </article>
  `;
}

function renderUserPreviewModal(plans) {
  // Re-uses .plan-grid / .plan-card from styles.css (the very same classes
  // app.js renders for /billing/plans), so this is literally what the user
  // sees, sandboxed inside an overlay.
  const visible = (plans || []).filter((p) => p.is_active !== false);
  return `
    <div class="admin-modal" id="plans-preview-modal" role="dialog" aria-modal="true">
      <div class="admin-modal__backdrop" data-dismiss-modal></div>
      <div class="admin-modal__panel admin-modal__panel--wide">
        <header class="admin-modal__head">
          <h3>Превью пользовательского экрана</h3>
          <button type="button" class="icon-button" data-dismiss-modal title="Закрыть">${icon("X")}</button>
        </header>
        <div class="admin-modal__body">
          <div class="plans-screen plans-screen--in-modal">
            <header class="plans-header">
              <h1>Тарифы PactumAI</h1>
              <p>Выберите подписку, которая подходит вам. Платежи проводятся через ЮKassa.</p>
            </header>
            <section class="plan-grid">
              ${visible.map((p) => renderUserPlanCard(p)).join("")}
            </section>
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderUserPlanCard(plan) {
  const isFree = plan.code === "free";
  const description = wrapMarkdown(plan.description_md || "");
  const priceLabel = isFree ? "Бесплатно" : `${Math.round(plan.price_rub).toLocaleString("ru-RU")} ₽`;
  const periodLabel = isFree ? ""
    : plan.duration_days === 30 ? "/ месяц"
    : plan.duration_days === 365 ? "/ год"
    : plan.duration_days === 730 ? "/ 2 года"
    : plan.duration_days ? `/ ${plan.duration_days} дней` : "";
  const cta = isFree ? "Включён по умолчанию" : "Купить";
  return `
    <article class="plan-card">
      <h2>${escapeHtml(plan.title)}</h2>
      <div class="plan-card__price">${escapeHtml(priceLabel)}<span>${escapeHtml(periodLabel)}</span></div>
      <div class="plan-card__description">${description}</div>
      <button class="plan-card__cta" type="button" disabled>${escapeHtml(cta)}</button>
    </article>
  `;
}

function wrapMarkdown(text) {
  if (window.marked?.parse && window.DOMPurify?.sanitize) {
    window.marked.setOptions({ breaks: true, gfm: true });
    return window.DOMPurify.sanitize(window.marked.parse(text || ""));
  }
  return `<p>${escapeHtml(text || "")}</p>`;
}

async function deletePlan(code) {
  if (!(await customConfirm(
    `Удалить тариф "${code}"?\nДействие необратимо.`,
    { title: "Удаление тарифа", okLabel: "Удалить", danger: true }
  ))) return;
  setState({ plansError: "" });
  try {
    await api(`/api/admin/plans/${encodeURIComponent(code)}`, { method: "DELETE" });
    await loadPlans();
  } catch (err) {
    if (err instanceof AdminUnauthorized) {
      goToLogin();
      return;
    }
    setState({ plansError: err.message || String(err) });
  }
}

async function openCreatePlanDialog() {
  const code = (prompt("Код нового тарифа (латиница, цифры, _ или -):") || "").trim().toLowerCase();
  if (!code) return;
  const title = (prompt("Название (что увидит пользователь):") || "").trim();
  if (!title) return;
  const priceRaw = prompt("Цена в рублях:", "990");
  const durationRaw = prompt("Длительность в днях (например 30, 365):", "30");
  setState({ plansError: "" });
  try {
    await api("/api/admin/plans", {
      method: "POST",
      body: JSON.stringify({
        code,
        title,
        price_rub: Number(priceRaw || 0),
        duration_days: Number(durationRaw || 30),
        monthly_generation_limit: null,
        allow_edit: true,
        description_md: "",
        is_active: true,
      }),
    });
    await loadPlans();
  } catch (err) {
    if (err instanceof AdminUnauthorized) {
      goToLogin();
      return;
    }
    setState({ plansError: err.message || String(err) });
  }
}

function bindPlansSection(root) {
  root.querySelector("#plans-refresh")?.addEventListener("click", loadPlans);
  root.querySelector("#plans-add")?.addEventListener("click", openCreatePlanDialog);
  root.querySelector("#plans-preview")?.addEventListener("click", () => {
    setState({ plansPreviewOpen: true });
  });
  // Modal close handlers (backdrop + X button + Esc)
  root.querySelectorAll('[data-dismiss-modal]').forEach((el) => {
    el.addEventListener("click", () => setState({ plansPreviewOpen: false }));
  });
  if (state.plansPreviewOpen && !root._previewEscBound) {
    const onKey = (e) => {
      if (e.key === "Escape") {
        setState({ plansPreviewOpen: false });
        document.removeEventListener("keydown", onKey);
      }
    };
    document.addEventListener("keydown", onKey);
    root._previewEscBound = true;
  }

  // Drag-and-drop reorder via SortableJS (loaded from CDN in admin.html).
  // We persist the new order on `onEnd` — one HTTP roundtrip per drag.
  const grid = root.querySelector("#admin-plan-grid");
  if (grid && window.Sortable) {
    window.Sortable.create(grid, {
      animation: 150,
      handle: ".admin-plan-card__drag-handle",
      ghostClass: "admin-plan-card--ghost",
      dragClass: "admin-plan-card--drag",
      onEnd: () => {
        const order = Array.from(grid.querySelectorAll(".admin-plan-card[data-plan]"))
          .map((el) => el.getAttribute("data-plan"));
        persistPlanOrder(order);
      },
    });
  }

  root.querySelectorAll(".admin-plan-card").forEach((cardEl) => {
    const code = cardEl.getAttribute("data-plan");
    cardEl.querySelectorAll("[data-field]").forEach((input) => {
      const field = input.getAttribute("data-field");
      input.addEventListener("input", () => {
        if (field === "allow_edit" || field === "is_active") {
          patchPlanDraft(code, { [field]: input.checked });
        } else if (field === "price_rub") {
          patchPlanDraft(code, { price_rub: input.value === "" ? null : Number(input.value) });
        } else if (field === "duration_days") {
          patchPlanDraft(code, { duration_days: input.value === "" ? null : Number(input.value) });
        } else if (field === "monthly_generation_limit") {
          if (input.value === "") {
            patchPlanDraft(code, { clear_limit: true, monthly_generation_limit: null });
          } else {
            patchPlanDraft(code, { clear_limit: false, monthly_generation_limit: Number(input.value) });
          }
        } else {
          patchPlanDraft(code, { [field]: input.value });
        }
      });
    });
    cardEl.querySelector(`[data-save="${code}"]`)?.addEventListener("click", () => savePlan(code));
    cardEl.querySelector(`[data-reset="${code}"]`)?.addEventListener("click", () => {
      const next = { ...state.plansDirty };
      delete next[code];
      setState({ plansDirty: next });
    });
    cardEl.querySelector("[data-delete]")?.addEventListener("click", () => deletePlan(code));
  });
}

async function persistPlanOrder(order) {
  // Persist after a drag — server returns the updated list and we replace
  // the in-memory copy so the next render uses the canonical order.
  setState({ plansError: "" });
  try {
    const updated = await api("/api/admin/plans/order", {
      method: "POST",
      body: JSON.stringify({ order }),
    });
    state.plans = updated; // already in correct order; no need to re-render
  } catch (err) {
    if (err instanceof AdminUnauthorized) {
      goToLogin();
      return;
    }
    setState({ plansError: err.message || String(err) });
  }
}

function renderAdminsSection() {
  const rows = state.adminUsers.map((a) => `
    <li class="source-row">
      <div class="source-row__main">
        <div class="source-row__top">
          <h3 class="source-row__title">${escapeHtml(a.username)}</h3>
        </div>
        <div class="source-row__meta">
          <span>Создан ${fmtDate(a.created_at)}</span>
        </div>
      </div>
      <div class="source-row__actions">
        <button type="button" class="btn btn--small btn--danger" data-admin-delete="${escapeHtml(a.id)}" data-admin-name="${escapeHtml(a.username)}">
          Удалить
        </button>
      </div>
    </li>
  `).join("");

  const createdBlock = state.newAdminCreated ? `
    <div class="admin-card" style="border:2px solid var(--color-success,#16a34a);padding:1rem;margin-bottom:1rem;">
      <p><strong>Администратор создан: ${escapeHtml(state.newAdminCreated.username)}</strong></p>
      <p style="margin:0.5rem 0">Пароль (показывается один раз):</p>
      <code style="display:block;padding:0.5rem;background:var(--color-bg-code,#f3f4f6);border-radius:4px;word-break:break-all;font-size:1rem">${escapeHtml(state.newAdminCreated.password)}</code>
      <div style="display:flex;gap:0.5rem;margin-top:0.75rem">
        <button type="button" class="btn btn--primary" id="download-creds">Скачать файл с паролем</button>
        <button type="button" class="btn btn--ghost" id="dismiss-creds">Закрыть</button>
      </div>
    </div>
  ` : "";

  return `
    <section class="admin-section">
      <header class="admin-section__head">
        <div>
          <h1>Администраторы</h1>
          <p class="muted">Только владелец может управлять администраторами.</p>
        </div>
        <button type="button" class="icon-button" id="admins-refresh" title="Обновить">${icon("Refresh")}</button>
      </header>
      ${state.adminUsersError ? `<div class="admin-error">${escapeHtml(state.adminUsersError)}</div>` : ""}
      ${createdBlock}
      <div class="admin-card" style="padding:1rem;margin-bottom:1rem">
        <h3 style="margin:0 0 0.75rem">Добавить администратора</h3>
        ${state.newAdminError ? `<div class="admin-error" style="margin-bottom:0.5rem">${escapeHtml(state.newAdminError)}</div>` : ""}
        <div style="display:flex;gap:0.5rem;align-items:flex-end">
          <div style="flex:1">
            <label for="new-admin-username" class="form-label">Логин</label>
            <input id="new-admin-username" type="text" class="form-input" placeholder="username" value="${escapeHtml(state.newAdminUsername)}" ${state.newAdminBusy ? "disabled" : ""} />
          </div>
          <button type="button" class="btn btn--primary" id="create-admin-btn" ${state.newAdminBusy || !state.newAdminUsername.trim() ? "disabled" : ""}>
            ${state.newAdminBusy ? "Создаю..." : "Создать"}
          </button>
        </div>
      </div>
      ${state.adminUsersLoading
        ? `<p class="muted">Загрузка...</p>`
        : state.adminUsers.length
          ? `<ul class="source-list">${rows}</ul>`
          : `<div class="admin-card admin-card--empty"><div class="empty-state"><p>Администраторов нет.</p></div></div>`
      }
    </section>
  `;
}

function bindAdminsSection(root) {
  root.querySelector("#admins-refresh")?.addEventListener("click", loadAdminUsers);
  const input = root.querySelector("#new-admin-username");
  if (input) {
    input.addEventListener("input", (e) => {
      state.newAdminUsername = e.target.value;
      const btn = root.querySelector("#create-admin-btn");
      if (btn) btn.disabled = state.newAdminBusy || !state.newAdminUsername.trim();
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !state.newAdminBusy && state.newAdminUsername.trim()) createAdminUser();
    });
  }
  root.querySelector("#create-admin-btn")?.addEventListener("click", createAdminUser);
  root.querySelector("#download-creds")?.addEventListener("click", () => {
    if (state.newAdminCreated) downloadPasswordFile(state.newAdminCreated.username, state.newAdminCreated.password);
  });
  root.querySelector("#dismiss-creds")?.addEventListener("click", () => setState({ newAdminCreated: null }));
  root.querySelectorAll("[data-admin-delete]").forEach((btn) => {
    btn.addEventListener("click", () => deleteAdminUser(btn.getAttribute("data-admin-delete"), btn.getAttribute("data-admin-name")));
  });
}

function renderDealTypesSection() {
  if (state.dealTypesLoading && state.dealTypes === null) {
    return `<section class="admin-section"><h1>Типы договоров</h1><p class="muted">Загрузка...</p></section>`;
  }
  const count = state.dealTypes ? Object.keys(state.dealTypes).length : 0;
  return `
    <section class="admin-section admin-deal-types-section">
      <header class="admin-section__head">
        <div>
          <h1>Типы договоров</h1>
          <p class="muted">${count} тип(ов) в каталоге. Файл хранится в S3 (admin-npa/config/supported_deal_types.json) и применяется графом в течение нескольких секунд после сохранения.</p>
        </div>
        <button type="button" class="icon-button" id="deal-types-refresh" title="Загрузить актуальный JSON">${icon("Refresh")}</button>
      </header>
      ${state.dealTypesError ? `<div class="admin-error">${escapeHtml(state.dealTypesError)}</div>` : ""}
      <textarea id="deal-types-editor" class="admin-deal-types__editor" spellcheck="false">${escapeHtml(state.dealTypesText)}</textarea>
      <div class="admin-deal-types__actions">
        <button type="button" class="btn btn--primary" id="deal-types-save" ${(!state.dealTypesDirty || state.dealTypesSaving) ? "disabled" : ""}>
          ${state.dealTypesSaving ? "Сохраняю..." : "Сохранить"}
        </button>
        <button type="button" class="btn btn--ghost" id="deal-types-reset" ${!state.dealTypesDirty ? "disabled" : ""}>
          Отменить изменения
        </button>
      </div>
    </section>
  `;
}

function bindDealTypesSection(root) {
  root.querySelector("#deal-types-refresh")?.addEventListener("click", loadDealTypes);
  root.querySelector("#deal-types-save")?.addEventListener("click", saveDealTypes);
  root.querySelector("#deal-types-reset")?.addEventListener("click", () => {
    if (state.dealTypes) {
      setState({
        dealTypesText: JSON.stringify(state.dealTypes, null, 2),
        dealTypesDirty: false,
        dealTypesError: "",
      });
    }
  });
  const editor = root.querySelector("#deal-types-editor");
  if (editor) {
    editor.addEventListener("input", () => {
      // Mutate in place to preserve cursor position; flag dirty surgically.
      state.dealTypesText = editor.value;
      const wasDirty = state.dealTypesDirty;
      state.dealTypesDirty = true;
      if (!wasDirty) {
        const saveBtn = root.querySelector("#deal-types-save");
        const resetBtn = root.querySelector("#deal-types-reset");
        if (saveBtn) saveBtn.disabled = state.dealTypesSaving;
        if (resetBtn) resetBtn.disabled = false;
      }
    });
  }
}

function bindSourcesSection(root) {
  root.querySelector("#sources-refresh")?.addEventListener("click", loadSources);
  root.querySelector("#select-all")?.addEventListener("change", (e) => selectAll(e.target.checked));
  // make indeterminate checkboxes actually indeterminate (HTML attr doesn't do it)
  const selectAllEl = root.querySelector("#select-all");
  if (selectAllEl?.dataset.indeterminate === "true") selectAllEl.indeterminate = true;
  root.querySelectorAll('[data-action="toggle"]').forEach((cb) => {
    cb.addEventListener("change", () => toggleSelected(cb.getAttribute("data-id")));
  });
  root.querySelector("#bulk-clear")?.addEventListener("click", () => { clearSelection(); render(); });
  root.querySelector("#bulk-delete")?.addEventListener("click", bulkDelete);
  root.querySelector("#bulk-index")?.addEventListener("click", startBulkIndex);
  root.querySelector("#bulk-group")?.addEventListener("change", (e) => { state.bulkSourceGroup = e.target.value; });
  root.querySelector("#bulk-recreate")?.addEventListener("change", (e) => { state.bulkRecreate = e.target.checked; render(); });
  root.querySelector("#bulk-delete-existing")?.addEventListener("change", (e) => { state.bulkDeleteExisting = e.target.checked; render(); });
}

// ---- Shell + sidebar ----

async function logout() {
  try { await api("/api/admin/logout", { method: "POST" }); } catch {}
  goToLogin();
}

function renderShell() {
  const sectionMarkup = state.section === "upload"
    ? renderUploadSection()
    : state.section === "plans"
      ? renderPlansSection()
      : state.section === "deal_types"
        ? renderDealTypesSection()
        : state.section === "admins"
          ? renderAdminsSection()
          : renderSourcesSection();

  const inProgressCount = state.sources.filter((s) => s.status === "chunking" || s.status === "indexing").length;
  const failedCount = state.sources.filter((s) => s.status === "failed").length;

  return `
    <div class="admin-shell">
      <aside class="admin-sidebar">
        <div class="sidebar__top">
          <div class="brand">
            <svg class="lucide-icon" viewBox="0 0 24 24" aria-hidden="true">${iconPaths.Scale}</svg>
            <div class="brand__text">
              <strong>PactumAI</strong>
              <span>Админ-панель</span>
            </div>
          </div>
        </div>

        <nav class="admin-nav">
          <button type="button" class="nav-item ${state.section === "upload" ? "nav-item--active" : ""}" data-section="upload">
            ${icon("Upload")}<span>Загрузка</span>
          </button>
          <button type="button" class="nav-item ${state.section === "sources" ? "nav-item--active" : ""}" data-section="sources">
            ${icon("List")}<span>Источники</span>
            ${state.sources.length ? `<span class="nav-item__badge">${state.sources.length}</span>` : ""}
          </button>
          <button type="button" class="nav-item ${state.section === "plans" ? "nav-item--active" : ""}" data-section="plans">
            ${icon("DollarSign")}<span>Тарифы</span>
          </button>
          <button type="button" class="nav-item ${state.section === "deal_types" ? "nav-item--active" : ""}" data-section="deal_types">
            ${icon("FileText")}<span>Типы договоров</span>
          </button>
          ${state.admin?.is_owner ? `
          <button type="button" class="nav-item ${state.section === "admins" ? "nav-item--active" : ""}" data-section="admins">
            ${icon("List")}<span>Администраторы</span>
          </button>` : ""}
        </nav>

        ${(inProgressCount || failedCount) ? `
          <div class="sidebar__status">
            ${inProgressCount ? `<div class="sidebar__status-row"><span class="status-dot status-dot--processing"></span>В работе: ${inProgressCount}</div>` : ""}
            ${failedCount ? `<div class="sidebar__status-row"><span class="status-dot status-dot--danger"></span>С ошибками: ${failedCount}</div>` : ""}
          </div>` : ""}

        <div class="sidebar__bottom">
          <div class="user-badge">
            <div class="user-badge__info">
              <div class="user-badge__name">${escapeHtml(state.admin?.username || "")}</div>
              <div class="user-badge__email muted">${state.admin?.is_owner ? "владелец" : "администратор"}</div>
            </div>
            <div class="user-badge__actions">
              <button type="button" class="icon-button theme-toggle" id="admin-theme-toggle" aria-label="Переключить тему">
                ${state.theme === "dark" ? icon("Moon") : icon("Sun")}
              </button>
              <button type="button" class="icon-button" id="admin-logout" title="Выйти">${icon("Logout")}</button>
            </div>
          </div>
        </div>
      </aside>

      <main class="admin-main">
        ${sectionMarkup}
      </main>
    </div>
  `;
}

function bindShell(root) {
  root.querySelectorAll('[data-section]').forEach((btn) => {
    btn.addEventListener("click", () => {
      const next = btn.getAttribute("data-section");
      setState({ section: next });
      if (next === "plans" && state.plans.length === 0) loadPlans();
      if (next === "deal_types" && state.dealTypes === null) loadDealTypes();
      if (next === "admins" && state.adminUsers.length === 0 && !state.adminUsersLoading) loadAdminUsers();
    });
  });
  root.querySelector("#admin-logout")?.addEventListener("click", logout);
  root.querySelector("#admin-theme-toggle")?.addEventListener("click", toggleTheme);
  if (state.section === "upload") bindUploadSection(root);
  if (state.section === "sources") bindSourcesSection(root);
  if (state.section === "plans") bindPlansSection(root);
  if (state.section === "deal_types") bindDealTypesSection(root);
  if (state.section === "admins") bindAdminsSection(root);
}

// ---- Render ----

function render() {
  const root = document.getElementById("admin");
  if (!root) return;
  if (state.view === "loading") {
    root.innerHTML = `<div class="admin-loading">Загрузка…</div>`;
    return;
  }
  if (state.view === "login") {
    root.innerHTML = renderLogin();
    bindLogin(root);
    return;
  }
  root.innerHTML = renderShell();
  bindShell(root);
}

document.addEventListener("DOMContentLoaded", () => {
  applyTheme();
  render();
  bootstrap();
});
