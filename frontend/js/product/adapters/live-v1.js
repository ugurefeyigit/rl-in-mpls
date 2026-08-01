/* The live V1 session adapter.
 *
 * It owns the WebSocket, the REST reads and the execution controls. It is the
 * only adapter allowed to execute a policy or to hand a component link-level
 * telemetry, because it is the only one whose source has either.
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

async function get(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const error = new Error(detail?.detail?.message || detail?.detail || response.statusText);
    error.status = response.status;
    error.detail = detail?.detail;
    throw error;
  }
  return response.json();
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST", headers: JSON_HEADERS,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const error = new Error(detail?.detail?.message || detail?.detail || response.statusText);
    error.status = response.status;
    error.detail = detail?.detail;
    throw error;
  }
  return response.json();
}

async function put(path, body) {
  const response = await fetch(path, {
    method: "PUT", headers: JSON_HEADERS, body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw Object.assign(new Error(detail?.detail || response.statusText),
      { status: response.status, detail: detail?.detail });
  }
  return response.json();
}

async function remove(path) {
  const response = await fetch(path, { method: "DELETE" });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw Object.assign(new Error(detail?.detail || response.statusText),
      { status: response.status, detail: detail?.detail });
  }
  return response.json();
}

export const kind = "live_session";

/** Match the status schema returned by /api/simulation/status. */
export function hasActiveSession(status) {
  return Boolean(status?.session_id);
}

export const api = {
  capabilities: () => get("/api/product/capabilities"),
  contracts: () => get("/api/product/contracts"),
  displayMap: () => get("/api/product/display-map"),
  schema: (environment) => get(`/api/rl/schema?environment=${environment}`),
  scenarios: () => get("/api/scenarios"),
  status: () => get("/api/simulation/status"),

  snapshot: (algorithm) =>
    get(algorithm ? `/api/simulation/snapshot?algorithm=${algorithm}` : "/api/simulation/snapshot"),
  moment: (algorithm) =>
    get(algorithm ? `/api/simulation/moment?algorithm=${algorithm}` : "/api/simulation/moment"),
  decision: (algorithm) =>
    get(algorithm ? `/api/simulation/decision?algorithm=${algorithm}` : "/api/simulation/decision"),
  timeline: () => get("/api/simulation/timeline"),
  comparison: () => get("/api/simulation/comparison"),
  object: (objectType, objectId) =>
    get(`/api/simulation/object/${objectType}/${encodeURIComponent(objectId)}`),

  start: (config) => post("/api/simulation/start", config),
  pause: () => post("/api/simulation/pause"),
  resume: () => post("/api/simulation/resume"),
  step: () => post("/api/simulation/step"),
  reset: () => post("/api/simulation/reset"),
  stop: () => post("/api/simulation/stop"),
  retainedRuns: () => get("/api/simulation/retained-runs"),
  results: () => get("/api/product/results"),
  comparativeRuns: () => get("/api/product/comparative-runs"),
  assignComparativeRun: (slot, runId) =>
    put(`/api/product/comparative-runs/${slot}`, { run_id: runId }),
  clearComparativeRun: (slot) => remove(`/api/product/comparative-runs/${slot}`),
  clearComparativeRuns: () => remove("/api/product/comparative-runs"),
  swapComparativeRuns: () => post("/api/product/comparative-runs/swap"),
  speed: (speed) => post("/api/simulation/speed", { speed }),
  // `delegate` is required by advisor execution: a fast-forward applies the
  // controller's own actions for a stretch without individual approval, and the
  // server refuses to do that unless the caller says so.
  runUntil: (condition, maxSteps = 300, delegate = false) =>
    post("/api/simulation/run-until",
         { condition, max_steps: maxSteps, delegate }),
  saveRun: () => post("/api/export/save-run"),

  injectFailure: (link) => post("/api/failure/inject", { link }),
  recoverLink: (link) => post("/api/failure/recover", { link }),

  propose: () => post("/api/advisor/propose"),
  approve: () => post("/api/advisor/approve"),
  reject: () => post("/api/advisor/reject"),
  advisorStatus: () => get("/api/advisor/status"),

  counterfactual: (payload) => post("/api/simulation/counterfactual", payload),
};

/**
 * A reconnecting telemetry socket. While the connection is down the displayed
 * values freeze and are labelled as last received: showing a stale number as
 * the present is the failure this reconnect logic exists to prevent.
 */
export function connect({ onPayload, onState }) {
  let socket = null;
  let attempt = 0;
  let closed = false;
  let timer = null;

  function open() {
    if (closed) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${protocol}://${window.location.host}/ws/telemetry`);

    socket.addEventListener("open", () => { attempt = 0; onState("open"); });
    socket.addEventListener("message", (event) => {
      try { onPayload(JSON.parse(event.data)); } catch { /* ignore a malformed frame */ }
    });
    socket.addEventListener("close", () => {
      if (closed) return;
      onState("lost");
      const delay = Math.min(500 * 2 ** attempt, 8000);
      attempt += 1;
      timer = window.setTimeout(open, delay);
    });
    socket.addEventListener("error", () => socket && socket.close());
  }

  open();
  return {
    close() {
      closed = true;
      if (timer) window.clearTimeout(timer);
      if (socket) socket.close();
    },
  };
}
