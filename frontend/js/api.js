// Thin REST client. All endpoints documented in docs/API.md and /docs (OpenAPI).

async function req(method, url, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch { /* keep statusText */ }
    throw new Error(detail);
  }
  return r.json();
}

export const api = {
  topology: () => req("GET", "/api/topology"),
  display: () => req("GET", "/api/display"),
  scenarios: () => req("GET", "/api/scenarios"),
  trafficClasses: () => req("GET", "/api/traffic-classes"),
  checkpoints: () => req("GET", "/api/checkpoints"),
  benchmark: () => req("GET", "/api/benchmark"),
  events: (limit = 60) => req("GET", `/api/events?limit=${limit}`),
  start: (cfg) => req("POST", "/api/simulation/start", cfg),
  pause: () => req("POST", "/api/simulation/pause"),
  resume: () => req("POST", "/api/simulation/resume"),
  step: () => req("POST", "/api/simulation/step"),
  reset: () => req("POST", "/api/simulation/reset"),
  runUntil: (condition, max_steps = 300, util_threshold = 0.9) =>
    req("POST", "/api/simulation/run-until", { condition, max_steps, util_threshold }),
  speed: (speed) => req("POST", "/api/simulation/speed", { speed }),
  status: () => req("GET", "/api/simulation/status"),
  telemetry: () => req("GET", "/api/telemetry/current"),
  failLink: (link) => req("POST", "/api/failure/inject", { link }),
  recoverLink: (link) => req("POST", "/api/failure/recover", { link }),
  burst: (demand, factor, duration_min) =>
    req("POST", "/api/traffic/burst", { demand, factor, duration_min }),
  multiplier: (factor) => req("POST", "/api/traffic/multiplier", { factor }),
  advisorPropose: () => req("POST", "/api/advisor/propose"),
  advisorApprove: () => req("POST", "/api/advisor/approve"),
  advisorReject: () => req("POST", "/api/advisor/reject"),
  advisorStatus: () => req("GET", "/api/advisor/status"),
  metricsHistory: () => req("GET", "/api/metrics/history"),
  saveRun: () => req("POST", "/api/export/save-run"),
  runs: () => req("GET", "/api/runs"),
  trainStart: (cfg) => req("POST", "/api/agent/train", cfg),
  trainProgress: () => req("GET", "/api/training/progress"),
};

// ------------------------------------------------------------------- toasts
// A floating stack, so a failed intervention or a 409 is impossible to miss.
// Works on both pages; if the host page also has a decision tape, the notice
// is mirrored there so the tape stays a complete record of the session.
function toastHost() {
  let host = document.getElementById("toast-host");
  if (!host) {
    host = document.createElement("div");
    host.id = "toast-host";
    document.body.appendChild(host);
  }
  return host;
}

export function toast(msg, isError = false) {
  const el = document.createElement("div");
  el.className = "toast" + (isError ? " toast-error" : "");
  el.setAttribute("role", isError ? "alert" : "status");
  el.textContent = msg;
  toastHost().appendChild(el);
  setTimeout(() => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 300);
  }, isError ? 7000 : 4000);

  const tape = document.getElementById("tape-lines");
  if (tape) {
    const line = document.createElement("div");
    line.className = "tape-line " + (isError ? "crit" : "ok");
    line.innerHTML = `<span class="t">ui</span><span class="msg"></span>`;
    line.querySelector(".msg").textContent = msg;
    tape.prepend(line);
    while (tape.children.length > 120) tape.removeChild(tape.lastChild);
  }
}
