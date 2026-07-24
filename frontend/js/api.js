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
  scenarios: () => req("GET", "/api/scenarios"),
  trafficClasses: () => req("GET", "/api/traffic-classes"),
  checkpoints: () => req("GET", "/api/checkpoints"),
  start: (cfg) => req("POST", "/api/simulation/start", cfg),
  pause: () => req("POST", "/api/simulation/pause"),
  resume: () => req("POST", "/api/simulation/resume"),
  step: () => req("POST", "/api/simulation/step"),
  reset: () => req("POST", "/api/simulation/reset"),
  speed: (speed) => req("POST", "/api/simulation/speed", { speed }),
  status: () => req("GET", "/api/simulation/status"),
  failLink: (link) => req("POST", "/api/failure/inject", { link }),
  recoverLink: (link) => req("POST", "/api/failure/recover", { link }),
  burst: (demand, factor, duration_min) =>
    req("POST", "/api/traffic/burst", { demand, factor, duration_min }),
  multiplier: (factor) => req("POST", "/api/traffic/multiplier", { factor }),
  metricsHistory: () => req("GET", "/api/metrics/history"),
  saveRun: () => req("POST", "/api/export/save-run"),
  runs: () => req("GET", "/api/runs"),
  trainStart: (cfg) => req("POST", "/api/agent/train", cfg),
  trainProgress: () => req("GET", "/api/training/progress"),
};

export function toast(msg, isError = false) {
  // minimal, non-blocking notice in the decision tape area
  const box = document.getElementById("tape-lines");
  const div = document.createElement("div");
  div.className = "tape-line " + (isError ? "crit" : "ok");
  div.innerHTML = `<span class="t">ui</span><span class="msg"></span>`;
  div.querySelector(".msg").textContent = msg;
  box.prepend(div);
}
