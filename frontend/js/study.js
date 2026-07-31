/* ===========================================================================
   V2 study — sealed evidence record.

   Reads the read-only /api/v2 evidence surface and renders it. This module holds
   no scientific number of its own: if the API cannot serve a section, the section
   says so rather than showing a placeholder value.

   Two rules mirrored from the Python evidence layer:
     - development and final-holdout evidence render in separate regions and are
       never combined into one figure or one table;
     - where two published statistics share a name (no-op share, wall time), both
       are shown with their grain stated.
   =========================================================================== */

const SERIES = {
  masked_bandit: { label: "Masked contextual bandit", short: "Bandit", tok: "B", cls: "bandit" },
  maskable_ppo:  { label: "MaskablePPO",              short: "PPO",    tok: "P", cls: "ppo" },
  greedy:        { label: "Utilization-aware greedy", short: "Greedy", tok: "G", cls: "baseline" },
  cspf:          { label: "CSPF periodic reopt",      short: "CSPF",   tok: "C", cls: "baseline" },
  static:        { label: "Static shortest path",     short: "Static", tok: "S", cls: "baseline" },
};

const SCENARIO_LABEL = {
  full_day: "Full day",
  evening_peak: "Evening peak",
  flash_crowd: "Flash crowd",
  link_failure: "Link failure",
  deceptive_local_optimum: "Deceptive local optimum",
  ood_double_failure: "Double failure (out of distribution)",
  overload_stress: "Overload stress",
};

// --------------------------------------------------------------- utilities
const $ = (id) => document.getElementById(id);

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/** Fixed decimals with tabular alignment. Never rounds a sign away. */
function num(v, dp = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString("en-US",
    { minimumFractionDigits: dp, maximumFractionDigits: dp });
}
const int = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("en-US"));
/** Identifiers — seeds, training roots — are labels, not counts. No separators. */
const id = (v) => (v === null || v === undefined ? "—" : String(v));
const pct = (v, dp = 2) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(dp)}%`);
const sha = (s) => (s ? String(s).slice(0, 7) : "—");

function meta(algo) {
  return SERIES[algo] || { label: algo, short: algo, tok: "?", cls: "baseline" };
}
function token(algo) {
  const m = meta(algo);
  return `<span class="tok tok-${m.cls}" aria-hidden="true">${m.tok}</span>`;
}
function rowClass(algo) {
  return `row-${meta(algo).cls}`;
}

/** Keep an ECharts instance exactly as wide as the box CSS gave it. A chart that
 *  sizes itself from a pre-layout measurement pushes the whole page sideways. */
function bindChartSize(chart, el) {
  const fit = () => {
    if (el.clientWidth > 0) chart.resize({ width: el.clientWidth, height: el.clientHeight });
  };
  requestAnimationFrame(fit);
  // Both signals: a ResizeObserver catches layout changes the window never sees, and
  // the window listener covers engines where observer callbacks do not arrive.
  if (typeof ResizeObserver === "function") new ResizeObserver(fit).observe(el);
  window.addEventListener("resize", fit);
}

async function get(path) {
  const r = await fetch(path);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = body.detail || {};
    throw new Error(d.message || d.error || `${path} returned ${r.status}`);
  }
  return body;
}

function fail(el, err) {
  el.innerHTML = `<div class="unavailable"><strong>Not available.</strong> ${esc(err.message)}</div>`;
}

function banner(message) {
  const el = $("error-banner");
  el.textContent = message;
  el.hidden = false;
}

function table({ caption, head, rows, foot }) {
  const th = head.map((h) =>
    `<th scope="col"${h.num ? ' class="num"' : ""}>${esc(h.label)}</th>`).join("");
  const tb = rows.map((r) =>
    `<tr class="${r.cls || ""}">${r.cells.map((c, i) =>
      `<td${head[i]?.num ? ' class="num"' : ""}>${c}</td>`).join("")}</tr>`).join("");
  return `<table>${caption ? `<caption>${esc(caption)}</caption>` : ""}` +
    `<thead><tr>${th}</tr></thead><tbody>${tb}</tbody>` +
    (foot ? `<tfoot><tr><td colspan="${head.length}">${foot}</td></tr></tfoot>` : "") +
    `</table>`;
}

// ================================================================== boot
async function boot() {
  let study;
  try {
    study = await get("/api/v2/study");
  } catch (e) {
    banner(`The evidence API is unavailable: ${e.message}`);
    $("empty-state").hidden = false;
    document.body.dataset.state = "error";
    return;
  }
  renderIdentity(study);

  const sections = [
    renderVerdict(study),
    renderHoldout(),
    renderScenarios(),
    renderOperations(),
    renderDevelopment(),
    renderProvenance(),
    renderReplay(),
  ];
  await Promise.allSettled(sections);
  document.body.dataset.state = "ready";
  wireSpine();
}

// ------------------------------------------------------------- identity
function renderIdentity(study) {
  $("seal-text").textContent =
    `Study closed · ${study.environment} · one-shot holdout · sealed at ${sha(study.sources.closeout)}`;
  $("stage-badge").textContent = "Study closed";

  const s = study.sources;
  $("study-ident").innerHTML = [
    ["Environment", `${esc(study.environment)} · obs ${int(study.observation_dim)} · act ${int(study.action_count)}`],
    ["Evaluation source", esc(sha(s.evaluation))],
    ["Training sources", `${esc(sha(s.seed42_training))} · ${esc(sha(s.continuation_training))}`],
    ["Environment pin", esc(sha(s.environment_pin))],
    ["Holdout seeds", study.holdout_seeds.join(", ")],
  ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("");
}

// -------------------------------------------------------------- verdict
async function renderVerdict(study) {
  const el = $("verdict-body");
  try {
    const fh = await get("/api/v2/final-holdout");
    const c = fh.comparison;
    const ep = fh.episodes;
    const greedy = fh.aggregate.find((r) => r.algorithm === "greedy");

    const stats = `
      <div class="stat-row">
        <div class="stat stat-bandit">
          <p class="stat-label">Bandit · holdout return</p>
          <span class="stat-value">${num(c.bandit_return)}</span>
          <p class="stat-sub">mean over ${int(c.roots_total)} training roots</p>
        </div>
        <div class="stat stat-ppo">
          <p class="stat-label">PPO · holdout return</p>
          <span class="stat-value">${num(c.ppo_return)}</span>
          <p class="stat-sub">mean over ${int(c.roots_total)} training roots</p>
        </div>
        <div class="stat">
          <p class="stat-label">Advantage</p>
          <span class="stat-value">${num(c.advantage)}</span>
          <p class="stat-sub">bandit minus PPO, return points</p>
        </div>
        <div class="stat">
          <p class="stat-label">Strongest baseline</p>
          <span class="stat-value">${num(greedy.operational_return_mean)}</span>
          <p class="stat-sub">utilization-aware greedy</p>
        </div>
        <div class="stat">
          <p class="stat-label">Episodes</p>
          <span class="stat-value">${int(ep.total)}</span>
          <p class="stat-sub">${int(ep.per_policy)} per checkpoint or baseline</p>
        </div>
      </div>`;

    const findings = study.conclusions
      .map((t) => `<li>${esc(t)}</li>`).join("");

    el.innerHTML = `${stats}
      <div class="table-block">
        <h3 class="block-title">Findings as recorded at closeout</h3>
        <ol class="findings">${findings}</ol>
      </div>`;
    // findings list is prose, styled inline to stay within the record's rules
    el.querySelector(".findings").style.cssText =
      "margin:.5rem 0 0;padding-left:1.3rem;max-width:82ch;line-height:1.6;";
  } catch (e) {
    fail(el, e);
    $("empty-state").hidden = false;
  }
}

// -------------------------------------------------------- final holdout
async function renderHoldout() {
  const el = $("holdout-body");
  try {
    const fh = await get("/api/v2/final-holdout");

    const agg = table({
      caption: "Aggregate comparison. Learner rows are the mean of three training-root "
             + "means; baselines have no training root and ran once.",
      head: [
        { label: "Method" }, { label: "Return", num: true },
        { label: "Root SD", num: true }, { label: "Episode SD", num: true },
        { label: "Delivered", num: true }, { label: "SLA intervals", num: true },
        { label: "Max util", num: true }, { label: "Delay ms", num: true },
        { label: "Loss", num: true }, { label: "Roots", num: true },
        { label: "Episodes", num: true },
      ],
      rows: fh.aggregate.map((r) => ({
        cls: rowClass(r.algorithm),
        cells: [
          `${token(r.algorithm)}${esc(meta(r.algorithm).label)}`,
          `<strong>${num(r.operational_return_mean)}</strong>`,
          r.root_count > 1 ? num(r.root_mean_std) : "—",
          num(r.operational_return_std, 2),
          num(r.delivered_ratio_mean_mean, 4),
          num(r.sla_violations_demand_intervals_mean, 2),
          num(r.max_utilization_mean_mean),
          num(r.delay_ms_mean_mean, 2),
          num(r.loss_ratio_mean_mean, 4),
          int(r.root_count),
          int(r.episodes),
        ],
      })),
      foot: "Root SD is the sample standard deviation across the three training-root "
          + "means. Episode SD is pooled over every episode of the method; scenario "
          + "heterogeneity dominates it. The two are different statistics.",
    });

    const roots = table({
      caption: "Per training root. The checkpoint columns record which fixed "
             + "development-selected checkpoint was evaluated.",
      head: [
        { label: "Training root", num: true }, { label: "Bandit", num: true },
        { label: "PPO", num: true }, { label: "Advantage", num: true },
        { label: "Winner" }, { label: "Bandit checkpoint", num: true },
        { label: "PPO checkpoint", num: true },
      ],
      rows: fh.comparison.roots.map((r) => ({
        cells: [
          id(r.training_root), num(r.bandit), num(r.ppo),
          `<strong>${num(r.advantage)}</strong>`,
          `${token(r.winner)}${esc(meta(r.winner).short)}`,
          int(r.bandit_checkpoint), int(r.ppo_checkpoint),
        ],
      })),
      foot: `Bandit won ${fh.comparison.roots_won} of ${fh.comparison.roots_total} training roots.`,
    });

    el.innerHTML = `<div class="table-block">${agg}</div>
                    <div class="table-block">${roots}</div>`;
  } catch (e) {
    fail(el, e);
  }
}

// ------------------------------------------------- signature: divergence
function divergenceChart(rows) {
  const W = 900, rowH = 42, padT = 46, padB = 24;
  const labelW = 250, valW = 96;
  const plotL = labelW, plotR = W - valW;
  const mid = (plotL + plotR) / 2;
  const half = (plotR - plotL) / 2 - 10;
  const maxAdv = Math.max(...rows.map((r) => Math.abs(r.advantage)));
  const scale = (v) => (Math.abs(v) / maxAdv) * half;
  const H = padT + rows.length * rowH + padB;

  const ticks = [-1, -0.5, 0, 0.5, 1].map((f) => {
    const x = mid + f * half;
    return `<line class="grid" x1="${x}" y1="${padT - 8}" x2="${x}" y2="${H - padB}"/>`
         + `<text class="val" x="${x}" y="${padT - 14}" text-anchor="middle">`
         + `${f === 0 ? "0" : (Math.abs(f) * maxAdv).toFixed(0)}</text>`;
  }).join("");

  const bars = rows.map((r, i) => {
    const y = padT + i * rowH;
    const cy = y + rowH / 2;
    const w = scale(r.advantage);
    const toBandit = r.advantage > 0;
    const x = toBandit ? mid : mid - w;
    const cls = toBandit ? "bar-bandit" : "bar-ppo";
    const flag = toBandit ? "" :
      `<text class="flag" x="${plotL - 10}" y="${cy + 13}" text-anchor="end">PPO wins</text>`;
    return `
      <line class="row-rule" x1="0" y1="${y}" x2="${W}" y2="${y}"/>
      <text class="scen" x="${plotL - 10}" y="${cy + 1}" text-anchor="end"
            dominant-baseline="middle">${esc(SCENARIO_LABEL[r.scenario] || r.scenario)}</text>
      ${flag}
      <rect class="${cls}" x="${x}" y="${cy - 9}" width="${Math.max(w, 1.5)}" height="18">
        <title>${esc(SCENARIO_LABEL[r.scenario] || r.scenario)}: bandit ${num(r.bandit)}, `
          + `PPO ${num(r.ppo)}, margin ${num(Math.abs(r.advantage))} to `
          + `${esc(meta(r.winner).short)}</title>
      </rect>
      <text class="val" x="${plotR + 10}" y="${cy + 1}" dominant-baseline="middle">`
        + `${num(Math.abs(r.advantage))} ${esc(meta(r.winner).tok)}</text>`;
  }).join("");

  return `<svg class="diverge" viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Per-scenario margin between the two learners. Bars extend right when the bandit wins and left when PPO wins. The table below states every value.">
    <text class="head" x="${mid - 12}" y="18" text-anchor="end">◀ PPO wins</text>
    <text class="head" x="${mid + 12}" y="18">bandit wins ▶</text>
    ${ticks}
    ${bars}
    <line class="axis" x1="${mid}" y1="${padT - 8}" x2="${mid}" y2="${H - padB}"/>
    <line class="row-rule" x1="0" y1="${padT + rows.length * rowH}" x2="${W}"
          y2="${padT + rows.length * rowH}"/>
  </svg>`;
}

async function renderScenarios() {
  const fig = $("scenario-figure");
  const el = $("scenario-body");
  try {
    const d = await get("/api/v2/final-holdout/scenarios");
    const rows = [...d.scenarios].sort((a, b) => a.advantage - b.advantage);

    fig.innerHTML = divergenceChart(rows) + `
      <div class="legend">
        <span>${token("masked_bandit")} bandit margin, bar to the right</span>
        <span>${token("maskable_ppo")} PPO margin, bar to the left</span>
      </div>
      <figcaption>${esc(d.grain)}. Margins are in return points. The one
      left-pointing bar is the negative result the study preserved: PPO is better in
      the deceptive local optimum, so the bandit advantage is broad but not
      universal.</figcaption>`;

    el.innerHTML = `<div class="table-block">${table({
      caption: "Every scenario, including the baselines, with no omission.",
      head: [
        { label: "Scenario" }, { label: "Bandit", num: true }, { label: "PPO", num: true },
        { label: "Margin", num: true }, { label: "Winner" },
        { label: "Greedy", num: true }, { label: "CSPF", num: true },
        { label: "Static", num: true },
      ],
      rows: rows.map((r) => ({
        cls: rowClass(r.winner),
        cells: [
          esc(SCENARIO_LABEL[r.scenario] || r.scenario),
          num(r.bandit), num(r.ppo),
          `<strong>${num(Math.abs(r.advantage))}</strong>`,
          `${token(r.winner)}${esc(meta(r.winner).short)}`,
          num(r.baselines.greedy), num(r.baselines.cspf), num(r.baselines.static),
        ],
      })),
      foot: "Learner values average three training roots of five holdout seeds each. "
          + "Baselines have no training root and were evaluated once.",
    })}</div>`;
  } catch (e) {
    fail(fig, e);
    el.innerHTML = "";
  }
}

// ----------------------------------------------------------- operations
async function renderOperations() {
  const churnEl = $("churn-body");
  const rewardEl = $("reward-body");
  const actionEl = $("action-body");

  try {
    const fh = await get("/api/v2/final-holdout");
    const order = ["masked_bandit", "maskable_ppo", "greedy", "cspf", "static"];

    const ops = table({
      caption: "Operational outcome per method.",
      head: [
        { label: "Method" }, { label: "Delivered ratio", num: true },
        { label: "SLA intervals", num: true }, { label: "Peak util", num: true },
        { label: "Mean util", num: true }, { label: "Congested intervals", num: true },
        { label: "Overload", num: true }, { label: "Delay ms", num: true },
        { label: "Max delay ms", num: true }, { label: "Loss", num: true },
      ],
      rows: order.map((a) => {
        const r = fh.aggregate.find((x) => x.algorithm === a);
        return {
          cls: rowClass(a),
          cells: [
            `${token(a)}${esc(meta(a).label)}`,
            num(r.delivered_ratio_mean_mean, 4),
            num(r.sla_violations_demand_intervals_mean, 2),
            num(r.max_utilization_peak_mean),
            num(r.max_utilization_mean_mean),
            num(r.congested_link_intervals_mean, 2),
            num(r.overload_ratio_mean_mean, 5),
            num(r.delay_ms_mean_mean, 2),
            num(r.delay_ms_max_mean, 2),
            num(r.loss_ratio_mean_mean, 4),
          ],
        };
      }),
    });

    const churn = table({
      caption: "Route churn and protection events. The bandit's cost is visible here: "
             + "it moves more bandwidth than PPO.",
      head: [
        { label: "Method" }, { label: "Reroutes/h", num: true },
        { label: "Reversals", num: true }, { label: "Flaps/demand", num: true },
        { label: "Moved Mbps", num: true }, { label: "Accepted TE", num: true },
        { label: "Rejected TE", num: true }, { label: "Dwell active", num: true },
        { label: "FRR changes", num: true }, { label: "FRR disc.", num: true },
        { label: "Restorations", num: true },
      ],
      rows: order.map((a) => {
        const c = fh.churn[a];
        return {
          cls: rowClass(a),
          cells: [
            `${token(a)}${esc(meta(a).label)}`,
            num(c.reroutes_per_hour),
            num(c.te_reversals, 2),
            num(c.flaps_per_demand, 4),
            num(c.moved_mbps_total, 2),
            num(c.accepted_te_changes, 2),
            num(c.rejected_te_requests, 2),
            num(c.dwell_active_demand_intervals, 2),
            num(c.frr_changes, 2),
            num(c.frr_disconnections, 3),
            num(c.recovery_restorations, 3),
          ],
        };
      }),
      foot: "Bandit and PPO reroute at the same rate. The bandit reverses and flaps "
          + "less, but shifts more bandwidth per episode — far less than greedy.",
    });

    churnEl.innerHTML = `<div class="table-block">${ops}</div>
                         <div class="table-block">${churn}</div>`;

    // ---- reward components
    const rc = await get("/api/v2/final-holdout/reward-components");
    const comps = rc.component_names;
    rewardEl.innerHTML = `<div class="table-block">${table({
      caption: "Mean episodic reward components. Each row's twelve components sum to "
             + "its operational return.",
      head: [{ label: "Policy" }, ...comps.map((c) => ({ label: c, num: true })),
             { label: "Sum", num: true }, { label: "Return", num: true },
             { label: "Residual", num: true }],
      rows: rc.rows.map((r) => ({
        cls: rowClass(r.algorithm),
        cells: [
          `${token(r.algorithm)}${esc(r.policy_id)}`,
          ...comps.map((c) => num(r.components[c], 3)),
          num(r.sum, 3), num(r.operational_return_mean, 3),
          r.residual.toExponential(1),
        ],
      })),
      foot: `Component sums are exact: ${rc.exact ? "yes" : "no"}. Largest residual `
          + `recomputed here is ${rc.max_residual.toExponential(4)}; the study reported `
          + `${rc.reported_max_abs_residual.toExponential(4)} after separately `
          + `aggregating episode components.`,
    })}</div>`;

    // ---- actions and both no-op grains
    const act = await get("/api/v2/final-holdout/actions");
    const n = act.noop;
    const noopRows = order.map((a) => ({
      cls: rowClass(a),
      cells: [
        `${token(a)}${esc(meta(a).label)}`,
        pct(n.pooled_step_share[a]),
        pct(n.episode_mean_share[a]),
      ],
    }));
    const used = order.map((a) => {
      const rows = act.distribution.filter((d) => d.algorithm === a && d.count > 0
                                                 && d.action !== 0);
      const total = act.distribution.filter((d) => d.algorithm === a)
        .reduce((s, d) => s + d.count, 0);
      const top = [...rows].sort((x, y) => y.count - x.count).slice(0, 4);
      return {
        cls: rowClass(a),
        cells: [
          `${token(a)}${esc(meta(a).label)}`,
          int(rows.length),
          int(total),
          top.map((t) => `${t.action} (${int(t.count)})`).join(", ") || "—",
        ],
      };
    });

    actionEl.innerHTML = `
      <div class="table-block">
        <h3 class="block-title">No-op share, at both published grains</h3>
        <p class="block-note">These are two different statistics that are often given the
          same name. The final-holdout report quotes the pooled-step figure.</p>
        ${table({
          head: [{ label: "Method" },
                 { label: "Pooled over steps", num: true },
                 { label: "Mean over episodes", num: true }],
          rows: noopRows,
          foot: `Pooled: ${esc(n.pooled_grain)}. Episode mean: ${esc(n.episode_grain)}. `
              + `${int(n.steps_per_policy)} recorded steps per policy.`,
        })}
      </div>
      <div class="table-block">
        <h3 class="block-title">Action use</h3>
        <p class="block-note">All actions are retained in the frozen table, including
          those with zero count.</p>
        ${table({
          head: [{ label: "Method" }, { label: "Distinct non-no-op actions", num: true },
                 { label: "Recorded steps", num: true }, { label: "Most used" }],
          rows: used,
        })}
      </div>`;
  } catch (e) {
    fail(churnEl, e);
    rewardEl.innerHTML = "";
    actionEl.innerHTML = "";
  }
}

// ---------------------------------------------------------- development
async function renderDevelopment() {
  const el = $("development-body");
  const curvesEl = $("development-curves");
  try {
    const [cont, pilot] = await Promise.all([
      get("/api/v2/development/continuity"),
      get("/api/v2/development/seed42"),
    ]);
    const s = cont.summary;

    el.innerHTML = `
      <div class="table-block">
        <h3 class="block-title">Three-root continuity — development seeds ${s.evaluation_seeds.join(", ")}</h3>
        <p class="block-note">${esc(s.caption)} Holdout accessed during this stage:
          <span class="${s.holdout_accessed ? "status-bad" : "status-ok"}">${s.holdout_accessed ? "yes" : "no"}</span>.</p>
        ${table({
          head: [{ label: "Method" }, { label: "Root return", num: true },
                 { label: "Root SD", num: true }, { label: "Episode return", num: true },
                 { label: "Delivered", num: true }, { label: "SLA intervals", num: true },
                 { label: "Reroutes/h", num: true }, { label: "Moved Mbps", num: true },
                 { label: "Roots", num: true }],
          rows: s.methods.map((m) => ({
            cls: rowClass(m.algorithm),
            cells: [
              `${token(m.algorithm)}${esc(meta(m.algorithm).label)}`,
              `<strong>${num(m.root_return_mean)}</strong>`,
              m.roots > 1 ? num(m.root_return_std) : "—",
              num(m.episode_return_mean),
              num(m.delivered_ratio, 4),
              num(m.sla_violations_demand_intervals, 2),
              num(m.reroutes_per_hour),
              num(m.moved_mbps_total, 2),
              int(m.roots),
            ],
          })),
          foot: "Development figures. They are not the holdout result and must not be "
              + "read as generalization.",
        })}
      </div>
      <div class="table-block">
        <h3 class="block-title">Seed-42 pilot — single training root</h3>
        <p class="block-note">${esc(pilot.caption)}</p>
        ${table({
          head: [{ label: "Method" }, { label: "Return", num: true },
                 { label: "Episode SD", num: true }, { label: "Episodes", num: true }],
          rows: pilot.methods.map((m) => ({
            cls: rowClass(m.algorithm),
            cells: [
              `${token(m.algorithm)}${esc(meta(m.algorithm).label)}`,
              num(m.operational_return_mean),
              num(m.operational_return_std, 2),
              int(m.episodes),
            ],
          })),
        })}
      </div>`;

    renderCurves(curvesEl, cont.learning_curves);
  } catch (e) {
    fail(el, e);
    curvesEl.innerHTML = "";
  }
}

function renderCurves(host, curves) {
  const sel = table({
    head: [{ label: "Training root", num: true }, { label: "Method" },
           { label: "Selected checkpoint", num: true },
           { label: "Selected return", num: true },
           { label: "Checkpoints evaluated", num: true }],
    rows: curves.series.map((s) => {
      const chosen = s.points.find((p) => p.selected);
      return {
        cls: rowClass(s.algorithm),
        cells: [
          int(s.training_root),
          `${token(s.algorithm)}${esc(meta(s.algorithm).short)}`,
          int(s.selected_transition),
          chosen ? num(chosen.return) : "—",
          int(s.points.length),
        ],
      };
    }),
    foot: esc(curves.rule),
  });

  host.innerHTML = `
    <div class="table-block">
      <h3 class="block-title">Checkpoint curves</h3>
      <p class="block-note">${esc(curves.caption)}</p>
      <div class="chart-grid" id="curve-charts"></div>
      ${sel}
    </div>`;

  const grid = document.getElementById("curve-charts");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const roots = [...new Set(curves.series.map((s) => s.training_root))];

  // Every container is placed before any chart is initialised. Measuring a grid
  // cell while its siblings are still missing gives the first chart the whole
  // row's width, and it then paints wider than the box it lives in.
  const hosts = roots.map((root) => {
    const div = document.createElement("div");
    div.className = "chart";
    div.setAttribute("role", "img");
    grid.appendChild(div);
    return { root, div };
  });

  for (const { root, div } of hosts) {
    const series = curves.series.filter((s) => s.training_root === root);
    div.setAttribute("aria-label",
      `Development checkpoint returns for training root ${root}. ` +
      series.map((s) => `${meta(s.algorithm).short}: ` +
        s.points.map((p) => `${p.transition / 1000}k ${p.return.toFixed(2)}`).join(", "))
        .join(". ") + ". The selected checkpoint for each method is in the table below.");

    const chart = echarts.init(div, null, { renderer: "svg", width: div.clientWidth });
    chart.setOption({
      animation: !reduced,
      title: { text: `Training root ${root}`, left: 8, top: 6,
               textStyle: { fontSize: 12, fontWeight: 600, color: "#16202b" } },
      grid: { left: 52, right: 18, top: 42, bottom: 44 },
      tooltip: { trigger: "axis" },
      legend: { bottom: 4, itemWidth: 18, itemHeight: 8,
                textStyle: { fontSize: 11 } },
      xAxis: {
        type: "category",
        data: series[0].points.map((p) => `${p.transition / 1000}k`),
        axisLabel: { fontSize: 10 },
        name: "transitions", nameLocation: "middle", nameGap: 24,
        nameTextStyle: { fontSize: 10, color: "#62748a" },
      },
      yAxis: { type: "value", axisLabel: { fontSize: 10 },
               splitLine: { lineStyle: { color: "#dde4ea" } } },
      series: series.map((s) => ({
        name: meta(s.algorithm).short,
        type: "line",
        // shape differs per method as well as colour
        symbol: s.algorithm === "masked_bandit" ? "circle" : "rect",
        symbolSize: 7,
        lineStyle: {
          width: 2,
          type: s.algorithm === "masked_bandit" ? "solid" : "dashed",
        },
        color: s.algorithm === "masked_bandit" ? "#0a6355" : "#6a3ca8",
        data: s.points.map((p) => ({
          value: p.return,
          symbolSize: p.selected ? 13 : 7,
          itemStyle: p.selected ? { borderColor: "#7d5200", borderWidth: 2.5 } : {},
        })),
      })),
    });
    bindChartSize(chart, div);
  }
}

// ----------------------------------------------------------- provenance
async function renderProvenance() {
  const integEl = $("integrity-body");
  const provEl = $("provenance-body");
  const discEl = $("disclosure-body");

  try {
    const s = await get("/api/v2/final-holdout/integrity");
    const counters = Object.entries(s.counters);
    integEl.innerHTML = `
      <div class="table-block">
        <h3 class="block-title">Safety and integrity</h3>
        <p class="block-note">
          All checks passed:
          <span class="${s.all_checks_passed ? "status-ok" : "status-bad"}">${s.all_checks_passed ? "yes" : "no"}</span>
          across ${int(s.policies)} policies. Every episode reached normal truncation
          and none terminated abnormally.
        </p>
        ${table({
          head: [{ label: "Counter" }, { label: "Total", num: true }],
          rows: counters.map(([k, v]) => ({
            cells: [esc(k.replace(/_total$/, "").replace(/_/g, " ")),
                    `<span class="${v === 0 ? "status-ok" : "status-bad"}">${int(v)}</span>`],
          })),
          foot: `Protected disconnection demand-intervals are identical across every `
              + `method (${num(s.protected_disconnection_demand_intervals)}), as are `
              + `unprotected (${num(s.unprotected_disconnection_demand_intervals)}). `
              + `Rejected TE requests across all methods: ${num(s.rejected_te_requests_total, 0)}. `
              + `The gains did not change failure accounting.`,
        })}
      </div>`;

    const p = await get("/api/v2/final-holdout/provenance");
    const rt = p.runtime;
    provEl.innerHTML = `
      <div class="table-block">
        <h3 class="block-title">Checkpoint chain of custody</h3>
        <p class="block-note">Six fixed checkpoints, each bound to the source that
          trained it and to the single source that evaluated it.</p>
        ${table({
          head: [{ label: "Root", num: true }, { label: "Method" },
                 { label: "Transition", num: true }, { label: "Payload SHA-256" },
                 { label: "Sidecar SHA-256" }, { label: "Trained at" },
                 { label: "Evaluated at" }, { label: "Wall s", num: true },
                 { label: "Peak GPU bytes", num: true }],
          rows: p.checkpoints.map((c) => ({
            cls: rowClass(c.algorithm),
            cells: [
              id(c.training_root),
              `${token(c.algorithm)}${esc(meta(c.algorithm).short)}`,
              int(c.checkpoint_transition),
              `<code title="${esc(c.payload_sha256)}">${esc(c.payload_sha256.slice(0, 12))}…</code>`,
              `<code title="${esc(c.sidecar_sha256)}">${esc(c.sidecar_sha256.slice(0, 12))}…</code>`,
              `<code>${esc(sha(c.training_source_sha))}</code>`,
              `<code>${esc(sha(c.evaluation_source_sha))}</code>`,
              num(c.evaluation_wall_seconds, 3),
              int(c.peak_gpu_memory_bytes),
            ],
          })),
        })}
      </div>
      <div class="table-block">
        <h3 class="block-title">Runtime</h3>
        <p class="block-note">Two different wall-time figures, both reported. They are
          not interchangeable.</p>
        ${table({
          head: [{ label: "Measure" }, { label: "Seconds", num: true }, { label: "Grain" }],
          rows: [
            { cells: ["Whole one-shot runner", num(rt.total_runner_wall_seconds, 3),
                      esc(rt.total_grain)] },
            { cells: ["Six learner evaluations", num(rt.checkpoint_wall_seconds_sum, 3),
                      esc(rt.checkpoint_grain)] },
          ],
          foot: `${esc(rt.device)} · ${esc(rt.gpu)} · torch ${esc(rt.torch)} · CUDA `
              + `${esc(rt.cuda_runtime)}. Peak allocated GPU memory ranged `
              + `${int(rt.peak_gpu_memory_bytes_min)}–${int(rt.peak_gpu_memory_bytes_max)} bytes.`,
        })}
      </div>
      <div class="table-block">
        <h3 class="block-title">Artifact locations</h3>
        <p class="block-note">Compact evidence is committed. Full step traces and every
          checkpoint stay outside Git.</p>
        <table>
          <tbody>
            <tr><th scope="row">Compact evidence</th><td><code>${esc(p.artifact_path)}</code></td></tr>
            <tr><th scope="row">Full step traces</th><td><code>${esc(p.full_artifact_path || "not recorded")}</code></td></tr>
          </tbody>
        </table>
      </div>`;

    // ---- disclosures, progressively disclosed
    const d = await get("/api/v2/disclosures");
    const items = d.disclosures.map((x) => `
      <details class="disclosure">
        <summary>
          <span class="kind kind-${esc(x.kind)}">${esc(x.kind)}</span>
          <span>${esc(x.title)}</span>
        </summary>
        <div class="disclosure-detail">
          <p>${esc(x.summary)}</p>
          <dl>
            <dt>Stage</dt><dd>${esc(x.stage.replace("_", " "))}</dd>
            <dt>Study</dt><dd>${esc(x.study)}</dd>
            <dt>Preserved</dt><dd>${x.preserved ? "yes" : "no run directory was created"}</dd>
            <dt>Used in results</dt><dd>no</dd>
            ${x.path ? `<dt>Evidence</dt><dd><code>${esc(x.path)}</code></dd>` : ""}
          </dl>
        </div>
      </details>`).join("");

    const kinds = Object.entries(d.kinds).map(([k, v]) =>
      `<tr><td><span class="kind kind-${esc(k)}">${esc(k)}</span></td><td>${esc(v)}</td></tr>`).join("");

    discEl.innerHTML = `
      <div class="table-block">
        <h3 class="block-title">Invalidated, superseded and repaired runs</h3>
        <p class="block-note">Every run the study discarded or replaced is recorded here.
          None contributed to any reported result. The three statuses mean different
          things and are never merged.</p>
        <table><tbody>${kinds}</tbody></table>
        <div style="margin-top:1rem">${items}</div>
      </div>`;
  } catch (e) {
    fail(integEl, e);
    provEl.innerHTML = "";
    discEl.innerHTML = "";
  }
}

// --------------------------------------------------------------- replay
async function renderReplay() {
  const controls = $("replay-controls");
  const readout = $("replay-readout");
  const timeline = $("replay-timeline");
  const body = $("replay-body");

  let index;
  try {
    index = await get("/api/v2/replay/index");
  } catch (e) {
    fail(controls, e);
    return;
  }

  const policies = [...new Map(index.episodes.map((e) =>
    [e.policy_id, e])).values()];

  const opt = (v, label, sel) =>
    `<option value="${esc(v)}"${sel ? " selected" : ""}>${esc(label)}</option>`;

  const policyLabel = (p) => p.is_learner
    ? `${meta(p.algorithm).short} · root ${p.training_root} · ${p.checkpoint_transition / 1000}k`
    : `${meta(p.algorithm).short} · baseline`;

  controls.innerHTML = `
    <span class="recorded-chip">Recorded evidence</span>
    <label>
      <span class="field-label">Controller</span>
      <select id="rp-policy">${policies.map((p) =>
        opt(p.policy_id, policyLabel(p), p.policy_id === "root42_masked_bandit")).join("")}</select>
    </label>
    <label>
      <span class="field-label">Scenario</span>
      <select id="rp-scenario">${Object.keys(index.scenario_steps).map((s) =>
        opt(s, `${SCENARIO_LABEL[s] || s} · ${index.scenario_steps[s]} steps`,
            s === "link_failure")).join("")}</select>
    </label>
    <label>
      <span class="field-label">Holdout seed</span>
      <select id="rp-seed">${[...new Set(index.episodes.map((e) => e.seed))]
        .map((s) => opt(s, s, s === 1001)).join("")}</select>
    </label>
    <button id="rp-load" type="button">Load episode</button>`;

  if (!index.available) {
    body.innerHTML = `<div class="unavailable">
      <strong>Recorded traces are not configured on this machine.</strong>
      All ${int(index.episodes.length)} episodes are catalogued above, but their step
      files live outside Git. Set <code>V2_FULL_ARTIFACTS</code> to the directory named
      in <code>results/v2_final_holdout/manifest.json</code> under
      <code>full_artifact_path</code>, then restart the server.</div>`;
    $("rp-load").disabled = true;
    return;
  }

  body.innerHTML = `<p class="block-note">Catalogued episodes:
    ${int(index.episodes.length)}. Traces read from
    <code>${esc(index.artifact_root)}</code>.</p>`;

  $("rp-load").addEventListener("click", loadEpisode);
  await loadEpisode();

  async function loadEpisode() {
    const policy_id = $("rp-policy").value;
    const scenario = $("rp-scenario").value;
    const seed = $("rp-seed").value;
    readout.innerHTML = `<p class="loading">Reading recorded trace…</p>`;
    timeline.innerHTML = "";
    let ep;
    try {
      ep = await get(`/api/v2/replay/episode?policy_id=${encodeURIComponent(policy_id)}`
        + `&scenario=${encodeURIComponent(scenario)}&seed=${encodeURIComponent(seed)}`);
    } catch (e) {
      fail(readout, e);
      return;
    }
    drawEpisode(ep);
  }

  function drawEpisode(ep) {
    const p = ep.provenance;
    const steps = ep.steps;

    // The playback surface may only ever show a recorded trace. If the payload is
    // not marked as one, refuse rather than let it read as a live evaluation.
    if (p.kind !== "recorded_replay" || p.live !== false) {
      readout.innerHTML = `<div class="unavailable"><strong>Refused.</strong>
        This payload is not marked as recorded evidence, so it will not be
        played back here.</div>`;
      return;
    }
    const acted = steps.filter((s) => s.action !== 0);

    readout.innerHTML = `
      <div class="stat-row">
        <div class="stat">
          <p class="stat-label">Recorded return</p>
          <span class="stat-value">${num(p.operational_return, 3)}</span>
          <p class="stat-sub">sum of ${int(steps.length)} recorded step rewards</p>
        </div>
        <div class="stat">
          <p class="stat-label">Actions taken</p>
          <span class="stat-value">${int(acted.length)}</span>
          <p class="stat-sub">non-no-op control intervals</p>
        </div>
        <div class="stat">
          <p class="stat-label">Peak busiest-link util</p>
          <span class="stat-value">${num(Math.max(...steps.map((s) => s.max_util)), 3)}</span>
          <p class="stat-sub">recorded maximum</p>
        </div>
        <div class="stat">
          <p class="stat-label">SLA intervals</p>
          <span class="stat-value">${int(steps.reduce((a, s) => a + (s.sla_violations || 0), 0))}</span>
          <p class="stat-sub">demand-intervals over the episode</p>
        </div>
      </div>
      <div class="table-block">
        <table><tbody>
          <tr><th scope="row">Record</th><td>
            <span class="recorded-chip">Recorded</span>
            playback of a preserved trace — not a live policy evaluation</td></tr>
          <tr><th scope="row">Controller</th><td>${token(p.algorithm)}${esc(meta(p.algorithm).label)}${
            p.is_learner ? ` · training root ${id(p.training_root)} · checkpoint ${int(p.checkpoint_transition)}`
                         : " · baseline, no training root"}</td></tr>
          <tr><th scope="row">Episode</th><td>${esc(SCENARIO_LABEL[p.scenario] || p.scenario)} ·
            holdout seed ${id(p.seed)} · episode seed ${id(p.episode_seed)} ·
            ${int(p.episode_length)} intervals · ${p.truncated ? "normal truncation" : "abnormal end"}</td></tr>
          <tr><th scope="row">Provenance</th><td>evaluated at
            <code>${esc(sha(p.evaluation_source_sha))}</code>${
            p.training_source_sha ? `, trained at <code>${esc(sha(p.training_source_sha))}</code>` : ""} ·
            <code>${esc(p.artifact_path)}</code></td></tr>
        </tbody></table>
      </div>`;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    timeline.innerHTML = `<div class="chart" id="rp-chart" role="img"
      aria-label="Recorded per-interval busiest-link utilization, mean delay and reward
      for ${esc(SCENARIO_LABEL[p.scenario] || p.scenario)}, seed ${p.seed}. Peak
      utilization ${Math.max(...steps.map((s) => s.max_util)).toFixed(3)}. The table
      above states the episode totals."></div>`;

    const rpEl = document.getElementById("rp-chart");
    const chart = echarts.init(rpEl, null,
      { renderer: "svg", width: rpEl.clientWidth });
    const x = steps.map((s) => s.step_index);
    chart.setOption({
      animation: !reduced,
      grid: { left: 56, right: 56, top: 34, bottom: 46 },
      tooltip: { trigger: "axis" },
      legend: { bottom: 4, textStyle: { fontSize: 11 } },
      xAxis: { type: "category", data: x, name: "control interval",
               nameLocation: "middle", nameGap: 24,
               nameTextStyle: { fontSize: 10, color: "#62748a" },
               axisLabel: { fontSize: 10 } },
      yAxis: [
        { type: "value", name: "util / delay", nameTextStyle: { fontSize: 10 },
          axisLabel: { fontSize: 10 }, splitLine: { lineStyle: { color: "#dde4ea" } } },
        { type: "value", name: "reward", nameTextStyle: { fontSize: 10 },
          axisLabel: { fontSize: 10 }, splitLine: { show: false } },
      ],
      series: [
        { name: "busiest-link util", type: "line", data: steps.map((s) => s.max_util),
          symbol: "none", lineStyle: { width: 2 }, color: "#0a6355" },
        { name: "mean delay ms ÷ 100", type: "line",
          data: steps.map((s) => s.mean_delay_ms / 100), symbol: "none",
          lineStyle: { width: 2, type: "dashed" }, color: "#6a3ca8" },
        { name: "step reward", type: "bar", yAxisIndex: 1,
          data: steps.map((s) => s.reward), color: "#c6d0d9" },
      ],
    });
    bindChartSize(chart, rpEl);
  }
}

// ------------------------------------------------------------ spine state
function wireSpine() {
  const links = [...document.querySelectorAll(".spine-list a")];
  const sections = links
    .map((a) => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);
  const obs = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      links.forEach((a) => a.removeAttribute("aria-current"));
      const active = links.find((a) => a.getAttribute("href") === `#${e.target.id}`);
      if (active) active.setAttribute("aria-current", "true");
    }
  }, { rootMargin: "-20% 0px -70% 0px" });
  sections.forEach((s) => obs.observe(s));
}

boot();
