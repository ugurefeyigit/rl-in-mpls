// Shared number/label formatting. Every number rendered in either UI goes
// through one of these so the two frontends can never disagree on rounding.
//
// Rules (docs/UI_ACCEPTANCE_TESTS.md):
//   rate         < 1000 Mbps -> "740 Mbps";  >= 1000 -> "1.2 Gbps"
//   utilization  1.275       -> "128%"
//   loss         0.00823     -> "0.82%"
//   delay        21.37       -> "21.4 ms"
//   reward       -76.8712    -> "-76.9"
//   sim time     19.1667     -> "19:10"

// Display-only scale for Presentation Mode's "scaled national backbone" view.
// Mirrors mplssim/display.py :: scale_mbps() — applied identically to loads AND
// capacities, so utilization, delay, loss, SLA counts and rewards are invariant.
// Single source of truth for the JS side; the server-side constant is tested by
// tests/test_correctness_fixes.py::test_display_scale_keeps_utilization_invariant.
export const DISPLAY_SCALE = 10;

export const SCALE_NOTE =
  "Scaled national-backbone view: traffic and capacities displayed at 10× " +
  "laboratory scale; utilization, delay, loss, actions, and rewards are unchanged.";

/** Traffic rate in Mbps -> "740 Mbps" / "1.2 Gbps". `scale` multiplies first. */
export function rate(mbps, scale = 1) {
  const v = (Number(mbps) || 0) * scale;
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(1)} Gbps`;
  return `${Math.round(v)} Mbps`;
}

/** Rate without the unit, for table cells that carry the unit in the header. */
export function rateValue(mbps, scale = 1) {
  const v = (Number(mbps) || 0) * scale;
  return Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) : String(Math.round(v));
}

export function rateUnit(mbps, scale = 1) {
  return Math.abs((Number(mbps) || 0) * scale) >= 1000 ? "Gbps" : "Mbps";
}

/** Utilization fraction -> "128%". Values above 1 are real overload, not clipped. */
export function util(fraction, decimals = 0) {
  return `${((Number(fraction) || 0) * 100).toFixed(decimals)}%`;
}

/** Loss *fraction* (0.00823) -> "0.82%". */
export function loss(fraction, decimals = 2) {
  return `${((Number(fraction) || 0) * 100).toFixed(decimals)}%`;
}

/** Loss already expressed in percent (snapshot demand.loss_pct) -> "0.82%". */
export function lossPct(percent, decimals = 2) {
  return `${(Number(percent) || 0).toFixed(decimals)}%`;
}

export function delay(ms) {
  return `${(Number(ms) || 0).toFixed(1)} ms`;
}

/** Reward, 1 decimal, explicit sign so gains and losses read at a glance. */
export function reward(value, sign = false) {
  const v = Number(value) || 0;
  return `${sign && v > 0 ? "+" : ""}${v.toFixed(1)}`;
}

export function signed(value, decimals = 1) {
  const v = Number(value) || 0;
  return `${v > 0 ? "+" : ""}${v.toFixed(decimals)}`;
}

/** Fractional hour-of-day -> "19:10". */
export function simTime(hour) {
  const h = Number(hour) || 0;
  const total = Math.round(h * 60) % 1440;
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:` +
         `${String(total % 60).padStart(2, "0")}`;
}

/** Percentage of a ratio in 0..1 -> "98.9%" (delivered ratio, fairness, …). */
export function ratio(value, decimals = 1) {
  return `${((Number(value) || 0) * 100).toFixed(decimals)}%`;
}

export function int(value) {
  return String(Math.round(Number(value) || 0));
}

/** HTML-escape — every label from the API passes through this before innerHTML. */
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
