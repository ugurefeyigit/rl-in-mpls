/* Number and time formatting.
 *
 * Formatting is a truth surface here, not a cosmetic one. A probability and a
 * bandit score are formatted by *different* functions on purpose: there is no
 * single "format a policy number" call that could put a percent sign on an
 * unnormalized immediate-reward estimate.
 *
 * Every changing value reserves width through tabular numerals, so a readout
 * does not resize while a presenter is pointing at it.
 */

export const UNAVAILABLE = "—";

export function num(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return UNAVAILABLE;
  return Number(value).toFixed(digits);
}

export function signed(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return UNAVAILABLE;
  const text = Number(value).toFixed(digits);
  return Number(value) > 0 ? `+${text}` : text;
}

export function percent(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return UNAVAILABLE;
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function points(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return UNAVAILABLE;
  const text = (Number(value) * 100).toFixed(digits);
  return `${Number(value) > 0 ? "+" : ""}${text} pp`;
}

export function mbps(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return UNAVAILABLE;
  const v = Number(value);
  return v >= 1000 ? `${(v / 1000).toFixed(2)} Gbps` : `${v.toFixed(0)} Mbps`;
}

export function ms(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return UNAVAILABLE;
  return `${Number(value).toFixed(digits)} ms`;
}

export function count(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return UNAVAILABLE;
  return String(Math.round(Number(value)));
}

export function clock(hour) {
  if (hour === null || hour === undefined || Number.isNaN(hour)) return UNAVAILABLE;
  const total = Math.round(Number(hour) * 60) % (24 * 60);
  const h = String(Math.floor(total / 60)).padStart(2, "0");
  const m = String(total % 60).padStart(2, "0");
  return `${h}:${m}`;
}

export function shortHash(value) {
  return value ? String(value).slice(0, 10) : UNAVAILABLE;
}

/**
 * A PPO action probability. Only ever called for `semantics === "probabilities"`,
 * because only a real normalized masked distribution may be shown as a percent.
 */
export function probability(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return UNAVAILABLE;
  return `${(Number(value) * 100).toFixed(1)}%`;
}

/**
 * A masked-bandit action score. Unnormalized, may be negative, never a percent
 * and never called a probability or a confidence.
 */
export function score(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return UNAVAILABLE;
  return signed(value, 3);
}

/** Pick the right formatter from the *declared* semantics, never from the value. */
export function policyValue(semantics, value) {
  if (semantics === "probabilities") return probability(value);
  if (semantics === "scores") return score(value);
  return UNAVAILABLE;
}

export function metricValue(unit, value) {
  if (value === null || value === undefined) return UNAVAILABLE;
  switch (unit) {
    case "share": return percent(value, 1);
    case "ms": return ms(value);
    case "count": return count(value);
    case "mbps": return mbps(value);
    default: return num(value);
  }
}

export function metricDelta(unit, value) {
  if (value === null || value === undefined) return UNAVAILABLE;
  if (unit === "share") return points(value);
  if (unit === "count") return signed(value, 0);
  return signed(value, unit === "ms" ? 1 : 3);
}

export function plural(n, one, many) {
  return `${count(n)} ${Number(n) === 1 ? one : many}`;
}
