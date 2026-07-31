/* Minimal DOM helpers. No framework, no build step, no CDN. */

export const SVG_NS = "http://www.w3.org/2000/svg";

export const $ = (id) => document.getElementById(id);

export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  apply(node, props);
  append(node, children);
  return node;
}

export function svg(tag, props = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "text") node.textContent = String(value);
    else if (key === "class") node.setAttribute("class", value);
    else node.setAttribute(key, String(value));
  }
  append(node, children);
  return node;
}

function apply(node, props) {
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "html") node.innerHTML = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key in node && key !== "list" && key !== "type") {
      node[key] = value;
    } else {
      node.setAttribute(key, value === true ? "" : String(value));
    }
  }
}

function append(node, children) {
  const list = Array.isArray(children) ? children : [children];
  for (const child of list) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
}

export function clear(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function fill(node, children) {
  clear(node);
  append(node, children);
  return node;
}

/** A labelled unavailable state. Never an empty container. */
export function unavailable(head, reason) {
  return el("div", { class: "unavailable" }, [
    el("strong", { class: "unavailable__head", text: head }),
    el("span", { text: reason || "No source supplies this value." }),
  ]);
}

export function facts(rows) {
  const node = el("dl", { class: "facts" });
  for (const [term, value] of rows) {
    if (value === null || value === undefined) continue;
    node.appendChild(el("dt", { text: term }));
    node.appendChild(el("dd", typeof value === "string" || typeof value === "number"
      ? { text: String(value) } : {}, typeof value === "object" ? [value] : []));
  }
  return node;
}

export function icon(name) {
  const node = svg("svg", { class: "icon", "aria-hidden": "true" });
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `#i-${name}`);
  node.appendChild(use);
  return node;
}

export function tag(text, state) {
  return el("span", { class: "tag", dataset: { state }, text });
}

/** Focus trap for a drawer. Returns a teardown function. */
export function trapFocus(container, onEscape) {
  const selector = 'a[href], button:not([disabled]), input, select, textarea, ' +
    '[tabindex]:not([tabindex="-1"])';
  function onKeyDown(event) {
    if (event.key === "Escape") { event.stopPropagation(); onEscape(); return; }
    if (event.key !== "Tab") return;
    const focusable = [...container.querySelectorAll(selector)]
      .filter((node) => node.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault(); last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault(); first.focus();
    }
  }
  container.addEventListener("keydown", onKeyDown);
  return () => container.removeEventListener("keydown", onKeyDown);
}

export function isTypingTarget(node) {
  if (!node) return false;
  const name = node.tagName;
  return name === "INPUT" || name === "TEXTAREA" || name === "SELECT"
    || node.isContentEditable;
}
