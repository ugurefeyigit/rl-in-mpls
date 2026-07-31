/* The topology stage.
 *
 * Node positions come from the display registry and are written once. Data
 * updates rewrite text and state attributes only — a node never moves, because
 * a moving node destroys the one stable anchor the whole product depends on.
 *
 * There is no force-directed layout and no physics. Zoom and pan change the
 * viewBox; they never change a relationship.
 */

import { $, clear, svg } from "./dom.js";
import { mbps, percent } from "./format.js";

const PLATE_W = 15.5;
const PLATE_H = 6.2;

export class TopologyAtlas {
  constructor({ onSelect, onFocusChange }) {
    this.root = $("atlas-svg");
    this.linkLayer = $("atlas-links");
    this.overlayLayer = $("atlas-overlays");
    this.nodeLayer = $("atlas-nodes");
    this.onSelect = onSelect;
    this.onFocusChange = onFocusChange;
    this.map = null;
    this.nodes = new Map();
    this.links = new Map();
    this.order = [];
    this.zoom = 1;
    this.center = { x: 50, y: 50 };
    this.selected = null;
  }

  /** Draw the fixed geometry once. */
  build(map) {
    this.map = map;
    clear(this.linkLayer);
    clear(this.overlayLayer);
    clear(this.nodeLayer);
    this.nodes.clear();
    this.links.clear();

    const byId = new Map(map.nodes.map((n) => [n.id, n]));

    for (const link of map.links) {
      const a = byId.get(link.a);
      const z = byId.get(link.z);
      if (!a || !z) continue;
      const points = [[a.x, a.y], ...link.bends, [z.x, z.y]];
      const d = pathData(points);
      const group = svg("g", {
        class: "lnk",
        role: "button",
        tabindex: "-1",
        "data-link": link.id,
        "aria-label": `${link.label} link, ${link.id}`,
      }, [
        svg("path", { class: "lnk__focus", d }),
        svg("path", { class: "lnk__hit", d }),
        svg("path", { class: "lnk__line", d, "stroke-width": link.stroke / 4 }),
        svg("g", { class: "lnk__ticks" }),
        svg("g", { class: "lnk__break" }),
      ]);
      group.addEventListener("click", () => this.select("link", link.id, true));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          this.select("link", link.id, true);
        }
      });
      this.linkLayer.appendChild(group);
      this.links.set(link.id, { def: link, group, points });
    }

    // Geographic reading order: west to east, then north to south.
    this.order = [...map.nodes].sort((p, q) => (p.x - q.x) || (p.y - q.y))
      .map((n) => n.id);

    for (const node of map.nodes) {
      const anchor = plateAnchor(node);
      const group = svg("g", {
        class: "node",
        role: "button",
        tabindex: "-1",
        "data-node": node.id,
        "data-role": node.role,
        "aria-label": `${node.city}, ${node.role_label}, ${node.id}`,
      }, [
        svg("rect", {
          class: "node__focus",
          x: anchor.x - 0.8, y: anchor.y - 0.8,
          width: PLATE_W + 1.6, height: PLATE_H + 1.6,
        }),
        svg("rect", {
          class: "node__plate",
          x: anchor.x, y: anchor.y, width: PLATE_W, height: PLATE_H,
        }),
        svg("path", {
          class: "node__edge",
          d: `M${anchor.x} ${anchor.y}v${PLATE_H}`,
        }),
        svg("path", {
          class: "node__notch",
          d: `M${anchor.x + PLATE_W / 2 - 1} ${anchor.y}l1 -1.1 1 1.1z`,
        }),
        svg("text", {
          class: "node__city", x: anchor.x + 1.1, y: anchor.y + 2.7,
          text: node.title,
        }),
        svg("text", {
          class: "node__meta", x: anchor.x + 1.1, y: anchor.y + 5.1,
          "data-meta": node.id, text: node.id,
        }),
        svg("path", { class: "node__marker", d: "" }),
        svg("circle", { class: "node__dot", cx: node.x, cy: node.y, r: 0.55,
                        fill: "var(--route-base)" }),
      ]);
      group.addEventListener("click", () => this.select("router", node.id, true));
      group.addEventListener("keydown", (event) => this.onNodeKey(event, node.id));
      this.nodeLayer.appendChild(group);
      this.nodes.set(node.id, { def: node, group, anchor });
    }

    this.setFocusable(this.order[0]);
  }

  onNodeKey(event, nodeId) {
    const moves = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 1, ArrowUp: -1 };
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      this.select("router", nodeId, true);
      this.onSelect?.("router", nodeId, { open: true });
      return;
    }
    if (!(event.key in moves)) return;
    event.preventDefault();
    const index = this.order.indexOf(nodeId);
    const next = this.order[(index + moves[event.key] + this.order.length) % this.order.length];
    this.setFocusable(next);
    this.nodes.get(next)?.group.focus();
    this.onFocusChange?.(next);
  }

  setFocusable(nodeId) {
    for (const [id, entry] of this.nodes) {
      entry.group.setAttribute("tabindex", id === nodeId ? "0" : "-1");
    }
  }

  select(objectType, objectId, notify = false) {
    this.selected = { objectType, objectId };
    for (const [id, entry] of this.nodes) {
      entry.group.setAttribute("aria-selected",
        objectType === "router" && id === objectId ? "true" : "false");
    }
    for (const [id, entry] of this.links) {
      entry.group.setAttribute("aria-selected",
        objectType === "link" && id === objectId ? "true" : "false");
    }
    if (notify) this.onSelect?.(objectType, objectId, { open: false });
  }

  /** Update state and telemetry without moving anything. */
  update(snapshot, { showTelemetry = true, selectedDemand = null } = {}) {
    if (!this.map || !snapshot) return;

    for (const link of snapshot.links || []) {
      const entry = this.links.get(link.id);
      if (!entry) continue;
      const { group, points } = entry;
      group.dataset.state = link.state;
      group.dataset.band = showTelemetry ? link.band : "quiet";
      group.setAttribute("aria-label", telemetryLabel(link, showTelemetry));

      const ticks = group.querySelector(".lnk__ticks");
      const breaks = group.querySelector(".lnk__break");
      clear(ticks);
      clear(breaks);
      if (showTelemetry && link.pressure_ticks > 0) {
        const mid = midpoint(points);
        for (let i = 0; i < link.pressure_ticks; i += 1) {
          ticks.appendChild(svg("path", {
            d: `M${mid.x - 1 + i * 1} ${mid.y - 1.1}v2.2`,
          }));
        }
      }
      if (!link.up) {
        const mid = midpoint(points);
        breaks.appendChild(svg("path", {
          d: `M${mid.x - 1} ${mid.y - 1}l2 2M${mid.x + 1} ${mid.y - 1}l-2 2`,
        }));
      }
    }

    for (const node of snapshot.nodes || []) {
      const entry = this.nodes.get(node.id);
      if (!entry) continue;
      const meta = entry.group.querySelector(".node__meta");
      const worst = node.worst_adjacent_utilization;
      meta.textContent = showTelemetry
        ? `${node.id}  ${node.n_lsps} LSP · ${percent(worst, 0)}`
        : node.id;
      const condition = node.has_failed_link ? "failure"
        : (worst !== null && worst >= 0.9 ? "pressure" : "normal");
      entry.group.dataset.condition = condition;
      const marker = entry.group.querySelector(".node__marker");
      marker.setAttribute("d", condition === "normal" ? ""
        : markerPath(entry.anchor, condition));
      entry.group.setAttribute("aria-label",
        `${node.city}, ${node.role_label}, ${node.id}. ` +
        (showTelemetry
          ? `${node.n_lsps} active LSPs, busiest adjacent link ${percent(worst, 0)}.`
          : "Reference topology; no recorded link telemetry."));
    }

    this.drawRoutes(snapshot, selectedDemand);
  }

  /** Route overlays: current, alternates, comparator, and Decision Lens. */
  drawRoutes(snapshot, selectedDemand, lens = null) {
    clear(this.overlayLayer);
    const demand = selectedDemand
      ? (snapshot.demands || []).find((d) => d.id === selectedDemand)
      : null;

    if (demand) {
      this.overlayLayer.appendChild(svg("path", {
        class: "route route--current",
        d: pathData(this.routerPoints(demand.current_path)),
      }));
    }
    if (lens) {
      if (lens.oldPath) {
        this.overlayLayer.appendChild(svg("path", {
          class: "route route--old", d: pathData(this.routerPoints(lens.oldPath)),
        }));
      }
      if (lens.proposedPath) {
        const d = pathData(this.routerPoints(lens.proposedPath));
        const node = svg("path", { class: "route route--proposed", d });
        this.overlayLayer.appendChild(node);
        const length = node.getTotalLength ? node.getTotalLength() : 200;
        node.style.setProperty("--route-len", String(length));
      }
      if (lens.comparatorPath) {
        this.overlayLayer.appendChild(svg("path", {
          class: "route route--comparator",
          d: pathData(this.routerPoints(lens.comparatorPath)),
        }));
      }
    }
    this.lens = lens;
  }

  showAlternates(demand) {
    if (!demand) return;
    for (const candidate of demand.candidates || []) {
      if (candidate.is_current) continue;
      const points = this.routerPoints(candidate.routers);
      if (points.length < 2) continue;
      this.overlayLayer.appendChild(svg("path", {
        class: "route route--alternate", d: pathData(points),
      }));
      const mid = midpoint(points);
      this.overlayLayer.appendChild(svg("text", {
        class: "route-key", x: mid.x, y: mid.y - 1.2,
        text: `p${candidate.path_idx}`,
      }));
    }
  }

  routerPoints(routers) {
    return (routers || [])
      .map((id) => this.nodes.get(id)?.def)
      .filter(Boolean)
      .map((n) => [n.x, n.y]);
  }

  setZoom(zoom, center = this.center) {
    this.zoom = Math.min(4, Math.max(0.6, zoom));
    this.center = center;
    const size = 100 / this.zoom;
    const x = Math.max(0, Math.min(100 - size, center.x - size / 2));
    const y = Math.max(0, Math.min(100 - size, center.y - size / 2));
    this.root.setAttribute("viewBox", `${x} ${y} ${size} ${size}`);
  }

  resetView() { this.setZoom(1, { x: 50, y: 50 }); }

  fitTo(objectId) {
    const node = this.nodes.get(objectId)?.def;
    if (node) this.setZoom(1.8, { x: node.x, y: node.y });
    else this.resetView();
  }
}

function plateAnchor(node) {
  switch (node.label_anchor) {
    case "left": return { x: node.x + 1.4, y: node.y - PLATE_H / 2 };
    case "right": return { x: node.x - PLATE_W - 1.4, y: node.y - PLATE_H / 2 };
    case "above": return { x: node.x - PLATE_W / 2, y: node.y - PLATE_H - 1.6 };
    default: return { x: node.x - PLATE_W / 2, y: node.y + 1.6 };
  }
}

function markerPath(anchor, condition) {
  const x = anchor.x + PLATE_W - 2.6;
  const y = anchor.y + 1.2;
  if (condition === "failure") return `M${x} ${y}l1.8 1.8M${x + 1.8} ${y}l-1.8 1.8`;
  return `M${x + 0.9} ${y}l1.1 2h-2.2z`;
}

function pathData(points) {
  if (!points.length) return "";
  const [first, ...rest] = points;
  return `M${first[0]} ${first[1]}` + rest.map((p) => `L${p[0]} ${p[1]}`).join("");
}

function midpoint(points) {
  const index = Math.floor((points.length - 1) / 2);
  const a = points[index];
  const b = points[index + 1] || points[index];
  return { x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2 };
}

function telemetryLabel(link, showTelemetry) {
  const head = `${link.label} link, ${link.id}, ${mbps(link.capacity_mbps)}`;
  if (!showTelemetry) return `${head}. Reference topology; no recorded link telemetry.`;
  if (!link.up) return `${head}. Failed.`;
  return `${head}. Busiest direction ${link.worst_direction} at ` +
         `${percent(link.worst_utilization, 0)}, ${link.band_label}.`;
}
