// Topology view: Cytoscape canvases (1 in single mode, 2 in compare mode;
// Presentation Mode reuses the same class with larger type and no hovercard).
// Edges are undirected physical links; color = worst-direction utilization,
// width = capacity, dashed red = failed. Selected LSP overlays glow.
//
// Node labels are CITY names (display layer). The internal router ID stays in
// the element data and is shown in the hover card — it is never renamed.

import { city, linkLabel, linkTechnical, demandLabel, pathLabel } from "./display.js";
import { rate, util, delay, loss } from "./fmt.js";

const UTIL_COLORS = [
  [0.50, "#2ea043"], [0.75, "#d29922"], [0.90, "#f0883e"],
  [1.00, "#f85149"], [Infinity, "#db61a2"],
];

export function utilColor(u) {
  for (const [t, c] of UTIL_COLORS) if (u < t) return c;
  return "#db61a2";
}

const NODE_SHAPE = { PE_IN: "round-rectangle", PE_OUT: "round-rectangle", P: "ellipse", AGG: "diamond" };
const NODE_COLOR = { PE_IN: "#1f4e8c", PE_OUT: "#274764", P: "#2a3340", AGG: "#3b3050" };

export class TopoView {
  /**
   * @param {string} containerId
   * @param {HTMLElement|null} hovercardEl  omit for Presentation Mode
   * @param {{fontSize?:number, nodeScale?:number, scale?:number,
   *          hover?:boolean, failCross?:boolean}} opts
   *        scale = display-only Mbps multiplier (see fmt.js DISPLAY_SCALE)
   */
  constructor(containerId, hovercardEl, opts = {}) {
    this.container = document.getElementById(containerId);
    this.hover = hovercardEl;
    this.cy = null;
    this.selectedDemand = null;
    this.highlightPath = null;      // explicit router chain (advisor proposal)
    this.lastSnapshot = null;
    this.onDemandSelect = null;
    this.opts = {
      fontSize: 11, nodeScale: 1, scale: 1,
      hover: Boolean(hovercardEl), failCross: false, ...opts,
    };
  }

  /** Change the display-only Mbps multiplier (Presentation Mode scale toggle). */
  setScale(scale) { this.opts.scale = scale; }

  init(topology) {
    const { fontSize, nodeScale } = this.opts;
    const elements = [];
    for (const r of topology.routers) {
      elements.push({
        data: { id: r.id, label: city(r.id), rid: r.id, role: r.role },
        position: { x: r.x, y: r.y },
      });
    }
    for (const l of topology.links) {
      elements.push({ data: { id: l.id, source: l.a, target: l.z,
                              capacity: l.capacity_mbps } });
    }
    this.cy = cytoscape({
      container: this.container,
      elements,
      layout: { name: "preset", fit: true, padding: 30 },
      autoungrabify: false,
      style: [
        { selector: "node", style: {
            shape: (n) => NODE_SHAPE[n.data("role")] || "ellipse",
            "background-color": (n) => NODE_COLOR[n.data("role")] || "#2a3340",
            "border-width": 1.5 * nodeScale, "border-color": "#4a5a70",
            label: "data(label)", color: "#dbe4ee",
            "font-family": "Segoe UI, system-ui, sans-serif",
            "font-size": fontSize, "font-weight": 600,
            "text-valign": "center", "text-halign": "center",
            "text-outline-width": nodeScale > 1 ? 2 : 0,
            "text-outline-color": "#0d1117",
            width: (n) => (n.data("role").startsWith("PE") ? 62 : 52) * nodeScale,
            height: (n) => (n.data("role").startsWith("PE") ? 34 : 44) * nodeScale,
        }},
        { selector: "edge", style: {
            "curve-style": "bezier", "line-color": "#2ea043",
            width: 2 * nodeScale, opacity: 0.9,
        }},
        { selector: "edge.failed", style: {
            "line-style": "dashed", "line-color": "#f85149", opacity: 0.8,
            label: this.opts.failCross ? "✕" : "",
            color: "#f85149", "font-size": fontSize * 1.6, "font-weight": 700,
            "text-outline-width": 3, "text-outline-color": "#0d1117",
        }},
        { selector: "edge.lsp-path", style: {
            "line-color": "#58a6ff", opacity: 1,
            "z-index": 10, "overlay-color": "#58a6ff", "overlay-opacity": 0.25,
            "overlay-padding": 3 * nodeScale,
        }},
        { selector: "node.lsp-node", style: { "border-color": "#58a6ff", "border-width": 3 * nodeScale } },
        { selector: "edge.proposed-path", style: {
            "line-color": "#bc8cff", "line-style": "solid", opacity: 1,
            "z-index": 12, "overlay-color": "#bc8cff", "overlay-opacity": 0.3,
            "overlay-padding": 5 * nodeScale,
        }},
        { selector: "node.proposed-node", style: { "border-color": "#bc8cff", "border-width": 4 * nodeScale } },
      ],
      wheelSensitivity: 0.2,
    });

    if (this.opts.hover && this.hover) {
      this.cy.on("mouseover", "edge", (ev) => this._edgeHover(ev));
      this.cy.on("mouseover", "node", (ev) => this._nodeHover(ev));
      this.cy.on("mouseout", "edge,node", () => this.hover.classList.add("hidden"));
    }
    this.cy.on("tap", (ev) => {
      if (ev.target === this.cy) this.selectDemand(null);
    });
  }

  update(snapshot) {
    this.lastSnapshot = snapshot;
    // aggregate directed links -> undirected display values
    const agg = {};
    for (const dl of snapshot.links) {
      const a = agg[dl.link] || (agg[dl.link] = { dirs: [], up: true });
      a.dirs.push(dl);
      a.up = a.up && dl.up;
    }
    for (const [lid, a] of Object.entries(agg)) {
      const e = this.cy.getElementById(lid);
      if (!e.length) continue;
      const maxu = Math.max(...a.dirs.map((d) => d.utilization));
      a.maxu = maxu;
      e.data("agg", a);
      e.toggleClass("failed", !a.up);
      if (a.up) {
        e.style({
          "line-style": "solid",
          "line-color": utilColor(maxu),
          width: (1.5 + Math.min(6, a.dirs[0].capacity_mbps / 400)
                  + Math.min(4, 4 * maxu)) * this.opts.nodeScale,
        });
      }
    }
    this._applyPathHighlight();
  }

  /** The busiest live link as {id, util, label} — drives the KPI card. */
  busiestLink() {
    if (!this.lastSnapshot) return null;
    let best = null;
    for (const dl of this.lastSnapshot.links) {
      if (!dl.up) continue;
      if (!best || dl.utilization > best.utilization) best = dl;
    }
    return best ? { id: best.link, utilization: best.utilization,
                    label: linkLabel(best.link) } : null;
  }

  selectDemand(demandId) {
    this.selectedDemand = demandId;
    this._applyPathHighlight();
    if (this.onDemandSelect) this.onDemandSelect(demandId);
  }

  /** Highlight an explicit router chain in violet (advisor's proposed route). */
  showProposedPath(routers) {
    this.highlightPath = routers && routers.length ? routers : null;
    this._applyPathHighlight();
  }

  _edgesAlong(path) {
    const edges = [];
    for (let i = 0; i < path.length - 1; i++) {
      edges.push(this.cy.edges().filter((e) =>
        (e.source().id() === path[i] && e.target().id() === path[i + 1]) ||
        (e.source().id() === path[i + 1] && e.target().id() === path[i])));
    }
    return edges;
  }

  _applyPathHighlight() {
    this.cy.elements().removeClass("lsp-path lsp-node proposed-path proposed-node");
    if (this.selectedDemand && this.lastSnapshot) {
      const d = this.lastSnapshot.demands.find((x) => x.id === this.selectedDemand);
      if (d && !d.disconnected) {
        for (const r of d.current_path) this.cy.getElementById(r).addClass("lsp-node");
        for (const e of this._edgesAlong(d.current_path)) e.addClass("lsp-path");
      }
    }
    if (this.highlightPath) {
      for (const r of this.highlightPath) this.cy.getElementById(r).addClass("proposed-node");
      for (const e of this._edgesAlong(this.highlightPath)) e.addClass("proposed-path");
    }
  }

  _edgeHover(ev) {
    const a = ev.target.data("agg");
    const id = ev.target.id();
    const s = this.opts.scale;
    const lines = [`${linkLabel(id)}  —  ${a && !a.up ? "FAILED" : "up"}`,
                   `${linkTechnical(id)}`];
    if (a) {
      for (const d of a.dirs) {
        lines.push(
          `${city(d.src)} → ${city(d.dst)}  ${rate(d.load_mbps, s)} / ` +
          `${rate(d.capacity_mbps, s)}  (${util(d.utilization)})`,
          `   queue ${delay(d.queue_delay_ms)}  loss ${loss(d.loss_fraction)}` +
          `  LSPs ${d.n_lsps}  w=${d.weight}`);
      }
    }
    this._showHover(ev, lines.join("\n"));
  }

  _nodeHover(ev) {
    const id = ev.target.id();
    const snap = this.lastSnapshot;
    const lines = [`${city(id)}  (${id}, ${ev.target.data("role")})`];
    if (snap) {
      const r = snap.routers.find((x) => x.id === id);
      if (r) lines.push(`neighbours: ${r.neighbors.map(city).join(", ")}`);
      const lsps = snap.demands.filter((d) => !d.disconnected && d.current_path.includes(id));
      lines.push(`traffic flows through: ${lsps.length}`);
      for (const d of lsps.slice(0, 8))
        lines.push(`  ${demandLabel(d.src, d.dst, d.class)} — ` +
                   `${rate(d.volume_mbps, this.opts.scale)} (${d.id})`);
      if (lsps.length > 8) lines.push(`  … +${lsps.length - 8} more`);
    }
    this._showHover(ev, lines.join("\n"));
  }

  _showHover(ev, text) {
    if (!this.hover) return;
    const pos = ev.renderedPosition || ev.position;
    const rect = this.container.getBoundingClientRect();
    this.hover.textContent = text;
    this.hover.classList.remove("hidden");
    this.hover.style.left = Math.min(rect.left + pos.x + 14, window.innerWidth - 380) + "px";
    this.hover.style.top = (rect.top + pos.y + 14) + "px";
  }
}

export { pathLabel };
