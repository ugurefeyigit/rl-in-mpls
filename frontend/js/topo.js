// Topology view: Cytoscape canvases (1 in single mode, 2 in compare mode).
// Edges are undirected physical links; color = worst-direction utilization,
// width = capacity, dashed red = failed. Selected LSP overlays glow.

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
  constructor(containerId, hovercardEl) {
    this.container = document.getElementById(containerId);
    this.hover = hovercardEl;
    this.cy = null;
    this.selectedDemand = null;
    this.lastSnapshot = null;
    this.onDemandSelect = null;
  }

  init(topology) {
    const elements = [];
    for (const r of topology.routers) {
      elements.push({ data: { id: r.id, label: r.id, role: r.role },
                      position: { x: r.x, y: r.y } });
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
            "border-width": 1.5, "border-color": "#4a5a70",
            label: "data(label)", color: "#dbe4ee",
            "font-family": "Consolas, monospace", "font-size": 11,
            "text-valign": "center", "text-halign": "center",
            width: (n) => n.data("role").startsWith("PE") ? 46 : 40,
            height: (n) => n.data("role").startsWith("PE") ? 30 : 40,
        }},
        { selector: "edge", style: {
            "curve-style": "bezier", "line-color": "#2ea043",
            width: 2, opacity: 0.9,
        }},
        { selector: "edge.failed", style: {
            "line-style": "dashed", "line-color": "#f85149", opacity: 0.7,
        }},
        { selector: "edge.lsp-path", style: {
            "line-color": "#58a6ff", opacity: 1,
            "z-index": 10, "overlay-color": "#58a6ff", "overlay-opacity": 0.25,
            "overlay-padding": 3,
        }},
        { selector: "node.lsp-node", style: { "border-color": "#58a6ff", "border-width": 3 } },
      ],
      wheelSensitivity: 0.2,
    });

    this.cy.on("mouseover", "edge", (ev) => this._edgeHover(ev));
    this.cy.on("mouseover", "node", (ev) => this._nodeHover(ev));
    this.cy.on("mouseout", "edge,node", () => this.hover.classList.add("hidden"));
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
      e.data("agg", a);
      e.toggleClass("failed", !a.up);
      if (a.up) {
        e.style({
          "line-style": "solid",
          "line-color": utilColor(maxu),
          width: 1.5 + Math.min(6, a.dirs[0].capacity_mbps / 400)
                 + Math.min(4, 4 * maxu),
        });
      }
    }
    this._applyPathHighlight();
  }

  selectDemand(demandId) {
    this.selectedDemand = demandId;
    this._applyPathHighlight();
    if (this.onDemandSelect) this.onDemandSelect(demandId);
  }

  _applyPathHighlight() {
    this.cy.elements().removeClass("lsp-path lsp-node");
    if (!this.selectedDemand || !this.lastSnapshot) return;
    const d = this.lastSnapshot.demands.find((x) => x.id === this.selectedDemand);
    if (!d || d.disconnected) return;
    const path = d.current_path;
    for (let i = 0; i < path.length; i++) {
      this.cy.getElementById(path[i]).addClass("lsp-node");
      if (i < path.length - 1) {
        // find the undirected link connecting path[i] and path[i+1]
        const edge = this.cy.edges().filter((e) =>
          (e.source().id() === path[i] && e.target().id() === path[i + 1]) ||
          (e.source().id() === path[i + 1] && e.target().id() === path[i]));
        edge.addClass("lsp-path");
      }
    }
  }

  _edgeHover(ev) {
    const a = ev.target.data("agg");
    if (!a) return;
    const lines = [`link ${ev.target.id()}  ${a.up ? "UP" : "FAILED"}`];
    for (const d of a.dirs) {
      lines.push(
        `${d.src}>${d.dst}  ${d.load_mbps.toFixed(0)}/${d.capacity_mbps} Mbps ` +
        `(${(d.utilization * 100).toFixed(1)}%)`,
        `   qdelay ${d.queue_delay_ms.toFixed(2)} ms  loss ${(d.loss_fraction * 100).toFixed(2)}%` +
        `  lsps ${d.n_lsps}  w=${d.weight}`);
    }
    this._showHover(ev, lines.join("\n"));
  }

  _nodeHover(ev) {
    const id = ev.target.id();
    const snap = this.lastSnapshot;
    const lines = [`${id}  (${ev.target.data("role")})`];
    if (snap) {
      const r = snap.routers.find((x) => x.id === id);
      if (r) lines.push(`neighbors: ${r.neighbors.join(" ")}`);
      const lsps = snap.demands.filter((d) => !d.disconnected && d.current_path.includes(id));
      lines.push(`LSPs through: ${lsps.length}`);
      for (const d of lsps.slice(0, 8))
        lines.push(`  ${d.id} ${d.class} ${d.volume_mbps.toFixed(0)} Mbps`);
      if (lsps.length > 8) lines.push(`  … +${lsps.length - 8} more`);
    }
    this._showHover(ev, lines.join("\n"));
  }

  _showHover(ev, text) {
    const pos = ev.renderedPosition || ev.position;
    const rect = this.container.getBoundingClientRect();
    this.hover.textContent = text;
    this.hover.classList.remove("hidden");
    this.hover.style.left = Math.min(rect.left + pos.x + 14, window.innerWidth - 360) + "px";
    this.hover.style.top = (rect.top + pos.y + 14) + "px";
  }
}
