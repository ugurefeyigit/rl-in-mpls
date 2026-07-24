"""Topology loading and graph construction."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import yaml

from mplssim.core.model import DirectedLink, LinkDef, Router

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class Topology:
    """Immutable topology: routers, undirected link defs, directed links, graph.

    Directed links are ordered: for each YAML link (in file order) the A->Z
    direction comes first, then Z->A. That ordering defines the per-link axis
    of every numpy array and of the RL observation vector.
    """

    def __init__(self, routers: list[Router], link_defs: list[LinkDef]) -> None:
        self.routers: dict[str, Router] = {r.id: r for r in routers}
        self.link_defs: dict[str, LinkDef] = {l.id: l for l in link_defs}

        self.dlinks: list[DirectedLink] = []
        self.dlink_by_pair: dict[tuple[str, str], DirectedLink] = {}
        for ld in link_defs:
            for src, dst in ((ld.a, ld.z), (ld.z, ld.a)):
                dl = DirectedLink(
                    id=f"{ld.id}:{src}>{dst}",
                    undirected_id=ld.id,
                    src=src,
                    dst=dst,
                    capacity_mbps=ld.capacity_mbps,
                    delay_ms=ld.delay_ms,
                    weight=ld.weight,
                    index=len(self.dlinks),
                )
                self.dlinks.append(dl)
                self.dlink_by_pair[(src, dst)] = dl

        self.n_dlinks = len(self.dlinks)

        # Directed graph over all links (weights = admin metric) for path finding.
        self.graph = nx.DiGraph()
        for r in routers:
            self.graph.add_node(r.id)
        for dl in self.dlinks:
            self.graph.add_edge(dl.src, dl.dst, weight=dl.weight, dlink=dl)

        if not nx.is_strongly_connected(self.graph):
            raise ValueError("Topology graph is not strongly connected")

    def neighbors(self, router_id: str) -> list[str]:
        return sorted(self.graph.successors(router_id))

    def path_dlink_indices(self, routers: tuple[str, ...]) -> tuple[int, ...]:
        """Directed-link indices along an ordered router sequence (validates adjacency)."""
        return tuple(
            self.dlink_by_pair[(routers[i], routers[i + 1])].index
            for i in range(len(routers) - 1)
        )


def load_topology(path: Path | None = None) -> Topology:
    """Load configs/topology.yaml into a Topology object."""
    path = path or CONFIG_DIR / "topology.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    routers = [Router(**r) for r in raw["routers"]]
    link_defs = [LinkDef(**l) for l in raw["links"]]
    return Topology(routers, link_defs)
