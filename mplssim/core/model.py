"""Core domain dataclasses for the MPLS-TE simulation.

Everything here is a static *definition*; dynamic state (loads, failures,
current paths) lives in :class:`mplssim.sim.engine.SimulationEngine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Router:
    """A network node. Roles: PE_IN (ingress LER), PE_OUT (egress LER), P (core LSR), AGG."""

    id: str
    role: str
    x: float
    y: float


@dataclass(frozen=True)
class LinkDef:
    """An undirected link definition from topology.yaml (expanded to two DirectedLinks)."""

    id: str
    a: str
    z: str
    capacity_mbps: float
    delay_ms: float
    weight: float


@dataclass(frozen=True)
class DirectedLink:
    """One direction of a physical link. All load accounting is per direction.

    ``index`` is the position of this link in every per-link numpy array used
    by the engine and in the observation vector (ordering is stable and equals
    the order of links in topology.yaml, A->Z direction first).
    """

    id: str            # e.g. "L11:P2>P5"
    undirected_id: str  # e.g. "L11"
    src: str
    dst: str
    capacity_mbps: float
    delay_ms: float
    weight: float
    index: int


@dataclass(frozen=True)
class TrafficClass:
    """Service class with SLA thresholds (see configs/traffic_classes.yaml)."""

    name: str
    priority: int
    max_latency_ms: float
    max_loss_pct: float
    protected: bool
    profile: str
    burstiness: float
    color: str


@dataclass(frozen=True)
class Demand:
    """A PE-to-PE traffic demand (one FEC, mapped to exactly one LSP at a time).

    ``index`` is the demand's stable position in observation/action spaces
    (equals its order in traffic_classes.yaml).
    """

    id: str
    src: str
    dst: str
    cls: TrafficClass
    base_mbps: float
    index: int
    candidate_paths: tuple[tuple[str, ...], ...] = field(default=())

    def path_links(self, topo: "TopologyLike", path_idx: int) -> tuple[int, ...]:
        """Directed-link indices for candidate path ``path_idx``."""
        routers = self.candidate_paths[path_idx]
        return tuple(
            topo.dlink_by_pair[(routers[i], routers[i + 1])].index
            for i in range(len(routers) - 1)
        )


class TopologyLike:
    """Structural protocol for objects exposing ``dlink_by_pair`` (avoids a cyclic import)."""

    dlink_by_pair: dict[tuple[str, str], DirectedLink]
