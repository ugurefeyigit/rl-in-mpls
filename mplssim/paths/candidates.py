"""Candidate LSP path generation.

Uses Yen's k-shortest simple paths (networkx ``shortest_simple_paths``) over
administrative weights, with a hop-count cap so candidates stay operationally
sane (no absurd detours). Candidates are loop-free by construction (simple
paths) and are ordered by ascending admin cost, so index 0 is always the
IGP-shortest path — the static baseline pins each demand there.
"""

from __future__ import annotations

import math
from itertools import islice

import networkx as nx

from mplssim.core.topology import Topology


def generate_candidate_paths(
    topo: Topology,
    src: str,
    dst: str,
    k: int = 4,
    max_hop_factor: float = 2.5,
) -> tuple[tuple[str, ...], ...]:
    """Return up to ``k`` loop-free candidate paths src->dst as router tuples.

    The hop cap is ``ceil(min_hops * max_hop_factor) + 1``; paths longer than
    that are skipped. Always returns at least one path (the shortest).
    """
    min_hops = nx.shortest_path_length(topo.graph, src, dst)  # unweighted hops
    hop_cap = math.ceil(min_hops * max_hop_factor) + 1

    paths: list[tuple[str, ...]] = []
    # Scan more than k simple paths because some get rejected by the hop cap.
    for path in islice(nx.shortest_simple_paths(topo.graph, src, dst, weight="weight"), 40):
        if len(path) - 1 <= hop_cap:
            paths.append(tuple(path))
        if len(paths) >= k:
            break
    if not paths:  # hop cap can never exclude the shortest path, but be safe
        paths.append(tuple(nx.shortest_path(topo.graph, src, dst, weight="weight")))
    return tuple(paths)


def path_admin_cost(topo: Topology, routers: tuple[str, ...]) -> float:
    return sum(
        topo.dlink_by_pair[(routers[i], routers[i + 1])].weight
        for i in range(len(routers) - 1)
    )
