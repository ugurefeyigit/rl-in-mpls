"""V2 candidate LSP path generation (role-valid, deterministically ordered).

Governing document: docs/RL_ENVIRONMENT_V2_SPEC.md, "Candidate generation".

V1 (:mod:`mplssim.paths.candidates`) enumerates Yen's weighted shortest simple
paths and keeps the first ``k`` that satisfy a hop cap. That is role-blind, so
two of the 68 shipped V1 candidates transit a PE (D10 candidate 3 through the
egress PE7, D16 candidate 3 through the ingress PE3) — not credible core LSPs.
It also inherits NetworkX's enumeration order for equal-cost paths, which makes
the action index of a tied candidate an accident of the graph traversal.

V2 enforces, in this order:

1. loop-free (guaranteed by simple-path enumeration, re-checked anyway);
2. endpoints match the demand;
3. every intermediate router role is ``P`` or ``AGG``;
4. hop count within the existing cap ``ceil(min_hops * max_hop_factor) + 1``;
5. propagation delay no greater than
   ``min(1.75*shortest_prop, shortest_prop + 10 ms)``;
6. unique router sequence;
7. ascending ``(admin_cost, propagation_delay, router_tuple)``;
8. exactly ``k`` candidates, or startup fails.

``shortest_prop`` is the propagation delay of the *administratively shortest
role-valid path* — the first path surviving filters 1-4 and 6 in Yen order.
Taking the reference before filter 5 is applied keeps the definition
non-circular and guarantees that the reference path always satisfies its own
bound, so candidate 0 is always the role-valid administrative shortest path.

Enumeration is exhaustive with respect to the admin-cost tie group at the
``k``-th position: scanning continues until a Yen path is strictly more
expensive than the ``k``-th kept candidate. Because Yen yields paths in
non-decreasing admin cost, no later path can then displace one already kept, so
the sorted top-``k`` does not depend on how NetworkX ordered equal-cost paths.

Consequence for action semantics: applying rule 7 reorders equal-cost
candidates relative to V1 for D4, D5, D7, D10, D13 and D15 in addition to
replacing the two PE-transit paths. V2 metadata therefore stores the complete
ordered router-sequence table — the action version alone cannot validate a
checkpoint. See docs/RL_ENVIRONMENT_V2_SPEC.md, "Exact action schema".
"""

from __future__ import annotations

import math
from itertools import islice

import networkx as nx

from mplssim.core.topology import Topology
from mplssim.paths.candidates import path_admin_cost

#: Router roles allowed as an *intermediate* hop on a V2 candidate.
TRANSIT_ROLES = frozenset({"P", "AGG"})

#: Hard cap on how many Yen paths are enumerated per demand. Generation fails
#: closed rather than silently truncating: the shipped topology needs at most 8.
MAX_ENUMERATED_PATHS = 200

#: Candidate paths depend only on (topology, k, hop factor, delay bound), never
#: on dynamic state. Kept separate from V1's ``_CANDIDATE_CACHE`` so importing
#: or exercising V2 can never populate or mutate a V1 cache.
_CANDIDATE_CACHE_V2: dict[tuple, tuple[tuple[str, ...], ...]] = {}


class CandidatePathError(ValueError):
    """Raised when V2 candidate generation cannot satisfy the specification."""


def path_propagation_ms(topo: Topology, routers: tuple[str, ...]) -> float:
    """One-way propagation delay along an ordered router sequence."""
    return sum(
        topo.dlink_by_pair[(routers[i], routers[i + 1])].delay_ms
        for i in range(len(routers) - 1)
    )


def path_directed_edges(topo: Topology, routers: tuple[str, ...]) -> frozenset[int]:
    """Directed-link indices traversed by a router sequence, as a set."""
    return frozenset(topo.path_dlink_indices(routers))


def is_role_valid(topo: Topology, routers: tuple[str, ...]) -> bool:
    """True when every intermediate router is a core (``P``) or aggregation node.

    Endpoints are the demand's ingress/egress PEs and are exempt; only transit
    matters. A two-router path has no intermediates and is trivially valid.
    """
    return all(topo.routers[r].role in TRANSIT_ROLES for r in routers[1:-1])


def is_loop_free(routers: tuple[str, ...]) -> bool:
    return len(set(routers)) == len(routers)


def hop_cap_for(topo: Topology, src: str, dst: str, max_hop_factor: float) -> int:
    """The existing V1 hop cap, reused unchanged by V2."""
    min_hops = nx.shortest_path_length(topo.graph, src, dst)
    return math.ceil(min_hops * max_hop_factor) + 1


def propagation_bound(shortest_prop: float, delay_factor: float,
                      delay_additive_ms: float) -> float:
    """``min(factor*shortest_prop, shortest_prop + additive_ms)``."""
    return min(delay_factor * shortest_prop, shortest_prop + delay_additive_ms)


def generate_candidate_paths_v2(
    topo: Topology,
    src: str,
    dst: str,
    k: int = 4,
    max_hop_factor: float = 2.5,
    delay_factor: float = 1.75,
    delay_additive_ms: float = 10.0,
) -> tuple[tuple[str, ...], ...]:
    """Return exactly ``k`` role-valid candidate paths, deterministically ordered.

    Raises :class:`CandidatePathError` if fewer than ``k`` paths satisfy every
    rule — V2 never pads, never relaxes a filter and never returns a short list.
    """
    if src == dst:
        raise CandidatePathError(f"demand endpoints coincide ({src})")
    hop_cap = hop_cap_for(topo, src, dst, max_hop_factor)

    reference_prop: float | None = None
    bound: float | None = None
    seen: set[tuple[str, ...]] = set()
    kept: list[tuple[float, float, tuple[str, ...]]] = []
    exhausted = False
    n_enumerated = 0

    gen = nx.shortest_simple_paths(topo.graph, src, dst, weight="weight")
    for raw in islice(gen, MAX_ENUMERATED_PATHS):
        n_enumerated += 1
        routers = tuple(raw)
        cost = path_admin_cost(topo, routers)

        # Yen yields non-decreasing admin cost, so once a path is strictly more
        # expensive than the k-th kept candidate the sorted top-k is final.
        if len(kept) >= k and cost > sorted(kept)[k - 1][0]:
            exhausted = True
            break

        if routers[0] != src or routers[-1] != dst:
            continue
        if not is_loop_free(routers):
            continue
        if len(routers) - 1 > hop_cap:
            continue
        if not is_role_valid(topo, routers):
            continue
        if routers in seen:
            continue
        seen.add(routers)

        prop = path_propagation_ms(topo, routers)
        if reference_prop is None:
            # Administratively shortest role-valid path: fixes the delay bound
            # before the bound is used, so the reference always passes it.
            reference_prop = prop
            bound = propagation_bound(reference_prop, delay_factor, delay_additive_ms)
        if prop > bound:
            continue
        kept.append((cost, prop, routers))
    else:
        # No cost-based break. The enumeration is complete only if the
        # generator ran dry before the cap, i.e. every simple path was seen.
        exhausted = n_enumerated < MAX_ENUMERATED_PATHS

    kept.sort()
    if len(kept) < k:
        raise CandidatePathError(
            f"{src}->{dst}: only {len(kept)} role-valid candidate path(s) within "
            f"hop cap {hop_cap} and propagation bound "
            f"{bound if bound is not None else float('nan')} ms; {k} required"
        )
    if not exhausted:
        raise CandidatePathError(
            f"{src}->{dst}: enumeration cap of {MAX_ENUMERATED_PATHS} Yen paths "
            f"reached before the admin-cost tie group at position {k} was "
            f"exhausted; the candidate table would depend on enumeration order"
        )
    return tuple(routers for _, _, routers in kept[:k])


def cached_candidate_paths_v2(
    topo: Topology,
    src: str,
    dst: str,
    k: int = 4,
    max_hop_factor: float = 2.5,
    delay_factor: float = 1.75,
    delay_additive_ms: float = 10.0,
) -> tuple[tuple[str, ...], ...]:
    """:func:`generate_candidate_paths_v2` memoized per (topology, parameters)."""
    key = (id(topo), src, dst, k, max_hop_factor, delay_factor, delay_additive_ms)
    cands = _CANDIDATE_CACHE_V2.get(key)
    if cands is None:
        cands = generate_candidate_paths_v2(
            topo, src, dst, k=k, max_hop_factor=max_hop_factor,
            delay_factor=delay_factor, delay_additive_ms=delay_additive_ms,
        )
        _CANDIDATE_CACHE_V2[key] = cands
    return cands


def validate_candidate_table(
    topo: Topology,
    table: dict[str, tuple[tuple[str, ...], ...]],
    demands,
    k: int = 4,
    max_hop_factor: float = 2.5,
    delay_factor: float = 1.75,
    delay_additive_ms: float = 10.0,
) -> None:
    """Re-check every specification rule against an already-built table.

    Used at engine startup and by ``scripts/validate_env_v2.py`` so a cache hit
    is held to the same standard as a fresh generation.
    """
    for d in demands:
        cands = table[d.id]
        if len(cands) != k:
            raise CandidatePathError(f"{d.id}: {len(cands)} candidates, {k} required")
        if len(set(cands)) != len(cands):
            raise CandidatePathError(f"{d.id}: duplicate candidate router sequence")
        hop_cap = hop_cap_for(topo, d.src, d.dst, max_hop_factor)
        bound = propagation_bound(path_propagation_ms(topo, cands[0]),
                                  delay_factor, delay_additive_ms)
        keys = []
        for p_idx, routers in enumerate(cands):
            where = f"{d.id} candidate {p_idx}"
            if routers[0] != d.src or routers[-1] != d.dst:
                raise CandidatePathError(f"{where}: endpoints {routers[0]}->{routers[-1]}"
                                         f" != {d.src}->{d.dst}")
            if not is_loop_free(routers):
                raise CandidatePathError(f"{where}: not loop-free")
            for i in range(len(routers) - 1):
                if (routers[i], routers[i + 1]) not in topo.dlink_by_pair:
                    raise CandidatePathError(
                        f"{where}: no directed link {routers[i]}->{routers[i + 1]}")
            if not is_role_valid(topo, routers):
                bad = [r for r in routers[1:-1] if topo.routers[r].role not in TRANSIT_ROLES]
                raise CandidatePathError(f"{where}: PE transit through {bad}")
            if len(routers) - 1 > hop_cap:
                raise CandidatePathError(f"{where}: {len(routers) - 1} hops exceeds cap {hop_cap}")
            prop = path_propagation_ms(topo, routers)
            if prop > bound:
                raise CandidatePathError(
                    f"{where}: propagation {prop} ms exceeds bound {bound} ms")
            keys.append((path_admin_cost(topo, routers), prop, routers))
        if keys != sorted(keys):
            raise CandidatePathError(
                f"{d.id}: candidates are not ascending in "
                f"(admin_cost, propagation_delay, router_tuple)")


def build_candidate_table(
    topo: Topology,
    demands,
    k: int = 4,
    max_hop_factor: float = 2.5,
    delay_factor: float = 1.75,
    delay_additive_ms: float = 10.0,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Full ``demand_id -> ordered candidate router sequences`` table.

    Generation is deterministic, so repeated calls return byte-identical
    tables; :func:`validate_candidate_table` re-asserts every rule afterwards.
    """
    table = {
        d.id: cached_candidate_paths_v2(
            topo, d.src, d.dst, k=k, max_hop_factor=max_hop_factor,
            delay_factor=delay_factor, delay_additive_ms=delay_additive_ms,
        )
        for d in demands
    }
    validate_candidate_table(topo, table, demands, k=k, max_hop_factor=max_hop_factor,
                             delay_factor=delay_factor,
                             delay_additive_ms=delay_additive_ms)
    return table
