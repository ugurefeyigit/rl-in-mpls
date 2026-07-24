"""Documented analytic approximations for queueing delay and loss.

The simulation is FLOW-LEVEL: no packets, no per-packet queues. Delay and loss
are analytic functions of directed-link utilization rho = offered / capacity.

Queueing delay (per link, ms):
    d_q(rho) = Q_COEFF * rho' / (1 - rho'),   rho' = min(rho, RHO_CLAMP)
    capped at Q_MAX_MS. Shape follows M/M/1 waiting time; the clamp and cap
    keep values finite and bounded — a documented approximation, not a claim
    of packet-level fidelity (see docs/REPORT.md, Limitations).

Loss fraction (per link):
    rho <= LOSS_ONSET            : 0
    LOSS_ONSET < rho <= 1        : SOFT_LOSS_MAX * ((rho - ONSET)/(1 - ONSET))^2
                                   (early queue-tail drops, up to 2%)
    rho > 1                      : 1 - (1 - SOFT_LOSS_MAX)/rho
                                   (all offered load above ~capacity is dropped)

End-to-end demand loss = 1 - prod(1 - loss_link) over the LSP's links.
Link offered load is computed from full offered demand volumes (upstream drops
are not cascaded downstream) — a conservative, documented simplification.
"""

from __future__ import annotations

import numpy as np

Q_COEFF_MS = 1.5     # queueing delay scale per link
RHO_CLAMP = 0.98     # utilization clamp inside the M/M/1 term
Q_MAX_MS = 60.0      # hard cap on per-link queueing delay
PROC_DELAY_MS = 0.2  # fixed per-hop processing delay
LOSS_ONSET = 0.90    # utilization where soft loss begins
SOFT_LOSS_MAX = 0.02 # soft loss at exactly 100% utilization
CONGESTION_UTIL = 0.90  # a link at/above this counts as "congested"


def queue_delay_ms(util: np.ndarray) -> np.ndarray:
    rho = np.minimum(util, RHO_CLAMP)
    return np.minimum(Q_COEFF_MS * rho / (1.0 - rho), Q_MAX_MS)


def loss_fraction(util: np.ndarray) -> np.ndarray:
    loss = np.zeros_like(util)
    soft = (util > LOSS_ONSET) & (util <= 1.0)
    loss[soft] = SOFT_LOSS_MAX * ((util[soft] - LOSS_ONSET) / (1.0 - LOSS_ONSET)) ** 2
    over = util > 1.0
    loss[over] = 1.0 - (1.0 - SOFT_LOSS_MAX) / util[over]
    return loss


def jain_fairness(x: np.ndarray) -> float:
    """Jain's fairness index over link utilizations (1 = perfectly even)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or float(np.sum(x * x)) == 0.0:
        return 1.0
    return float(np.sum(x) ** 2 / (x.size * np.sum(x * x)))
