"""Per-interval min-max-utilization MILP baseline with single-move tracking.

Each control interval:

1. Solve, with the current offered volumes ``v`` and live candidates, the
   path-based multicommodity problem

       min_{y, U}  U + eps * sum_d (1 - y_{d, x_d})
       s.t.        sum_p y_{d,p} = 1                       for every connected demand d
                   sum_{d,p: e in p} v_d y_{d,p} <= U c_e  for every directed link e
                   y_{d,p} = 0 if candidate p traverses a failed link
                   y binary,

   on *gross* loads (no loss), where ``x_d`` is the current path and the
   ``eps`` term breaks ties in favour of the current configuration (so equal
   optima do not cause churn). This is the classical "minimize maximum link
   utilization" TE objective restricted to the 4 candidate LSPs per demand.
2. If the target improves the current configuration's gross MLU by at least
   ``min_gain``, submit the single legal move towards the target with the
   largest moved volume (V2 permits one TE change per interval); otherwise
   no-op. ``select="smallest"`` instead moves the smallest-volume mismatched
   demand, which is cheaper under V2's volume-proportional move cost.

It uses only current telemetry: a myopic, optimization-based controller in
the spirit of one-shot TE (DOTE/Teal re-solve every interval), adapted to
incremental reconfiguration. It does not model loss, delay SLAs or the
protected-class rule in the optimization; the environment's mask still
enforces the latter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


class MilpTargetPolicy:
    name = "milp_track"

    def __init__(self, eps: float = 1e-3, min_gain: float = 0.02,
                 time_limit: float = 5.0, select: str = "largest") -> None:
        if select not in ("largest", "smallest"):
            raise ValueError(select)
        self.eps = float(eps)
        self.min_gain = float(min_gain)
        self.select = select
        self.time_limit = float(time_limit)
        self.solves = 0
        self.failures = 0

    def reset(self) -> None:
        pass

    def target(self, eng: Any) -> tuple[np.ndarray, float]:
        n_d, k, n_e = eng.n_demands, eng.k, len(eng.capacity)
        live = eng.candidate_available_matrix()
        connected = ~eng.disconnected
        v = eng.demand_offered
        nvar = n_d * k + 1
        cost = np.zeros(nvar)
        cost[-1] = 1.0
        for d in range(n_d):
            cost[d * k + int(eng.current_path[d])] -= self.eps if connected[d] else 0.0
        A_eq = lil_matrix((n_d, nvar))
        b_eq = np.zeros(n_d)
        for d in range(n_d):
            if connected[d] and live[d].any():
                A_eq[d, d * k:(d + 1) * k] = 1.0
                b_eq[d] = 1.0
        A_cap = lil_matrix((n_e, nvar))
        for d in range(n_d):
            for p in range(k):
                for e in eng._cand_links[d][p]:
                    A_cap[e, d * k + p] += v[d]
        A_cap[:, -1] = -eng.capacity.reshape(-1, 1)
        ub = np.ones(nvar)
        ub[:-1] = live.ravel().astype(float)
        ub[-1] = np.inf
        res = milp(cost, integrality=np.r_[np.ones(n_d * k), 0],
                   bounds=Bounds(np.zeros(nvar), ub),
                   constraints=[LinearConstraint(A_eq.tocsr(), b_eq, b_eq),
                                LinearConstraint(A_cap.tocsr(), -np.inf, 0.0)],
                   options={"time_limit": self.time_limit})
        self.solves += 1
        if res.x is None:
            self.failures += 1
            return eng.current_path.copy(), float("inf")
        y = res.x[:-1].reshape(n_d, k)
        return np.argmax(y, axis=1), float(res.x[-1])

    def act(self, observation: np.ndarray, mask: np.ndarray, env: Any) -> int:
        eng = env.eng
        target, u_star = self.target(eng)
        u_now = float(np.max(eng.gross_link_load / eng.capacity))
        if not np.isfinite(u_star) or u_now - u_star < self.min_gain:
            return 0
        sign = 1.0 if self.select == "largest" else -1.0
        best, best_key = 0, -np.inf
        for d in np.flatnonzero((target != eng.current_path) & ~eng.disconnected):
            a = 1 + int(d) * eng.k + int(target[d])
            key = sign * float(eng.demand_offered[d])
            if mask[a] and key > best_key:
                best, best_key = a, key
        return best
