# Mathematical formulation of the MPLS-TE decision problem (Environment V2)

This document states, from the code, exactly what decision problem the
repository solves. Every symbol is tied to a source location. Nothing here is
a design proposal: where the code is unusual, the code wins and the deviation
is noted. Source of truth: `mplssim/sim/engine_v2.py`, `mplssim/rl/env_v2.py`,
`mplssim/rl/reward_v2.py`, `mplssim/paths/candidates_v2.py`,
`mplssim/traffic/model.py`, `mplssim/sim/models.py`, and the YAML files under
`configs/`.

## 1. Network

* Directed graph $G=(V,E)$ with $|V|=18$ routers of roles
  $\{\mathrm{PE_{in}},\mathrm{PE_{out}},\mathrm{P},\mathrm{AGG}\}$ (4/4/8/2) and
  32 undirected links, each modelled as two directed links:
  $|E|=64$ (`configs/topology.yaml`, `mplssim/core/topology.py`).
* Each directed link $e$ has capacity $c_e\in\{250,500,1000,2000\}$ Mb/s (per
  direction), propagation delay $\tau_e$ (2–8 ms) and administrative weight
  $w_e$.
* Link availability $u_e(t)\in\{0,1\}$ is **exogenous**: set by scripted or
  randomized `link_down`/`link_up` events; both directions fail together.

## 2. Demands and candidate paths

* Demand set $\mathcal D$, $|\mathcal D|=17$; demand $d$ has ingress/egress PE,
  class $\kappa(d)$ with priority $\pi_d\in\{1,\dots,6\}$, delay SLA $\bar\delta_d$
  (60–400 ms), loss SLA $\bar\ell_d$ (0.2–5 %), protected flag
  $\mathrm{prot}_d$ (voice, critical: D1, D11, D14, D17), and base rate $b_d$.
  Priority weight $q_d=\pi_d/6$.
* **Candidate set** $\mathcal P_d=(p_{d,0},\dots,p_{d,3})$, fixed for the
  episode: Yen's $k$-shortest simple paths by admin cost, filtered to (i)
  intermediate routers in $\{\mathrm P,\mathrm{AGG}\}$, (ii) hop count
  $\le\lceil 2.5\,h^{\min}_d\rceil+1$, (iii) propagation delay
  $\le\min(1.75\,\tau^{\mathrm{ref}}_d,\ \tau^{\mathrm{ref}}_d+10\text{ ms})$,
  ordered by (admin cost, propagation, router tuple); exactly $k=4$ per demand
  (`candidates_v2.py`). $p_{d,0}$ is the role-valid administrative shortest
  path. Each path is a set of directed links $E(p)\subseteq E$.

## 3. Exogenous traffic

Offered rate of demand $d$ at simulated minute $t$ (`traffic/model.py`):
$$
v_d(t)=b_d\; f_{\kappa(d)}\big(h(t)\big)\; m\; \eta_d(t)\;\operatorname{clip}\!\big(1+n_d(t),0.4,2\big),
\qquad n_d(t)=0.9\,n_d(t-1)+\sigma\beta_{\kappa(d)}\,\varepsilon_d(t),
$$
with $f_\kappa$ a piecewise-linear diurnal profile of the hour $h(t)$, $m$ the
scenario multiplier, $\eta_d(t)$ the product of active burst / flash-crowd /
global-multiplier event factors, $\varepsilon\sim\mathcal N(0,1)$ i.i.d.,
$\sigma$ the scenario noise scale and $\beta_\kappa$ the class burstiness. The
noise advances once per one-minute micro-tick. **No term depends on routing**:
the exogenous process
$$\xi_t=\big(t,\ n(t),\ u(t),\ \text{materialized scenario events}\big)$$
evolves independently of every action. Two independent generators
(`SeedSequence([episode_seed,1|2])`) drive scenario materialization and AR
noise, which is what makes paired comparisons exact.

## 4. Controllable (routing) state

For each demand $d$ the engine keeps
$x_d\in\{0,..,3\}$ (current candidate), $z_d\in\{0,1\}$ (disconnected),
$\omega_d\in\{0,..,3\}$ (TE dwell remaining), $\chi_d\in\{-1,0,..,3\}$ (previous
TE path, for reversal detection), $\lambda_d$ (step of last TE change) and
$\alpha_d$ (path age). Write $y_t=(x,z,\omega,\chi,\lambda,\alpha)_t$. The full
Markov state is
$$
s_t=(y_t,\ \xi_t).
$$

## 5. Flow model (per micro-tick)

Given routes $x$ and offered rates $v$, routed rate $\tilde v_d=v_d(1-z_d)$.
*Gross* load $G_e=\sum_{d:\,e\in E(p_{d,x_d})}\tilde v_d$ (used for projections
and protected safety). *Carried* flow solves the fixed point (damping 0.5,
tolerance $10^{-10}$, cap 64 iterations; non-convergence aborts)
$$
\ell_e=L\!\left(\tfrac{\sum_{d}\mathrm{in}_{d,e}}{c_e}\right),\qquad
\mathrm{in}_{d,e_{h+1}}=\mathrm{in}_{d,e_h}(1-\ell_{e_h}),\ \mathrm{in}_{d,e_1}=\tilde v_d,
$$
with link loss curve (`sim/models.py`)
$$
L(\rho)=\begin{cases}0&\rho\le0.9\\ 0.02\left(\frac{\rho-0.9}{0.1}\right)^2&0.9<\rho\le1\\ 1-\frac{0.98}{\rho}&\rho>1.\end{cases}
$$
Utilization $\rho_e=\mathrm{in}_e/c_e$; queueing delay
$q(\rho)=\min\!\big(1.5\,\rho'/(1-\rho'),60\big)$ ms with $\rho'=\min(\rho,0.98)$;
demand delay $D_d=\sum_{e\in p}(\tau_e+q(\rho_e)+0.2)$; demand loss
$1-\prod_{e\in p}(1-\ell_e)$; delivered rate = final-hop output.

## 6. Decision process

**Epochs.** Decisions at 5-minute boundaries $t=0,\dots,T-1$; $T=$
duration/5 (288 for 24-h scenarios, 48–84 for the others). Each interval is
simulated as 5 one-minute micro-ticks.

**Actions.** $\mathcal A=\{0\}\cup\{1+4d+p\}$, $|\mathcal A|=69$; $0$ = no TE
change, $1+4d+p$ = "move demand $d$ to candidate $p$". At most one TE change
per interval network-wide.

**Action mask.** $m_t(0)=1$ always, and for $a=1+4d+p$
$$
m_t(a)=\mathbb 1\big[\,u_e(t)=1\ \forall e\in E(p_{d,p})\big]\;
\mathbb 1\big[\,p\ne x_d \vee z_d=1\big]\;
\mathbb 1\big[\,\omega_d=0\big]\;
\mathbb 1\Big[\neg\mathrm{prot}_d \ \vee\ \max_{e\in E(p_{d,p})}\tfrac{G_e - v_d\mathbb 1[e\in E(p_{d,x_d})](1-z_d)+v_d}{c_e}\le 1\Big].
$$
The legal set is $\mathcal A(s_t)=\{a: m_t(a)=1\}\ni 0$, so it is never empty.
The last factor makes the **feasible set of protected demands depend on where
other demands are routed** — the only channel through which one TE decision
changes which *other* decisions are legal.
(`engine_v2.validate_te_action`, vectorized as `te_action_matrix`; equality
of the two is tested exhaustively and re-audited in `docs/MASK_VALIDATION.md`.)

**Transition** (`env_v2.step`, `engine_v2.step_interval`):
1. If $a_t=1+4d+p$ is legal: $x_d\leftarrow p$, $z_d\leftarrow0$,
   $\omega_d\leftarrow3$, $\chi_d\leftarrow$ old path, $\lambda_d\leftarrow t$,
   $\alpha_d\leftarrow0$. The change carries traffic **in the same interval**
   (zero actuation latency).
2. For each of 5 micro-ticks: advance clock and AR noise; apply link events in
   $(t_{\text{old}},t_{\text{new}}]$; on a failure, *fast reroute* every
   demand whose path broke to its cheapest live candidate (or disconnect it),
   clearing $\chi_d$; on a recovery, restore disconnected demands; solve flow.
3. $\omega\leftarrow\max(\omega-1,0)$, $\alpha\leftarrow\alpha+1$.

Hence $P(s_{t+1}\mid s_t,a_t)=P_\xi(\xi_{t+1}\mid\xi_t)\;\mathbb 1[y_{t+1}=\Psi(y_t,a_t,\xi_{t:t+1})]$:
the action affects the future **only through the routing state $y$**, which
persists until changed.

**Termination.** Never; truncation at scenario end ($T$ steps).

## 7. Reward

Interval metrics (worst micro-tick for connectivity and max-utilization,
ratio of sums for delivery, means otherwise; `aggregate_interval`):
$$
U_t = 2\,\mathrm{DR}_t - 30\,\mathrm{PD}_t - 8\,\mathrm{UD}_t - 6\,\mathrm{SEV}_t
     - 2\,g(\rho^{\max}_t) - 6\,\mathrm{OV}_t,
\qquad g(\rho)=\log\!\Big(1+\max\big(0,\tfrac{\rho-0.7}{0.3}\big)\Big),
$$
with DR delivered ratio, PD/UD priority-weighted protected/unprotected
disconnected fractions, SEV the priority-weighted mean of
$\big(h(\Delta^{\text{delay}}_d)+2h(\Delta^{\text{loss}}_d)\big)/3$ with
$h(x)=x/(1+x)$ and $\Delta$ the normalized SLA excess, $\rho^{\max}$ the worst
link utilization, OV total overload over total capacity. The scalar reward is
$$
r_t = U_t + \underbrace{0.2\big(0.995\,\Phi(s_{t+1})-\Phi(s_t)\big)}_{F_t}
      - \underbrace{\mathbb 1[\text{accepted}]\big(0.08+0.30\,\mathrm{vol}_d+0.12\,\mathrm{div}_{d}+0.30\,\mathrm{rev}\big)}_{C^{\mathrm{TE}}_t}
      - 0.05\,\mathbb 1[\text{rejected}],
$$
where $\Phi(s)=\tanh(U(s)/10)$ is the same utility evaluated on the boundary
snapshot, $\mathrm{vol}_d$ the moved demand's share of total offered traffic,
$\mathrm{div}_d$ the Jaccard distance between old and new edge sets, and
$\mathrm{rev}$ indicates a return to the previous TE path within 6 steps. The
reward is exactly the ordered sum of 12 logged components
(`reward_v2.COMPONENT_ORDER`). Typical per-interval range: $U\approx 2$ in a
healthy network, down to $\approx -30$ during protected outages.

## 8. Which formal object is this?

* **Full state:** $(\mathcal S,\mathcal A,P,r,\gamma)$ with $s=(y,\xi)$ is a
  finite-horizon MDP with state-dependent action sets (masks). It is *not* a
  contextual bandit: $y_{t+1}$ depends on $a_t$, routes persist, the dwell
  $\omega$ makes a move lock the demand for 3 intervals, the reversal cost
  depends on $\chi,\lambda$, and the protected-class mask couples demands.
* **Agent's view:** the learners observe $o_t=\phi(s_t)\in[0,1]^{604}$
  (`env_v2._obs`): per-link utilization and status; per-demand offered volume,
  priority, protection, delay/loss relative to SLA, path age, dwell,
  disconnection, current and previous TE path (one-hot), candidate liveness,
  propagation and *projected* bottleneck after a move. It omits the clock,
  the AR noise state and future events. With respect to the exogenous
  component the problem is therefore a **POMDP**; with respect to the
  controllable component $y$, $o_t$ is essentially complete: $x,z,\omega,\chi$
  are observed directly, $\lambda$ through the path age (capped at 12 > the
  6-step reversal window), and the protected-class mask through the projected
  bottleneck block.
* **Constraints** are hard (masked), not penalized: this is not a constrained
  MDP in the Lagrangian sense; the only soft "safety" terms are the large
  disconnection coefficients.

### 8.1 What the contextual bandit discards

The masked bandit (`experiments/masked_bandit.py`) fits
$\hat r_\theta(o,a)\approx\mathbb E[r_t\mid o_t=o,a_t=a]$ and acts
$a=\arg\max_{a\in\mathcal A(s)}\hat r_\theta(o,a)$. Relative to the MDP it
discards $\gamma\,\mathbb E[V(s_{t+1})]$, i.e. (i) the persistence of the new
configuration beyond the first interval, (ii) the opportunity cost of the
3-interval dwell lock, (iii) future reversal costs, and (iv) changes in other
demands' legal sets. It keeps the first interval of the move's effect,
because effect and decision coincide in time (§6, step 1), and it keeps the
one-step shaping difference $F_t$.

### 8.2 When is acting on the immediate reward nearly optimal? (inference, not a theorem)

Because $\xi$ is action-independent and $y$ persists under no-op, the
$H$-interval value of a first action $a$ followed by no-op is
$$
G_H(s,a)=-C^{\mathrm{TE}}(a)+\sum_{\tau=0}^{H-1}\mathbb E\big[U(x^{a},\xi_{t+\tau})\big]+\text{shaping},
$$
where $x^a$ is the configuration after $a$ (ignoring failures that force FRR).
The myopic choice $\arg\max_a G_1$ coincides with $\arg\max_a G_H$ when the
ranking of reachable configurations by per-interval utility is stable over the
next $H$ intervals and one-off move costs do not reverse that ranking. With
slowly varying diurnal traffic (AR(1) $\phi=0.9$ per minute, i.e. ≈0.59 per
interval) and move costs small relative to congestion penalties, this is
plausible for most states, and it fails around abrupt exogenous changes
(burst onsets, failures) and for moves whose benefit is less than their cost
within one interval but larger over several. Whether it holds is an
empirical question; `docs/SEQUENTIALITY_AUDIT.md` measures it with clairvoyant
rollouts.

## 9. Learning objective and evaluation statistic

PPO maximizes $\mathbb E\big[\sum_t\gamma^t r_t\big]$ with $\gamma=0.995$
(effective horizon $1/(1-\gamma)=200$ intervals ≈ 16.7 h). The bandit
maximizes $\mathbb E[r_t]$ per step ($\gamma=0$). Both are evaluated by the
undiscounted **operational return** $R=\sum_{t<T} r_t$ of a deterministic
(greedy) policy on fixed scenario/seed pairs; comparisons are paired by
(scenario, seed) because $\xi$ is identical across controllers.

## 10. Variant used for the controlled-coupling experiment

`mplssim/study/variants.py` (`DelayedTeEnvV2`, identity
`mpls-te-v2.0.0+delay-L`) changes only step 1 of §6: a legal request at $t$ is
charged $C^{\mathrm{TE}}$ at $t$ and applied at boundary $t+L$ if still legal
(else cancelled); a demand with a pending request is masked; the observation
appends the pending target (one-hot) and the remaining delay fraction
(689 features). With $L=0$ it is bit-identical to V2 (tested). For $L\ge1$,
$\mathbb E[r_t\mid s_t,a_t]$ no longer contains any effect of $a_t$, only its cost.
