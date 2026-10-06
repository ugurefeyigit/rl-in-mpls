# Algorithms

Precise descriptions of every policy evaluated, written from the code. Notation
follows `docs/MATHEMATICAL_FORMULATION.md`: observation $o_t\in[0,1]^{604}$,
legal set $\mathcal A(s_t)$ given by the mask $m_t$, reward $r_t$.

| Policy | Learns | Uses future value | Information at decision time | Source |
|---|---|---|---|---|
| MaskablePPO | yes | yes ($\gamma=0.995$) | $o_t$, $m_t$ | `sb3_contrib.MaskablePPO`, `experiments/trainers_v2.py` |
| Masked contextual bandit | yes | **no** ($\gamma=0$) | $o_t$, $m_t$ | `experiments/masked_bandit.py` |
| Masked Q-learner ($\gamma>0$) | yes | yes (bootstrapped) | $o_t$, $m_t$ | `study/learners.py` |
| Static | no | no | topology, link status | `baselines/controllers.py:StaticShortestPathController` |
| Greedy | no | no | current link utilization | `…:GreedyUtilizationController` |
| CSPF | no | no | current volumes, admin costs | `…:CspfController` |
| No-op / random-valid | no | no | $m_t$ | `study/episode.py` |
| Clairvoyant H-step oracle | no | $H$ intervals, **exact future traffic** | full simulator state | `study/oracles.py` |

## 1. MaskablePPO

**Policy and value.** Separate MLPs (`MlpPolicy`, `net_arch=[256,256]`, tanh)
produce logits $z_\theta(o)\in\mathbb R^{69}$ and value $V_\phi(o)$.

**Masking.** `MaskableCategorical.apply_masking` replaces the logits of
illegal actions by $-10^{8}$:
$\pi_\theta(a\mid o,m)=\dfrac{m(a)\,e^{z_a}}{\sum_{b}m(b)\,e^{z_b}}$
(the $-10^8$ logits underflow to probability exactly 0 in float32). Masks are
passed at rollout, at every gradient step (the stored rollout masks), and at
inference (`trainers_v2.MaskablePpoLearner.predict`). The entropy bonus is
computed over the legal actions only.

**Advantages.** GAE with $\gamma=0.995$, $\lambda=0.95$:
$\delta_t=r_t+\gamma V_\phi(o_{t+1})(1-d_t)-V_\phi(o_t)$,
$\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$, returns
$\hat R_t=\hat A_t+V_\phi(o_t)$. Time-limit truncation at scenario end is
bootstrapped with the terminal observation (SB3 ≥ 2 behaviour).

**Loss** (per minibatch, advantages normalized):
$$
\mathcal L(\theta,\phi)=-\mathbb E_t\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big]
+c_v\,\mathbb E_t\big[(V_\phi(o_t)-\hat R_t)^2\big]-c_e\,\mathbb E_t\big[\mathcal H(\pi_\theta(\cdot\mid o_t,m_t))\big],
$$
$\rho_t=\pi_\theta(a_t\mid o_t,m_t)/\pi_{\theta_{\text{old}}}(a_t\mid o_t,m_t)$,
$\epsilon=0.2$, $c_v=0.5$, $c_e=0.01$, gradient-norm clip 0.5, Adam $3\cdot10^{-4}$.

**Schedule.** 16 environments × 512 steps = 8,192 transitions per rollout;
8 epochs × 16 minibatches of 512. A 400k budget gives 48 complete rollouts
(48 × 8 = 384 SB3 "updates" = 6,144 gradient steps); the final 6,784
transitions are collected but not trained on (exact-budget stop, as in the
closed study). Deterministic inference = masked argmax of the logits.

```
for rollout in 1..48:                                   # trainers_v2.train_experiment
    for step in 1..512:  a ~ π(·|o, m) over 16 envs; store (o, a, m, r, V, logπ)
    compute GAE(γ=0.995, λ=0.95)
    for epoch in 1..8: for minibatch of 512: gradient step on L(θ, φ)
```

## 2. Masked contextual bandit (closed study)

**Model.** $\hat r_\theta:o\mapsto\mathbb R^{69}$, MLP 604–256–256–69, ReLU
(`ImmediateRewardNetwork`). One output per action; only the selected head is
trained.

**Action selection** (`predict`, lines 139–172): scores of illegal actions set
to $-\infty$; greedy $a=\arg\max_{a:m(a)=1}\hat r_\theta(o,a)$; during
training with probability $\varepsilon_n$ a uniformly random **legal** action
instead, $\varepsilon_n$ linear from 0.20 to 0.02 over the first 200k
transitions.

**Update** (`update`, lines 195–224): replay of the last 100,000
$(o,a,m,r)$ tuples; after 4,096 warm-up transitions, every 4 vector steps
(= 64 transitions) one Adam step ($3\cdot10^{-4}$) on
$$
\mathcal L(\theta)=\frac1B\sum_{i=1}^{B}\mathrm{Huber}_1\big(\hat r_\theta(o_i,a_i)-r_i\big),\quad B=512,
$$
gradient-norm clip 1.0. 400k transitions → 6,187 updates (≈ PPO's 6,144
gradient steps). There is no next state, discount, target network or
bootstrapping. The policy has no memory beyond the observation.

**Cost.** Inference: one forward pass (≈ 0.24 M multiply-adds; 0.24 M parameters). Training
memory is dominated by the replay buffer (100k × 604 float32 ≈ 242 MB).

```
for vstep in 1..25000:                                  # trainers_v2.train_experiment
    m = masks(vec);  a = ε-greedy_masked(r̂θ(o), m)
    o', r = vec.step(a);  replay.add(o, a, m, r)
    if vstep % 4 == 0 and n ≥ 4096: Huber step on r̂θ(o_i, a_i) → r_i, batch 512
    o = o'
```

## 3. Masked Q-learner with discount γ (added in this study)

Identical to §2 in network, replay, exploration, cadence, optimizer, loss and
clipping. For $\gamma>0$ the regression target becomes the *normalized*
double-DQN target
$$
y_i=(1-\gamma)\,r_i+\gamma\,(1-\mathrm{term}_i)\,Q_{\bar\theta}\big(o'_i,\ \arg\max_{a':m'_i(a')=1}Q_\theta(o'_i,a')\big),
$$
with a hard target-network copy every 250 updates; the successor maximization
is over **legal** successor actions only. The $(1-\gamma)$ factor rescales all
action values of a given $\gamma$ by the same constant (greedy policy
unchanged) and keeps the Huber transition point comparable across $\gamma$.
With $\gamma=0$ the class delegates to §2 and is bit-identical to it
(`tests/test_study.py::test_masked_q_with_gamma_zero_is_the_bandit_exactly`).

## 4. Baselines

All baselines propose moves through a read-only adapter
(`evaluation_v2.V2BaselineAdapter`); **only the first legal proposal** is
submitted per interval (V2 allows one TE change per interval).

* **Static**: keep every demand on $p_{d,0}$; move a demand back to $p_{d,0}$
  when it becomes available again after FRR.
* **Greedy** (thresholds from `configs/baselines.yaml`): if
  $\max_e\rho_e\ge0.85$, consider demands on the hottest link by decreasing
  volume and move the first to the candidate with the lowest current
  bottleneck utilization if it improves by ≥ 0.05.
* **CSPF**: every 6 intervals, place demands in (priority, volume) order on the
  cheapest candidate whose reservation stays ≤ 90 % of capacity; move only if
  the bottleneck improves by ≥ 0.08 (first such move submitted).

## 5. Clairvoyant H-step oracle (reference, not a controller)

For each legal $a$, clone the simulator (including the traffic RNG) and
compute $G_H(a)=\sum_{\tau<H}r_{t+\tau}$ for "take $a$, then no-op"; play
$\arg\max_a G_H(a)$ (ties → no-op). $H=1$ maximizes the exact immediate
reward, i.e. the quantity the bandit estimates. The oracle sees future
traffic and therefore upper-bounds what any *H-step greedy* non-clairvoyant
controller can achieve; it is not an optimal policy for the episode.
Cost: $|\mathcal A(s)|\cdot H$ simulated intervals per decision (≈ 50·H).
