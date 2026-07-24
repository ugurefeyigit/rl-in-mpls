# Demonstration script

## 5-minute version

Preparation: `python scripts/demo.py` (session is created paused at 17:00,
RL vs static, seed 42). Full-screen the browser.

1. **[0:00] Frame the question.** "Same network, same traffic, two
   controllers. Left: reinforcement learning. Right: static shortest-path —
   what most IGPs do. Can the RL controller keep this backbone out of
   trouble during an evening peak, a flash crowd and a backbone failure?"
2. **[0:45] Show the network.** Hover a backbone link: capacity, load,
   delay, loss, LSP count. Point out the traffic classes in the LSP tab —
   voice with 60 ms SLA vs 400 ms bulk.
3. **[1:15] Press Resume (1×).** Narrate the evening ramp: video demand
   climbs, links go amber. Point at the decision tape: most intervals the
   agent does nothing — "doing nothing is a decision; rerouting costs."
4. **[2:00] First RL reroute.** Pause when the tape shows one. Open the
   Agent tab: action probability, safety-filter verdict, reward breakdown,
   and the counterfactual — "had it done nothing, max utilization would have
   been X%; it is Y%." Emphasize the label: this explanation is computed
   from telemetry, not the network's introspection.
5. **[2:45] Speed 5×** until the 20:00 flash crowd hits PE6, then the 20:15
   failure of L20 (P5–P8). Switch to Metrics: max-utilization curves
   diverge; static rides the broken shortest paths into loss, RL spreads
   flows. Show the SLA-violations series.
6. **[4:00] Recovery at 21:00.** Note hysteresis: the agent does not slam
   everything back instantly (reroute cost + cooldown).
7. **[4:30] Close on the summary.** Metrics tab, cumulative reward + SLA
   count. "Across 5 seeds and 7 scenarios the picture is: RL ≈ greedy/CSPF
   on calm days, ahead under failures and flash crowds, never free — every
   reroute is paid for. Numbers, not vibes: results/eval_stats.csv."

## 10–15 minute version

Add:
- **Deceptive local optimum** (scenario `deceptive_local_optimum`, compare
  `rl` vs `greedy`): both PE1/PE2 flows' individually-shortest paths share
  the hidden P5→P8 bottleneck; greedy chases local wins, RL learned to park
  bulk traffic on the A1 detour. Show the link table's Δutil column.
- **Manual interventions**: fail L11 (2 Gbps backbone) live from the control
  rail; show FRR entries in the tape, then the controller's cleanup.
- **Safety filter off** (Session → checkbox): rerun a failure, show REJECTED
  lines disappearing and what an unconstrained policy does differently.
- **Training tab**: the actual training curve, reward components in
  TensorBoard, and the honest statement of training cost (~40 min CPU).
- **A failure case** (docs/REPORT.md §Failure cases): the OOD double-failure
  scenario where RL's advantage shrinks/inverts — shown deliberately.

## Likely questions & defensible answers

**Q: Is this real MPLS?** Flow-level abstraction: FEC→LSP mapping over
explicit loop-free paths with local repair; no label stacks/RSVP signaling.
Standard for TE studies; the README says so explicitly.

**Q: Could the agent be cheating with future knowledge?** The observation
contains only current/derived telemetry + wall clock (env docstring lists
every feature). Traffic is exogenous; the counterfactual panel is computed
post-hoc and never fed back.

**Q: Why PPO with masking, not DQN?** The valid-action set changes each step
(failures, cooldowns, bandwidth checks). Masked policy-gradient handles that
natively; DQN would need ad-hoc Q-masking and was not needed.

**Q: Why does RL sometimes do nothing during congestion?** Reroute+flap
penalties and cooldowns make inaction optimal when no candidate path
improves the bottleneck — visible in the Agent tab's candidate list.

**Q: How does this reach a real network?** Advisory/shadow mode first:
telemetry in (SNMP/streaming), PCEP/SR-Policy or RSVP-TE out, operator
approval gate, hard constraint checker always on (docs/REPORT.md
§Deployment). This code is not that system.

**Q: One lucky seed?** The demo seed is fixed and disclosed; all claims come
from 5-seed paired evaluation with CIs (results/eval_stats.csv).
