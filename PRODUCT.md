# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

RL-in-MPLS serves three audiences through one product:

- Presenters and audience members with basic networking and reinforcement-learning knowledge. They need the whole project explained clearly while a simulation or recorded result is in view.
- Network engineers investigating MPLS traffic engineering, topology state, demands, paths, congestion, failure, FRR, restoration, utilization, delay, loss, SLA risk, and route stability.
- Reinforcement-learning practitioners and technical reviewers interrogating observations, masks, policy outputs, selected actions, safety validation, transitions, rewards, checkpoints, and experimental provenance.

The same person may move between these depths during a live presentation or technical defense. The product must preserve the current scenario, session, model, time, selection, and evidence context when they do.

## Product Purpose

RL-in-MPLS is an interactive, scientifically honest traffic-engineering simulation and evidence reader. It makes a fictional 18-router MPLS backbone, its traffic-engineering decisions, and the results of a governed learner comparison inspectable at an audience-appropriate depth.

Success means that a viewer can answer three questions without being misled:

1. What is happening in the simulated network now, and what changed?
2. What decision did an implemented policy or baseline make, why is it valid, and what outcome was observed?
3. What did the closed V2 study establish, what did it not establish, and which evidence stage supports the claim?

## Positioning

The product combines a runnable MPLS-TE simulation, policy-level decision inspection, synchronized controller comparison, and a fail-closed read-only record of a completed governed study. Its differentiator is not that reinforcement learning always wins. It is that live behavior, recorded replay, development evidence, and untouched final-holdout evidence remain distinguishable while positive, mixed, and negative findings stay visible.

## Operating Context

- Live presentation on a 16:9 display, with a presenter controlling story pace and an audience reading the topology from a distance.
- Network-operations exploration on desktop or laptop, with topology-to-table inspection and incident timelines.
- RL and scientific defense, with precise feature, action, reward, checkpoint, seed, and integrity provenance.
- Offline-capable local serving through FastAPI, plain ES modules, vendored Cytoscape and ECharts, and no required CDN or frontend build step.
- Direct links may open Presentation, Network Information, RL Information, Guided Story, or governed-study context without creating separate product identities.

## Capabilities and Constraints

### Product modes

Exp 2.1 has four primary modes: Presentation, Network Information, RL Information, and Comparative Run Results. Guided Story remains a dedicated workflow within Presentation, not a primary mode. Comparative Run Results operates only on two bounded, completed live-demonstration records in process memory; it does not change the four evidence/provenance source kinds or promote a demonstration to evidence.

### Network and simulation truth

- The topology is fictional and scaled for demonstration. It is not a real operator network.
- It contains 18 routers, 32 undirected links, 17 demands, and six traffic classes.
- Internal router, link, demand, scenario, action, model, and policy identifiers are immutable contracts. City and role names are display labels layered over those identifiers.
- The simulator models flow-level traffic engineering, analytic delay and loss, candidate LSP paths, failures, FRR-style local repair, restoration, dwell, reroutes, reversals, flaps, and moved bandwidth.
- It does not model packets, TCP dynamics, RSVP-TE or IGP convergence, label signaling, or a production operator control plane.
- Existing live sessions run the V1 environment and policies through `server/session.py`. The closed V2 study uses `MplsTeEnvV2`, a 604-value observation, a 69-action space, and 12 reward components. The product must state the environment version wherever that distinction matters.

### Evidence and provenance

Four states must never be blurred:

- `LIVE`: a running or paused simulation session.
- `RECORDED`: playback of an immutable recorded trace; never a controller execution.
- `DEVELOPMENT`: pilot, continuity, learning-curve, or checkpoint-selection evidence.
- `FINAL EVIDENCE`: the untouched one-shot final holdout.

Every scientific view carries its state, source, environment identity, scenario, seed, model or policy, and time or evidence grain. Missing or inconsistent evidence fails closed and is shown as unavailable, never as zero or an estimate.

Recorded V2 traces contain aggregate interval telemetry, not per-link utilization. Recorded replay therefore cannot animate or color a link-level topology.

The two no-op statistics remain separate: episode-level mean no-op frequency and step-pooled no-op share.

### Policy-output semantics

- PPO outputs may be described as action probabilities only when the backend exposes the masked distribution. Entropy and value estimates are shown only when exposed.
- Masked-bandit outputs are action scores or immediate-reward estimates, never probabilities or confidence.
- Actual observations, predicted outcomes, and observed post-action outcomes are separate fields.
- Changed-feature ranking is descriptive and is never called causal importance or an explanation of the policy's internal reasoning.
- A counterfactual is available only when the exact state can be cloned and evaluated without mutating the running session. It is labelled a simulated estimate.
- Values unavailable from the current engine are explicitly unavailable until a truthful backend addition supplies them.

### Implemented methods

- Live V1 session methods: MaskablePPO, static shortest path, utilization-aware greedy, CSPF periodic reoptimization, and the random sanity floor.
- Frozen V2 comparison methods: MaskablePPO, masked contextual bandit, static shortest path, utilization-aware greedy, and CSPF periodic reoptimization.
- The UI cannot invent a controller, baseline, model capability, or synchronization guarantee.

### Governed-study boundary

The V2 study is complete and frozen. Product work must not train, tune, resume, evaluate, reselect, sweep, retry, or alter learners; change reward, observation, action, topology, scenario, seed, mask, horizon, metric, or baseline semantics; or modify governed artifacts.

## Brand Commitments

- Product name: RL-in-MPLS.
- Voice: calm, precise, direct, technically literate, and explicit about uncertainty and evidence grain.
- Use policy names or “policy recommendation”; do not call a recommendation an AI advisor and do not anthropomorphize a model.
- Use city and role prominently with internal ID secondary, for example `ANKARA · LSR` with `P2` as technical detail.
- Preserve Turkish city names from `mplssim/display.py`: İstanbul, İzmir, Bursa, Antalya, Eskişehir, Ankara, Konya, Bolu, Kayseri, Adana, Gaziantep, Samsun, Sivas, Malatya, Trabzon, Erzurum, Diyarbakır, and Van.
- Preserve the fictional-topology disclosure wherever network imagery could be mistaken for a real operator view.
- Negative and mixed results are first-class product content, not disclaimers hidden after a positive narrative.

## Evidence on Hand

- Live topology, traffic-class, scenario, session, telemetry, metrics, LSP, link, action, reward, comparison, advisor-lookahead, event, export, and run-summary APIs.
- Read-only V2 study, final-holdout, scenario, reward-component, action, integrity, provenance, development, disclosure, and recorded-replay APIs.
- Frozen compact evidence under `results/v2_final_holdout/`, `results/v2_three_root_continuity/`, and `results/v2_seed42/`.
- Preserved recorded per-step artifacts outside Git when `V2_FULL_ARTIFACTS` is configured.
- A central city, scenario, class, link, and glossary registry in `mplssim/display.py`.
- Existing engineering-console, Presentation Mode, Guided Story, and sealed-study implementations as behavior and content evidence, not as visual authority for the redesign.
- No real operator topology, customer telemetry, production deployment evidence, causal model attribution, or per-link recorded V2 utilization exists and none may be fabricated.

## Product Principles

1. **State before story.** A viewer can always tell whether the product is live, recorded, development, or final evidence before interpreting a number.
2. **One network, four primary modes.** Presentation, Network Information, and RL Information preserve one live or recorded context at increasing depth. Comparative Run Results is the completed-run analytical surface and deep-links its selected interval back into those same Network and RL vocabularies without inventing a stored topology snapshot.
3. **Topology as the shared object.** The network and the current decision remain the anchor; panels support them instead of competing with them.
4. **Mechanism before marketing.** Show the real observation, action, transition, outcome, and evidence chain. Never substitute anthropomorphic or fabricated explanation.
5. **Negative results remain visible.** The bandit's overall win, PPO's deceptive-local-optimum win, movement cost, limitations, and the lack of positive evidence for temporal planning travel together.

## Accessibility & Inclusion

- Meet WCAG 2.1 AA contrast.
- Support keyboard operation, visible focus, screen-reader semantics, and a list/table alternative to the graphical topology.
- Never encode state by color alone.
- Respect `prefers-reduced-motion`; understanding and control cannot depend on animation.
- Remain usable at 1920×1080, 1440×900, 1280px, 768px, and 390px without horizontal page overflow.
- Use readable city labels, stable topology positions, tabular numerals for changing data, and language that works for both a room audience and an expert reviewer.
