# Unified RL-in-MPLS Surface Brief

**Primary target:** `frontend/index.html`

**Related targets:** `frontend/present.html`, `frontend/study.html`

**Canonical Impeccable record:** `.impeccable/surfaces/frontend-index-html.md`

## Purpose

Redesign the existing engineering console, Presentation Mode, and governed-study surface as one coherent network operations atlas with exactly three primary modes: Presentation, Network Information, and RL Information. Guided Story is a workflow inside Presentation, never a fourth mode.

## Audience and depth

- Presentation serves viewers with basic networking and RL knowledge, plus a presenter who needs live control and truthful narrative support.
- Network Information serves network engineers inspecting MPLS demand, LSP, congestion, failure, FRR, restoration, utilization, and SLA behavior.
- RL Information serves technical reviewers inspecting the actual observation-to-transition pipeline, learner outputs, action validity, rewards, and evidence provenance.

All modes share scenario, session, model, time, selection, provenance, and explanation context. Moving between modes changes depth, not the underlying moment.

## Visual direction: Dispatch Atlas

Use a calm national-network atlas crossed with the legibility of a dispatch diagram. The geographically recognizable Turkey topology is the signature surface and remains stable while events, paths, and decisions change around it. Quiet neutral chrome, humanist language typography, tabular data typography, generous space, and a small semantic palette support the topology rather than competing with it.

Avoid a generic dashboard grid, cyberpunk HUD, glassmorphism, marketing sections, constant motion, invented glow, and walls of monospace. City and role lead each node title; the unchanged internal router ID is secondary.

## Truth contract

The interface treats LIVE, RECORDED, DEVELOPMENT, and FINAL EVIDENCE as distinct typed states with persistent, restrained provenance. Recorded traces never imitate live topology telemetry. Final-holdout evidence is a frozen report, never a live comparator. Missing data is labelled unavailable; it is not inferred.

PPO probabilities are probabilities only when the backend exposes them. Masked-bandit outputs are scores or immediate reward estimates. Changed observation features are ranked changes, not causal importance. Predicted, expected, and observed outcomes remain separate. Episode-mean no-op frequency and step-pooled no-op share retain their distinct names and denominators.

## Core composition

The shell has three unmistakable mode controls, a persistent context rail, provenance status, contextual help, and an Explain this moment action with Presentation, Network, or RL depth. Direct references can focus the current incident, decision, node, link, demand, path, action, or reward event.

Presentation is topology-first, with one compact current-moment strip, a recommendation/outcome card beneath the map, a quiet comparison lane, hidden-by-default presenter cockpit, audience view, Guided Story progress, incident bookmarks, and Q&A jumps.

Network Information expands the same topology into a serious MPLS operations workspace with filters, bidirectional topology/table selection, inspectors, deltas, alternate paths, bottlenecks, and an incident-to-stabilization timeline.

RL Information makes the inference pipeline its main organizing axis: observation, action mask, policy outputs, selected action, safety validation, transition, twelve reward components, and next observation. It includes grouped search over the 604 V2 features, the full 69-action space, learner-specific output language, exact-sum reward integrity, and provenance.

## Responsive and interaction contract

At 1920x1080 and 1440x900, the topology and primary work area remain dominant. At 1280, secondary inspectors narrow or become drawers. At 768, one inspector is visible at a time. At 390, the screen preserves mode, provenance, moment summary, an accessible topology/list representation, and progressive drawers without horizontal page overflow.

All controls and topology objects are keyboard operable with visible focus. Status never depends on color alone. A synchronized list/table alternative mirrors topology focus. Motion is limited to causal continuity, supports reduced motion, preserves node positions, and targets smooth 60 fps behavior.

## Implementation guardrails

Reuse the repository's dependency-free frontend unless a later approved architecture change demonstrates necessity. Keep the existing live and evidence APIs available while typed presentation adapters are introduced. Preserve direct links: `/present` selects Presentation, `/` and `/advanced` select Network Information, and `/study` selects RL Information at the governed-study section.

Do not change frozen studies, learners, training code, reward or scenario semantics, protected manifests, recorded artifacts, or existing worktrees. Backend additions must be additive, truthful, and covered by contract tests. The complete behavior, data-source mapping, migration gates, exclusions, and acceptance criteria live in the unified product design specification.
