# ADR-002: Unified three-mode product shell with typed provenance adapters

**Status:** Accepted for Prompt 2 implementation

**Date:** 2026-07-31

**Decider:** Repository owner

## Context

RL-in-MPLS currently has three separate static frontends:

- `/` and `/advanced`: a live V1 engineering console;
- `/present`: a live V1 Presentation Mode with Guided Story;
- `/study`: a read-only record of the closed V2 study and its recorded replay.

They contain valuable behavior and safeguards, but they read as separate websites, duplicate state and rendering logic, use inconsistent terminology, and make it difficult to move from an audience explanation to network or RL detail without losing context.

The redesign requires exactly three primary modes—Presentation, Network Information, and RL Information—while preserving all live and evidence behavior. It must also keep `LIVE`, `RECORDED`, `DEVELOPMENT`, and `FINAL EVIDENCE` as distinct data types, not merely different labels.

## Decision

Build one framework-free ES-module application shell with a shared context store and source-specific adapters:

- `LiveSessionAdapter` consumes current live REST/WebSocket data and future versioned live-V2 decision telemetry.
- `RecordedReplayAdapter` consumes only payloads marked `kind: recorded_replay` and `live: false`.
- `DevelopmentEvidenceAdapter` consumes only `/api/v2/development/*` payloads.
- `FinalEvidenceAdapter` consumes only `/api/v2/final-holdout*` payloads.

The shell owns mode, route, provenance, session/evidence context, selected object, current moment, comparison, drawers, and Guided Story workflow. Adapters expose typed capabilities; unsupported values render as unavailable. They do not normalize away source grain.

The canonical application is served by all current product URLs with route-specific initial context:

- `/present` → Presentation mode;
- `/advanced` and `/` → Network Information mode;
- `/study` → RL Information mode, Governed Study view, `FINAL EVIDENCE` context.

Guided Story uses Presentation mode with `workflow=guided-story`; it is never registered as a primary mode.

## Options Considered

### Option A: Unified shell with typed adapters — chosen

| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Scientific safety | High: provenance types stay distinct |
| Context preservation | High |
| Migration risk | Medium, controllable through parallel routing |
| Team familiarity | High: plain ES modules and existing libraries |

**Pros:** One product identity; explicit source capabilities; shared selection and keyboard model; incremental migration; legacy URLs stay useful.

**Cons:** Requires a real frontend state model and decomposition of large incumbent modules.

### Option B: Shared header and stylesheet over the three current pages

| Dimension | Assessment |
|---|---|
| Complexity | Low initially |
| Scientific safety | Medium |
| Context preservation | Low |
| Migration risk | Low initially, high long term |
| Team familiarity | High |

**Pros:** Fastest visual convergence; minimal route changes.

**Cons:** Keeps duplicated stores, disconnected navigation, inconsistent data models, and separate accessibility behavior. It cannot satisfy “different depths of the same product.”

### Option C: One generic backend façade that flattens all live and evidence data

| Dimension | Assessment |
|---|---|
| Complexity | High |
| Scientific safety | Low: flattening invites grain mixing |
| Context preservation | High |
| Migration risk | High |
| Team familiarity | Medium |

**Pros:** Simplest client payload shape.

**Cons:** Treats provenance differences as presentation details, encourages final/development or live/recorded blending, and expands a stable read-only evidence backend without need.

## Trade-off Analysis

Option A adds intentional client architecture, but it is the only option that satisfies both coherence and scientific honesty. The key trade-off is accepting source-specific UI states instead of forcing every source into one universal telemetry shape. Capability checks make missing information visible and testable.

The existing no-build architecture remains. Prompt 2 should split modules by product responsibility rather than introduce a framework or bundler during the redesign.

## Consequences

- Mode changes preserve source, scenario, time, selection, and comparison when compatible.
- A source switch is explicit and may change which controls are available.
- Provenance becomes a discriminated state that components must handle exhaustively.
- Live V1 and future live V2 decisions may coexist, but the environment version is always visible.
- Recorded replay cannot reuse live topology telemetry components unless the payload genuinely provides the required fields.
- The current large `app.js`, `present.js`, and `study.js` modules are migrated into focused stores, adapters, views, and components; scientific readers in `mplssim/evidence/` remain unchanged.
- Direct URLs remain stable while visual and interaction behavior converge.

## Action Items

1. Add the shared context and typed source adapters with contract tests.
2. Add a capability/schema endpoint and versioned decision-observatory payloads without changing V2 scientific definitions.
3. Add exact paired-session initialization from a cloned starting engine and expose synchronization proof.
4. Build the unified shell and topology stage behind a non-default migration route.
5. Migrate Presentation and Guided Story, then Network Information, then RL Information.
6. Map `/present`, `/advanced`, `/`, and `/study` to the unified shell after parity gates pass.
7. Remove legacy frontend modules only after route, evidence, accessibility, and full-suite acceptance.
