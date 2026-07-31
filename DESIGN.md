# Design System — Dispatch Atlas

## Direction contract

**Thesis.** RL-in-MPLS is a dispatch atlas for one simulated network, not a dashboard of unrelated metrics. The stable Turkey topology and its route changes own the visual field; every surrounding element reads like the quiet notation, time band, and operating ledger of an infrastructure control room.

**Own world.** Matte blue-black drafting surfaces, cool mineral neutrals, fine cartographic rules, humanist language typography, condensed city plates, and tabular instrument numerals. Semantic color is used on routes and state, never as decorative glow.

**Story.** The viewer locates the network state, sees the incident and policy decision, follows old → proposed → observed routing, and can deepen the same moment into network or RL detail without losing context.

**First viewport.** A restrained application header and provenance ledger frame a geographically recognizable Turkey topology occupying most of the viewport. A time-distance incident band runs along its lower edge. Current facts sit in one quiet rail; a recommendation appears directly beneath the map only when an actual recommendation exists.

**Form.** A railway dispatcher's working diagram fused with a national infrastructure atlas. It refuses both the generic NOC card grid and the academic report page as the primary application form.

## Visual authority

This is a replacement visual world. Existing UI behavior, copy, data contracts, city registry, topology meaning, and evidence safeguards are source material; the current dark console, wallboard, and light study dossier are not authorities to blend together.

The world must remain recognizable across Presentation, Network Information, and RL Information. Density changes by mode, but the shell, route notation, typography, status grammar, object selection, and motion language do not.

## Direction exploration

Three genuinely different directions were evaluated:

| Direction | Product fit | Decision |
|---|---|---|
| **Dispatch Atlas** — national infrastructure atlas plus railway time-distance working diagram | Makes stable geography, incident time, route movement, and synchronized comparison one coherent object. Scales from a presentation stage to technical inspection. | Chosen |
| **Bench Instrument** — tactile creator hardware with keyed controls and a single amber readout | Excellent for presenter controls and state certainty, but turns the product into a control surface and makes the network secondary. | Rejected |
| **Timeband Console** — a 16-step instrument sequencer adapted to five-minute intervals | Strong story progress and causality, but overweights time, implies looping, and encourages ambient motion the product does not need. | Rejected |

The assigned Dispatch Atlas direction wins on both audience identification and product clarity. It looks native to people who read infrastructure diagrams without pretending to be a real operator system.

## Color strategy

Strategy: **full semantic palette on a restrained neutral field**. Most surfaces are neutral. Color belongs to network and evidence meaning.

### Foundation

| Token | Hex | Use |
|---|---:|---|
| `atlas-basalt` | `#111A1F` | Topology stage, fullscreen audience ground |
| `atlas-slate` | `#1B292F` | Inset stage areas, drawers, selected context |
| `atlas-fog` | `#E7ECEB` | Primary text on dark, light-mode page field where needed |
| `atlas-cloud` | `#A9B6B9` | Secondary text on dark; must not be used below AA contrast |
| `atlas-ink` | `#172126` | Text on light surfaces |
| `atlas-rule` | `#3B4A50` | Rules on dark, panel boundaries, topology guides |

### Semantic

| Token | Hex | Meaning | Non-color partner |
|---|---:|---|---|
| `state-normal` | `#5E9E92` | Healthy / available / recovered steady state | Solid line + check icon |
| `state-pressure` | `#D0A54B` | Congestion pressure / SLA risk | Double-line or triangle marker |
| `state-failure` | `#D4625F` | Failed / invalid / integrity failure | Broken dash + × icon |
| `state-recovery` | `#7AAE66` | Recovering / restoration event | Dashed-to-solid line + recovery arrow |
| `state-selection` | `#E07A3F` | Current selection / policy recommendation / primary focus | Outer keyline + pointer notch |
| `state-comparison` | `#8794C7` | Synchronized comparator lane | Parallel rail + `B` token |

Do not use gradients for surface decoration. A utilization legend may use discrete stepped swatches, not a continuous rainbow. Failure red is not used for ordinary negative reward; reward polarity uses signed bars and text first.

### Provenance states

Provenance is encoded as a rectangular ledger stamp, icon, word, and pattern:

- `LIVE`: solid leading rule, play/pause glyph, current clock.
- `RECORDED`: diagonal tape hatching, record glyph, recorded step index.
- `DEVELOPMENT`: open-grid hatching, flask/wrench glyph, selection-stage wording.
- `FINAL EVIDENCE`: double rule, seal glyph, source SHA and one-shot wording.

These states never share a generic “status badge” treatment. A user must distinguish them in grayscale.

## Typography

No CDN, npm, or new font binary is required.

- Language and interface: `Aptos, "Segoe UI Variable", "Segoe UI", system-ui, sans-serif`.
- City plates and narrow structural labels: `Bahnschrift, "Arial Narrow", Aptos, sans-serif` with restrained uppercase and generous tracking.
- Data and identities: `"Cascadia Mono", Consolas, ui-monospace, monospace`, with tabular numerals.

### Scale

| Role | Presentation | Desktop | Narrow |
|---|---:|---:|---:|
| Mode title | 28–34 px | 22–26 px | 20–22 px |
| City | 17–20 px | 13–16 px | 12–14 px |
| Top KPI | 40–56 px | 30–40 px | 28–34 px |
| Body | 18–20 px | 14–16 px | 15–16 px |
| Utility label | 13–14 px | 11–12 px | 11–12 px |
| Data | 18–24 px | 13–18 px | 13–16 px |

Changing numeric fields reserve width and use tabular figures. Units remain attached to their values. Numerals update without resizing their container.

## Spacing and geometry

Base unit: 4 px. Primary rhythm: 8, 12, 16, 24, 32, 48 px.

- Shell padding: 16 px at laptop, 24 px at desktop, 32 px at presentation.
- Main stage gutters: 16–24 px; stage-to-rail gap is a rule, not a floating-card gap.
- Text measure: 48–72 characters for explanation copy.
- Corner radius: 2 px for map plates and ledger cells, 4 px for interactive panels, 6 px maximum for modal/drawer containers.
- Do not create a grid of rounded cards. Use sectional rules, shared baselines, and deliberate negative space.

## Elevation

The interface is primarily flat.

- Level 0: page and stage.
- Level 1: inset inspector separated by a 1 px rule and tonal shift.
- Level 2: active recommendation or modal, with one 16 px soft shadow and a selection keyline.
- No glow. No stacked glass surfaces. No shadow on routine tables, chips, or controls.

## Iconography

Use a consistent 1.5 px outlined icon family with square terminals. Icons require text or an accessible name. Avoid decorative icon tiles.

Core icons: topology node, route, demand, congestion, SLA risk, failure, recovery, FRR, recommendation, approved, rejected, mask, observation, reward, checkpoint, seed ledger, live, recorded, development, final evidence, fullscreen, audience view, incident bookmark, and direct-link target.

## Topology language

### Node plate

Nodes are fixed-position plates, not bubbles:

```text
ANKARA · LSR
P2    3 LSP · 62%
```

- City and operational role lead.
- Internal ID is secondary and never removed.
- Microtelemetry is mode-specific and limited to two short values.
- PE ingress/egress, P core, and aggregation roles differ by plate-edge geometry and role token, not color alone.
- Selected nodes receive an outer keyline; keyboard focus receives a separate high-contrast focus ring.

### Links

- Base: thin neutral route line.
- Capacity: two or three discrete weight classes, with exact capacity in inspection.
- Directional telemetry: paired directional hairlines or a direction arrow in Network Information mode; Presentation defaults to worst-direction state with exact direction on focus.
- Congestion: stepped color plus pressure ticks.
- Failure: broken dash, × at the break, and reduced underlying line opacity.
- Recovery: temporary segmented line that resolves to solid.
- Primary/current path: solid selection rail.
- Alternate path: thinner dashed rail.
- Comparator: parallel offset rail, never a glow.

### Geography

Use a curated, geographically recognizable Turkey layout, not an exact GIS claim. West/east and regional relationships must feel correct. Node positions remain stable in every mode and state. Links use deliberate bends and shared trunks to reduce crossings. The long Kayseri–Samsun constraint and Ankara–Kayseri failure corridor must be legible.

## Charts and tables

- Prefer direct labels over legends.
- Use signed diverging bars for reward components and policy margins.
- Use step or line charts for time; no smoothing that invents intermediate values.
- Use paired rails for synchronized policy comparison.
- Use discrete utilization bands with printed values.
- Tables keep row identity fixed left and numeric measures right-aligned.
- Every chart has a table or textual equivalent and states source, grain, unit, and stage.
- Development and final evidence never share a plotting region.

## Motion

Motion explains change; nothing moves continuously.

- Controls: 120–180 ms.
- Drawers and panels: 200–320 ms.
- Topology event: 400–800 ms.
- An old route fades to a neutral ghost while the proposed route draws once; after execution the observed route settles to solid.
- Link pressure changes interpolate once between discrete states.
- Recommendation enters only after the causal event and appears directly below the topology.
- Timeline current-time marker advances only when the simulation or recorded step advances.
- No ambient pulses, bouncing, typewriter effects, looping chase lights, parallax, repeated number rolling, or topology physics after initial layout.
- `prefers-reduced-motion` replaces every transition with an immediate state swap while retaining before/after labels.

## Interaction principles

1. Selection is shared. A node, link, demand, action, or event selected in one mode remains the selected object after a mode switch when that object exists in the target depth.
2. Every decision links observation → mask → output → action → transition → reward → next observation.
3. “Explain this moment” changes depth without changing the underlying event.
4. The topology never executes an action by itself. Decision Lens previews; an explicit action control executes or approves.
5. Unavailable data remains a labelled state with the reason and required source, not an empty decorative container.

## Responsive adaptation

- 1920×1080: topology dominates; quiet right rail and presenter cockpit can hide.
- 1440×900: topology remains at least 60% of content width; inspectors use one rail.
- 1280 px: secondary inspector becomes a drawer; context strip abbreviates values but retains provenance word.
- 768 px: topology and selected-object summary remain; inspectors become bottom sheets; tables use internal scrolling.
- 390 px: one-column flow with mode control, provenance, moment summary, topology viewport, and focused object list. The topology has an accessible list twin and never forces page-level horizontal scrolling.

## Accessibility floor

- WCAG 2.1 AA for text and meaningful graphics.
- Minimum 44×44 px touch targets at 768 px and below.
- Visible focus not hidden by selection state.
- Topology objects are keyboard reachable in geographic order, with arrow-key local navigation and a list alternative.
- Incidents, failures, pressure, selection, provenance, and policy series use shape, label, line style, or token in addition to color.
- Live regions announce state changes and recommendations without narrating every telemetry tick.
- Focus returns to the invoking control when a drawer or modal closes.
- Zoom to 200% preserves reading order and control access.

## Absolute bans

- Generic admin-dashboard card mosaics.
- Cyberpunk HUD chrome, neon glow, scanlines, or decorative telemetry.
- Glassmorphism, arbitrary gradients, and pill proliferation.
- Anthropomorphic policy language or “AI advisor.”
- Fake link-level recorded replay.
- Probabilities assigned to bandit scores.
- Causal claims from changed-feature ranking.
- Evidence-stage colors without words and patterns.
- Animated topology layout, ambient motion, or interaction dependent on animation.
