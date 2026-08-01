# Exp 2.1 · Comparative Run Results

Exp 2.1 is a product/UI release label. It does not rename or alter the frozen V2 environment, reward, actions, checkpoints, evidence, or experiment identities.

## Migration to four primary modes

The primary navigation now contains Presentation, Network Information, RL Information, and Comparative Run Results. Guided Story remains inside Presentation. Existing `/`, `/advanced`, `/present`, and `/study` routes remain compatible; `/compare` is additive.

## Temporary A/B lifecycle

Only completed live-demonstration runs are eligible for A or B. The two slots hold normalized copies of real in-memory history, identity, checkpoint provenance, interval actions, reward components, metrics, failures, recoveries, and churn. Assigning a new run replaces that slot; Swap exchanges A and B; Clear A, Clear B, and Clear All are explicit. Reset run leaves completed slots intact. Full Reset clears both slots and the shared interval selection. A server restart also drops them. Nothing is written to `results/`, `runs/`, `models/`, or evidence paths.

## Comparison truth

A and B use fixed identity colors plus letter, line style, and marker shape. Those colors never imply winner and loser. Metric outcomes use direction-aware semantics: higher delivery and operational return are favorable, while higher SLA risk, loss, utilization, accepted changes, reversals, flaps, and moved bandwidth are unfavorable. Missing values remain unavailable.

Synchronized runs share environment, scenario, seed, step sequence, and simulation-time grid. When any field differs, the page continues to show both completed-run results but disables paired or causal interval conclusions.

## Analytical surface

The mode provides interval or explicitly toggled cumulative reward, maximum and mean utilization with 70% and 100% references, separate delivery and SLA-risk plots, a decision/incident timeline, count-only churn bars with Mbps reported separately, and a diverging reward-component comparison. Every graph includes axes, units, A/B labels, keyboard points, and a table of the same values. Stored-run deep links show the aggregate interval in Network or RL Information and explicitly refuse per-link or observation reconstruction when it was not archived.

Known limitations remain honest: stored A/B runs do not survive restart; older run histories lacking an action, moved-bandwidth field, or a complete V2 component set show those measures as unavailable; no recorded per-link utilization is derived from aggregate telemetry.
