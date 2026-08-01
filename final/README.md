# RL-in-MPLS — release

This directory is the operator's entry point. Three documents, in the order you
need them:

| Read this | When |
|---|---|
| **[RUNNING_IT_AGAIN.md](RUNNING_IT_AGAIN.md)** | you want it running — exact commands, exact directories |
| **[OPERATING_THE_UI.md](OPERATING_THE_UI.md)** | it is running and you want to drive it — every control, what each button does |
| **[RELEASE_NOTES.md](RELEASE_NOTES.md)** | you want to know what this release contains and what it deliberately refuses to do |

---

## The 30-second version

From the repository root:

```bash
python -m uvicorn server.main:app --port 8000
```

Then open **<http://127.0.0.1:8000/present>**.

In the left panel: press **Start run**. That is the whole minimum path — the
defaults are the governed V2 environment, the `demo_evening` scenario, seed 42
and the masked contextual bandit on training root 42.

---

## What this is

A flow-level simulation of an 18-router MPLS backbone, driven either by learned
controllers (a masked contextual bandit and MaskablePPO) or by conventional ones
(static shortest path, utilization-aware greedy, CSPF-style reoptimization). All
controllers face byte-identical seeded traffic, so every comparison is paired.

**It is a simulation study, not a production controller.** The fictional scaled
topology is not a real operator network, and the placement on screen is a fixed
engineering schematic, not geography.

## What the study established

The closed V2 study asked whether *planning* explains the learned gain. It does
not: the explicitly myopic bandit beat the planning agent.

| Method | One-shot holdout return |
|---|---:|
| Masked contextual bandit | **18.221** |
| MaskablePPO | 9.036 |
| Greedy (best baseline) | −2.327 |

These are the only numbers in this product that support a conclusion. The
application renders them from the frozen artifacts, never from a literal in the
source. Everything you produce by pressing Start run is a **demonstration** — it
is labelled as such everywhere it appears, and it is never averaged with the
numbers above.

## Three rules the interface keeps

1. **A value the engine does not have is absent with a reason, never zero.**
2. **A comparison renders only while both lanes can be proved to share one
   experiment.** When the proof breaks, you get the refusal and the fields that
   broke it — not a verdict with a caveat.
3. **A demonstration and a holdout result never share a table.** They answer
   different questions.

## Full documentation

Everything below lives in the repository's `docs/` directory:

| Doc | Contents |
|---|---|
| `docs/PRODUCT_UI.md` | modes, routes, sources, topology, unavailable-data rules |
| `docs/PRESENTATION_MODE.md` | audience view, Guided Story, controls, evidence treatment |
| `docs/RESULTS_AND_COMPARISON.md` | the paired comparison and the three record classes |
| `docs/API.md` | every HTTP route |
| `docs/ARCHITECTURE.md` | how the layers fit together |
| `docs/ACCESSIBILITY.md` | keyboard, focus, list alternative, reduced motion |
| `docs/TECHNICAL_DEFENSE.md` | methodology, roots, selection, limitations |
| `docs/ADR-003-...md` | why demonstrations and evidence never merge |

---

**Author:** Uğur Efe Yiğit · **License:** proprietary, all rights reserved. This
repository is publicly readable but not open source. See `LICENSE`.
