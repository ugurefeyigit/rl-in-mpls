# V3 research backlog

> ## UNAPPROVED — NOT EVALUATED
>
> **Nothing in this file is supported by V2 evidence.** These are hypotheses and
> proposals only. None has been run, tuned, measured, or authorized.
>
> The governed V2 study is **closed**. Its holdout was consumed by a single
> evaluation and cannot be reused. No item below may borrow V2's holdout seeds
> (1001–1005), reuse a V2 selected checkpoint as a tuned starting point, or cite a
> V2 holdout number as motivation for a specific design choice.
>
> Each item requires its own preregistration before any code runs: a stated
> hypothesis, a frozen environment definition, fixed training roots, **new**
> development seeds, and **untouched** evaluation seeds that no one has looked at.

## Why this file is separate

The V2 record is a closed scientific artifact. Ideas about what to do next are
not evidence, and mixing the two would let a speculative roadmap borrow
credibility from a completed study. They are kept apart deliberately: the `/study`
surface renders no item from this file, and the evidence API has no route that
serves it.

## What V2 actually leaves open

The holdout found no positive support for a need for temporal planning **in this
formulation**. It did not find that planning is irrelevant. The most defensible
reading is that the 604-dimensional observation may already encode enough state
that a strong myopic map is close to sufficient when actions are largely
reversible at the next 5-minute interval.

That reading is a **hypothesis V2 does not test**. Everything below follows from
it.

---

## Proposed items

### V3-1 · Does the observation already contain the future?

**Hypothesis.** The myopic learner wins because the observation leaks
future-relevant state (diurnal phase, recent trend, scheduled events). Remove that
leakage and a temporal learner should separate from the bandit.

**Sketch.** Preregister ablated observation variants that mask trend and
phase-carrying features. Train both learners under each variant. The bandit is the
control condition throughout.

**Why it is not V2.** Changing the observation changes a frozen scientific
definition. New environment version, new pin, new seeds.

**Status.** UNAPPROVED — NOT EVALUATED.

---

### V3-2 · A recurrent policy under an equal budget

**Hypothesis.** A recurrent policy (e.g. LSTM-conditioned) can exploit history
that a feed-forward PPO cannot, and would beat the myopic control condition where
feed-forward PPO did not.

**Sketch.** Same environment, same action mask, same budget. Bandit retained as
control. Preregistered checkpoint rule on new development seeds.

**Why it is not V2.** Adding a learner to the V2 comparison after the holdout has
run is exactly the reselection the governance forbids.

**Status.** UNAPPROVED — NOT EVALUATED.

---

### V3-3 · An explicit planner

**Hypothesis.** A model-based or lookahead controller with access to a demand
forecast beats both V2 learners, isolating "planning" from "learned reaction".

**Sketch.** Requires a forecast interface the V2 environment does not expose, and
a decision on whether forecast error is simulated. Both are scientific-definition
changes.

**Why it is not V2.** New capability, new environment surface, new evaluation.

**Status.** UNAPPROVED — NOT EVALUATED.

---

### V3-4 · A2C and other on-policy comparators

**Hypothesis.** PPO's instability in V2 is algorithm-specific rather than
intrinsic to the temporal formulation.

**Sketch.** A2C or another on-policy learner, equal budget, bandit as control.

**Why it is not V2.** The mission constraint is explicit: **A2C must never be
added to the V2 comparison.** V2's holdout is spent; a fourth arm evaluated
against it would not be a holdout result.

**Status.** UNAPPROVED — NOT EVALUATED.

---

### V3-5 · Reward design sensitivity

**Hypothesis.** The myopic advantage is partly an artefact of a reward whose
movement penalties are charged immediately, favouring a controller that optimises
the current interval.

**Sketch.** Preregistered reward variants with delayed or amortised movement cost.

**Why it is not V2.** Changing `reward.yaml` changes a frozen definition, and V2's
conclusion is scoped to the reward as specified.

**Status.** UNAPPROVED — NOT EVALUATED.

---

### V3-6 · Topology generalization

**Hypothesis.** Nothing in V1 or V2 demonstrates transfer. A controller trained on
this 18-router backbone may not transfer to another topology.

**Sketch.** A topology family, train-on-some / evaluate-on-held-out-topologies.
Observation and action shapes become topology-dependent, which the current
fixed-shape design does not support.

**Why it is not V2.** Structural redesign.

**Status.** UNAPPROVED — NOT EVALUATED.

---

### V3-7 · More training roots for a magnitude claim

**Hypothesis.** V2 supports a direction claim (3/3 roots) but not a magnitude one
(margins ranged 5.746 to 12.250). More roots would bound the effect size.

**Sketch.** Additional preregistered roots, development-seed selection, a new
untouched holdout.

**Why it is not V2.** Extending the root set and re-running an evaluation is
reopening a closed study.

**Status.** UNAPPROVED — NOT EVALUATED.

---

## Rules any V3 must inherit from V2

These worked and should not be relitigated:

1. Freeze and sign off the environment definition before training; a pin test
   fails if a scientific definition moves under a trained checkpoint.
2. Preregister roots, scenarios, development seeds, evaluation seeds and the
   checkpoint-selection rule before any run.
3. Select only on development seeds. Evaluation seeds stay unconstructed and
   uninspected until one authorized run.
4. Keep a myopic control condition in every comparison that makes a claim about
   temporal reasoning.
5. Make the evaluation workflow accept **no** selection inputs, fail closed, and
   offer no retry path.
6. Preserve invalidated, superseded and failed runs and disclose all three as
   distinct statuses.
7. Verify the component-sum, mask, safety and integrity invariants on every step
   and publish the residuals.
8. Never commit checkpoints, model binaries, replay buffers, TensorBoard data,
   raw step logs or large datasets.
