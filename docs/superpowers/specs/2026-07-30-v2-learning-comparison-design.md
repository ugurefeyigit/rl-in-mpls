# V2 Seed-42 Learning Comparison Design

## Scope

Run the first governed learning comparison on the frozen MPLS-TE V2 problem:
MaskablePPO versus a masked neural contextual bandit, followed by deterministic
checkpoint selection and comparison with the repository's static, greedy, and
CSPF controllers. Only training root 42 and continuity evaluation seeds
101-105 are in scope. Holdout seeds 1001-1005 and training roots 314159 and
271828 are prohibited.

The sixteen files pinned to environment commit
`dca533b5c6fa9953307d01470c23cac512eb2961` remain untouched. V1 entry points,
behavior, artifacts, models, and results remain untouched. No UI work is in
scope.

## Chosen Architecture

Add an explicit V2-only experiment layer rather than widening the existing
mixed V1/V2 training script. The V2 CLI refuses any environment selector and
always constructs `MplsTeEnvV2` through
`mplssim.experiments.v2_factory.make_env_v2`. This makes V1 fallback
structurally impossible while preserving the existing V1 path byte-for-byte.

The experiment layer has five bounded responsibilities:

1. Run creation, immutable configuration, device resolution, hardware/library
   inventory, seed recording, hashes, and definition-pin verification.
2. A common learner protocol exposing masked prediction, checkpoint save/load,
   training diagnostics, and resolved device.
3. MaskablePPO integration using the existing SB3-contrib implementation and
   governed PPO hyperparameters.
4. A masked contextual bandit trained by supervised regression on immediate
   rewards for selected actions only.
5. Shared V2 evaluation and checkpoint selection for both learners and the
   existing baseline controller implementations.

All large artifacts live below ignored `runs/v2/` directories. Each run path
must not already exist. Concise manifests and reports may be copied into a
tracked results directory after the experiment.

## Shared Data and Integrity Flow

Every environment is wrapped outside the frozen V2 implementation. The wrapper
records every reset's root seed, worker rank, episode index, and derived episode
seed. It caches the authoritative pre-action mask, rejects any attempted invalid
action before delegation, checks the post-step mask against `info`, and verifies
the signed reward-component order sums bit-for-bit to the scalar reward.

Training records aggregate transitions, wall time, reward components, action
counts, no-op frequency, rejected attempts, mask disagreement count, solver
iterations, episode lengths, truncation, and algorithm diagnostics. Evaluation
adds the full interval metrics and available TE/FRR/restoration, churn, dwell,
traffic, delay, loss, congestion, and safety accounting. Metrics that the
frozen engine does not expose are explicitly reported as unavailable rather
than synthesized.

Each checkpoint has an algorithm-native payload and a JSON sidecar containing
the environment identity, training pin, code SHA, configuration, transition
count, seed policy, resolved device, and payload SHA-256. Loading verifies the
hash, algorithm, environment identity, and definition pin before inference.

## Algorithms

### MaskablePPO

Use `sb3_contrib.MaskablePPO` with the current governed values in
`configs/training.yaml`: learning rate `3e-4`, `n_steps=512`,
`batch_size=512`, eight epochs, `gamma=0.995`, `gae_lambda=0.95`,
`clip_range=0.2`, entropy coefficient `0.01`, value coefficient `0.5`,
gradient norm `0.5`, and `[256, 256]` policy networks. The shared callback
records integrity and metrics, saves at every 50,000 aggregate transitions,
and stops exactly at the aggregate budget. MaskablePPO receives authoritative
action masks during rollout and all inference paths.

### Masked Contextual Bandit

Use a `604 -> 256 -> 256 -> 69` ReLU MLP with Adam at `3e-4`. Store
`(observation, selected action, validity mask, immediate reward)` in a
100,000-entry replay buffer. Warm up for 4,096 aggregate transitions, then
perform one batch-512 Huber-loss update every four vector steps. Clip gradient
norm at 1.0.

Exploration is masked epsilon-greedy: epsilon decreases linearly from 0.20 to
0.02 over the first 200,000 aggregate transitions and remains at 0.02.
Deterministic inference is masked argmax. The loss gathers only the selected
action output. Its target is exactly the immediate observed V2 reward. Replay
contains no next observation, done flag, discount, target network, Bellman
term, or `max Q(s',a')`.

No broad tuning is performed after meaningful results are seen.

## Throughput and Device Selection

Record CPU, logical/physical cores, RAM, GPU, VRAM, Python, PyTorch, CUDA, and
library versions. `--device auto` resolves truthfully. Benchmark vector counts
8, 12, and 16 with short disposable runs for both learners. CUDA is compared
only if the installed PyTorch build exposes it; a visible NVIDIA device with a
CPU-only PyTorch build is recorded as unavailable rather than reported as
CUDA execution.

Use one vector count for both learners. Select the stable count with the best
combined aggregate transitions/second, except that a count incapable of
stopping at exactly 400,000 vectorized transitions is ineligible. Do not enable
mixed precision.

## Checkpoint Selection and Evaluation

Save checkpoints at aggregate transitions 50k, 100k, 150k, 200k, 250k, 300k,
350k, and 400k. Evaluate every checkpoint deterministically on all seven
approved scenarios and continuity seeds 101-105. Disqualify any checkpoint
with a metadata/hash, pin, mask, invalid-action, reward decomposition, solver,
or safety failure. Select the highest mean episodic operational return; exact
ties select the earlier checkpoint.

Evaluate the selected PPO and bandit checkpoints plus the existing `static`,
`greedy`, and `cspf` controllers on the identical scenario/seed matrix and
horizons. Existing baseline controllers run through a read-only V2 adapter;
because V2 accepts one action per interval, only the controller's first legal
proposal is submitted. The environment remains the single source of masks,
actions, rewards, transitions, and metrics.

## Failure Handling

Freeze drift, NaN/Inf, invalid actions, mask disagreement, reward mismatch,
solver failure, seed collision, protected-safety failure, corrupt checkpoints,
and metadata/hash mismatches stop the affected run. Failed run directories are
preserved. Only training-tooling or learner defects may be repaired
autonomously; the governed problem definition is never changed.

## Outputs and Stop Condition

Produce compressed machine-readable training steps, episode metrics, seed
records, checkpoint metadata, evaluation steps/summaries, learning curves,
comparison CSV/JSON, a pilot Markdown report, and a complete manifest. Finish
with `NEXT_STAGE_HANDOFF.md`, commit only concise code/config/report artifacts,
push the branch, and stop after the seed-42 comparison.

