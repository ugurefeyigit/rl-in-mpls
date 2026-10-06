# Reproducibility

## 1. Environment

| Item | Tested value | Notes |
|---|---|---|
| OS | Linux 6.18 (x86-64) | The closed V2 study ran on Windows 11; the simulator is platform-independent (§5). |
| Python | 3.13.16 | ≥ 3.11 required |
| PyTorch | 2.14.1+cpu | CPU build is sufficient; no GPU is needed for any result in the paper |
| Stable-Baselines3 / SB3-contrib | 2.9.0 / 2.9.0 | MaskablePPO |
| Others | see `requirements/lock-cpu-py313.txt` | exact versions used for every post-V2 result |
| Solver | SciPy HiGHS (bundled with SciPy ≥ 1.9) | only for the MILP-track baseline; no commercial solver |
| Hardware | 4-core Intel Xeon @ 2.1 GHz, 15 GB RAM, no GPU | runtimes below are for this machine |

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/lock-cpu-py313.txt   # exact versions
python -m pip install -e . --no-deps                       # makes `mplssim` importable
```

`requirements.txt` (lower bounds) is kept for the original V1/UI workflow.
A `Dockerfile` exists for the dashboard; the research pipeline needs only the
Python environment above.

## 2. One-command entry points

| Command | What it does | Runtime (4 cores) |
|---|---|---|
| `python scripts/reproduce.py --suite sanity` | fast tests; exact baseline reproduction of the closed study's holdout table; 8,192-transition training + evaluation of both learners | ≈ 10 min |
| `python scripts/reproduce.py --suite core` | one additional training root of both learners (400k each), all reference policies on the 140 test episodes, one diagnostic seed, tables and figures | ≈ 3–4 h |
| `python scripts/reproduce.py --suite full` | every registered run (`experiments/registry/*.yaml`), every reference policy, both sequentiality diagnostics, the mask audit, tables, figures, paper | ≈ 20–24 h |
| `python scripts/reproduce.py --suite analysis` | rebuild processed tables, `paper/generated/numbers.tex`, figures and the PDF from existing raw results | ≈ 2 min |

Every step is an ordinary script (`scripts/study/*.py`) and is idempotent:
completed training runs and cached evaluations are skipped, so an interrupted
suite can be restarted.

## 3. Seeds and randomness

* Episode seeds: `root + worker_rank + 1024·episode_index` (training);
  explicit evaluation seeds from `mplssim/study/protocol.py`
  (validation 2001–2005, test 3001–3020, diagnostic 4001–4003; historical
  continuity 101–105 and holdout 1001–1005).
* The traffic process is driven by two independent generators per episode
  (`SeedSequence([episode_seed, 1|2])`); no routing decision consumes them, so
  every controller sees byte-identical traffic on a given episode.
* Learner randomness: network initialization (`torch.manual_seed(root)` inside
  `fork_rng`), ε-greedy and replay sampling (`numpy.default_rng(root)`), PPO
  via SB3 `seed=root`. On CPU with one thread, runs are deterministic: the
  study trainer reproduces the governed trainer's network parameters bit for bit
  (`pytest -m slow tests/test_study_trainer_equivalence.py`).

## 4. Reproducing the closed V2 study (E0)

The closed study's checkpoints were not committed. To retrain them exactly as
governed (clean checkout required by the trainer):

```bash
git worktree add --detach .worktrees/repro 1457e9b
cd .worktrees/repro
python scripts/train_v2.py --algorithm masked_bandit --run-dir runs/repro/seed42_masked_bandit \
    --purpose meaningful --root-seed 42 --n-envs 16 --device cpu      # ≈ 45–90 min
# … likewise maskable_ppo, and roots 314159, 271828
cd ../..
python scripts/study/eval_repro.py .worktrees/repro/runs/repro/seed42_masked_bandit
```

`eval_repro.py` evaluates each run under the historical protocol (select on
101–105, test on 1001–1005) and the post-V2 protocol (select on 2001–2005, test
on 3001–3020). The original runs used CUDA; neural training is not
bit-reproducible across devices, so learner results reproduce statistically,
not exactly (`docs/REPRODUCTION_REPORT.md`).

## 5. What reproduces exactly

* Simulator and baselines: all 21 (policy, scenario) holdout means and SDs of
  static/greedy/CSPF match the closed study's Windows/CUDA-machine table to
  ≤ 3·10⁻¹⁴ (`python scripts/study/check_baseline_reproduction.py`).
* Learners on CPU, fixed root: bit-identical across repeated runs and across the
  two trainers.

## 6. Artifact layout

| Path | Content | Versioned |
|---|---|---|
| `experiments/registry/` | experiment definitions | yes |
| `experiments/raw/` | per-episode evaluation outputs, diagnostics, mask audit, run manifests, per-episode training logs | yes (small) |
| `experiments/processed/` | tidy tables built by `scripts/study/analyze.py` | yes |
| `results/tables/`, `results/figures/paper/` | publication tables and figures | yes |
| `results/v2_*`, `results/environment_v2_validation/` | closed study's compact evidence (historical; untouched) | yes |
| `runs/`, `.worktrees/` | checkpoints, replay buffers, full step logs | no (`.gitignore`) |
