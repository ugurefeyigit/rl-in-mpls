# Running it again

Exact commands, and the exact directory to run each one from. Nothing here
assumes you remember anything from last time.

---

## 0. Where "the repository root" is

Every command on this page runs from the directory that contains `server/`,
`mplssim/`, `frontend/` and `requirements.txt`. On the machine this release was
built on that is:

```
C:\Users\ugure\OneDrive\Masaüstü\rl_in_mpls
```

Check you are in the right place — this should list `server`, `mplssim`,
`frontend`, `docs`, `final`:

```bash
ls
```

If you are in a git worktree (a directory under `.claude/worktrees/`), that
worktree is its own complete copy of the repository and is also a valid root.

---

## 1. One-time setup

Skip this entirely if you have run the project before on this machine.

### Python

Python **3.11 or newer**. Check:

```bash
python --version
```

On Windows, `py` also works and is what the test commands below use.

### Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

On Windows, PyTorch installs separately for CPU. If the line above fails on
`torch`, install it first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

then re-run `pip install -r requirements.txt`.

### The V2 checkpoints (only if you want the learned controllers)

The two V2 learners — the masked contextual bandit and MaskablePPO — load frozen
checkpoints from the training worktrees. The application finds them
automatically under the main worktree's `.worktrees/` directory. If yours live
somewhere else, point at the directory that contains `seed42` and
`continuity_v2`:

```bash
export V2_LIVE_CHECKPOINTS=/path/to/checkpoints
```

On Windows PowerShell:

```powershell
$env:V2_LIVE_CHECKPOINTS = "C:\path\to\checkpoints"
```

**You do not need this to run the application.** Without it, the learners appear
in the control panel as *unavailable, with the exact verification reason*, and
the baselines (greedy, CSPF, static) still run normally. Nothing is ever
silently substituted.

---

## 2. Start the application

From the repository root:

```bash
python -m uvicorn server.main:app --port 8000
```

Leave that terminal open — it is the server. You will see one log line per
request, which is useful when something misbehaves.

Then open one of these in a browser:

| URL | Opens |
|---|---|
| <http://127.0.0.1:8000/present> | **Presentation Mode** — start here |
| <http://127.0.0.1:8000/advanced> | Network Information mode |
| <http://127.0.0.1:8000/study> | RL Information mode |
| <http://127.0.0.1:8000/docs> | the live OpenAPI reference |

All four serve the same application; they differ only in which mode opens first.
`Alt+1`, `Alt+2` and `Alt+3` switch between them without losing the run.

### If port 8000 is busy

Pick another port and use it in the URL:

```bash
python -m uvicorn server.main:app --port 8123
```

### Disabling training from the UI

The training endpoint is enabled by default. Before a demonstration, turn it off
so nobody can start a training job by accident:

```bash
ALLOW_TRAINING=false python -m uvicorn server.main:app --port 8000
```

PowerShell:

```powershell
$env:ALLOW_TRAINING = "false"; python -m uvicorn server.main:app --port 8000
```

---

## 3. Stop the application

In the terminal running the server, press **Ctrl+C**.

Everything the session held is in memory only. Stopping the server drops the
live run and every retained run — deliberately. If you want a run kept, press
**Save this run** in the control panel *before* stopping; that writes a summary
row to `results/runs.db`.

---

## 4. Run the tests

From the repository root. The full suite takes about two minutes.

```bash
py -m pytest -q
```

Expected: **811 passed, 0 failed**.

The suites added in Part 2, if you want them alone:

```bash
py -m pytest tests/test_part2_comparison.py tests/test_part2_results.py tests/test_part2_v2_endpoints.py -q
```

The presentation and control-panel contract:

```bash
py -m pytest tests/test_presentation_controls.py tests/test_product_ui.py -q
```

The guard that V1's models, results, figures, configs and simulation source are
still byte-identical to the audited base:

```bash
py -m pytest tests/test_v1_v2_compatibility.py -q
```

---

## 5. Docker, if you prefer it

From the repository root:

```bash
docker compose up
```

Then open <http://127.0.0.1:8000/present> as above.

---

## 6. When something is wrong

| Symptom | What it means | What to do |
|---|---|---|
| Browser shows "Unavailable" or "Connection lost" | the server is not running, or was restarted | check the server terminal; restart it |
| The two V2 learners are greyed out in the control panel | no verified checkpoint on this machine | read the reason printed under the picker — it names the exact check that failed. Set `V2_LIVE_CHECKPOINTS`, or use the baselines |
| `Start run` is disabled | the seed is invalid, or the selected controller is unavailable | the blocked reason is printed directly under the button |
| A red banner appears across the bottom | the last action failed | the banner carries the server's own reason. The most common one is the advisor delegation refusal — see `OPERATING_THE_UI.md` |
| Frozen holdout seeds 1001–1005 are refused | they are reserved for the closed study | pick any other whole number |
| `ModuleNotFoundError` on startup | dependencies not installed, or wrong directory | re-run step 1 from the repository root |

The server terminal is the first place to look: an application error prints a
full traceback there.
