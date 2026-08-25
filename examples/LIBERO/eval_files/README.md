# LIBERO evaluation

Closed-loop eval of a GlanceWAM policy on **LIBERO 4-in-1** (`libero_spatial`, `libero_object`,
`libero_goal`, `libero_10`; 500 episodes per suite, 2000 total).

Evaluation is **two processes in two virtualenvs**: the policy server runs in this project's
`.venv` (it needs torch + the model), and the simulator client runs in LIBERO's own Python 3.8
venv. They talk over a websocket, so the sim venv needs no torch.

---

## 0. One-time sim-env setup

LIBERO is installed as a **sibling directory** of this repo:

```bash
bash examples/LIBERO/eval_files/install_libero.sh     # creates ../LIBERO with its own .venv
```

Override `LIBERO_HOME` / `LIBERO_python` if you keep it elsewhere. The client imports `glancewam`
(websocket client + `read_mode_config` only), which the launcher puts on `PYTHONPATH`.

## 1. Run a sweep (recommended)

`tools/eval_libero_sweep.py` handles the whole lifecycle: start the server, wait for the port,
fan out sharded clients across all four suites, tear down, and append a row to the report.

```bash
python tools/eval_libero_sweep.py \
    --ckpt results/Checkpoints/glancewam_libero/checkpoints/steps_15000_pytorch_model_ema.pt \
    --gpu 0
```

~35 min on one H200. The report lands in `examples/LIBERO/eval_summary.md`.

| flag | meaning |
|---|---|
| `--ckpt` / `--glob` / `--manifest` | checkpoint selection (several can be swept in sequence) |
| `--gpu` | GPU for the policy server (one server serves all four suites) |
| `--suites` | comma-separated subset, default all four |
| `--shard-budget` | total concurrent client shards, horizon-weighted across suites (default 32) |
| `--max-batch` | server-side request coalescing; defaults to the shard count |
| `--libero-python` / `--libero-home` | point at a LIBERO install outside the sibling default |

LIBERO is **stateless** — do not pass proprio flags. (RoboCasa kitchen is the opposite; see its
eval README.)

### Shards and batching

Each suite is split into strided `(task, episode)` shards run as concurrent client processes, all
hitting the one server so the model loads once. Shards are horizon-weighted by default: the
long-horizon `libero_10` gets more of them so every suite finishes together and the server batch
stays full. Lower `--shard-budget` / `--max-batch` on a smaller GPU.

## 2. Drive the two halves by hand

```bash
# terminal 1 — policy server (this repo's .venv)
bash examples/LIBERO/eval_files/run_policy_server.sh <ckpt.pt>

# terminal 2 — sim clients (LIBERO venv, resolved by the script)
bash examples/LIBERO/eval_files/start_libero_env.sh <ckpt.pt> libero_goal 6694 4
#                                                    ckpt     suite       port shards
```

## 3. Rendering notes

MuJoCo renders offscreen through EGL. `CUDA_VISIBLE_DEVICES` does **not** steer the EGL renderer
and EGL device order need not match `nvidia-smi` order, so `start_libero_env.sh` translates the
target index via `tools/egl_device_map.py` and exports `GLANCEWAM_EGL_DEVICE_ID`.

If some GPU's EGL context is wedged on your host (renders hang, `Xid 31`/`Xid 109` in dmesg), move
rendering off it without moving compute:

```bash
BAD_EGL_RENDER_GPUS=0,3 EGL_RENDER_FALLBACK_GPU=1 python tools/eval_libero_sweep.py ...
```

Both default to off — rendering happens on the same GPU as the server.

## Files

| file | role |
|---|---|
| `run_policy_server.sh` | policy server (this repo's `.venv`) |
| `start_libero_env.sh` | launches N sharded sim clients for one suite |
| `eval_libero.py` | the rollout client itself (runs in the LIBERO venv) |
| `model2libero_interface.py` | obs/action contract + websocket policy client |
| `adaptive_ensemble.py` | action-chunk ensembling across overlapping predictions |
| `install_libero.sh` | one-time sim-env installer |
