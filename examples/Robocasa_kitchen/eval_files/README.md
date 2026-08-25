# RoboCasa kitchen (cosmos-policy) evaluation

Closed-loop eval of a glancewam policy on the **cosmos-policy RoboCasa kitchen** benchmark
(24 tasks, single-arm Franka **PandaOmron**, OSC_POSE). The sim side is ported from
[cosmos-policy](https://github.com/nvidia-cosmos/cosmos-policy)'s
`run_robocasa_eval.py`; the policy runs in the glancewam websocket server, exactly like the
RoboCasa harnesses.

Train side: `examples/Robocasa_kitchen/train_files/` (`config_glancewam_robocasa_kitchen.yaml`,
`config_cotrain_baseline_robocasa_kitchen.yaml`). The eval obs/action contract mirrors the converter in
`train_files/convert_cosmos_robocasa_to_lerobot.py`.

---

## 0. One-time sim-env setup (separate py3.10 uv venv)

The simulator needs the `robocasa-cosmos-policy` fork + robosuite/mujoco, isolated from the
glancewam `.venv`. Cloned as a **sibling** and refactored to uv (same convention as
`../robocasa-gr1-tabletop-tasks`):

```bash
cd <parent-of-this-repo>          # the sim env is a SIBLING of the GlanceWAM checkout
git clone https://github.com/moojink/robocasa-cosmos-policy.git robocasa-cosmos-policy
cd robocasa-cosmos-policy
uv venv --python 3.10
uv pip install --python .venv/bin/python robosuite==1.5.1 mujoco==3.2.6 \
    tyro websockets msgpack rich opencv-python 'imageio[ffmpeg]'
uv pip install --python .venv/bin/python --no-deps -e .          # robocasa fork (skip its stale numpy/tianshou pins)
uv pip install --python .venv/bin/python numba scipy pygame Pillow pyyaml h5py lxml termcolor tqdm
uv pip install --python .venv/bin/python 'numpy==1.26.4'          # robosuite/mujoco are not numpy-2 safe
.venv/bin/python robocasa/scripts/setup_macros.py
yes | .venv/bin/python robocasa/scripts/download_kitchen_assets.py   # ~5 GB textures/fixtures/objects
```

The client imports glancewam (websocket client only) via `PYTHONPATH` — **no torch** is needed in
the sim venv because the policy runs server-side.

## 1. Contract reproduced from training (verify per checkpoint)

Embodied policies degrade **silently** on any train/eval mismatch. This harness reproduces:

| aspect | value | where |
| --- | --- | --- |
| cameras | 3 × 224², order **`[primary, secondary, wrist]`** (= `left, right, eye_in_hand`) | interface |
| image flip | `np.flipud` (RoboCasa renders upside-down; training stored upright) | `prepare_observation` |
| image jpeg | q95 round-trip (matches training `VideoJpeg`) | `interface._prep_image` |
| state | `[gripper_qpos(2), eef_pos(3), eef_quat(4 xyzw)]`, raw 9-d; **off by default** (stateless recipe) | `--args.include-state` |
| action | 7-d `[eef_Δpos(3), eef_Δrot(3), gripper(1 raw)]`; min_max-unnorm on the 6 pose dims, gripper raw | interface |
| action pad | 7 → 12 with mobile-base tail `[0,0,0,0,-1]` before `env.step` | `simulation_env` |
| open-loop | 16 steps/chunk (`action_horizon`) | `--args.n-action-steps` |
| scenes | `obj_instance_split=B`, 5 test layouts, scene = `episode_idx // 10` | `create_robocasa_env` |
| horizon | per-task `TASK_MAX_STEPS` (300–1000) | `simulation_env` |

## 2. Run a sweep (recommended)

`tools/eval_robocasa_kitchen_sweep.py` handles the whole lifecycle: start the server(s), wait for
the ports, fan the 24 tasks out as concurrent sharded clients, tear down, and append a row to
`examples/Robocasa_kitchen/eval_summary.md`.

```bash
python tools/eval_robocasa_kitchen_sweep.py \
    --ckpt results/Checkpoints/glancewam_robocasa_kitchen/checkpoints/steps_10000_pytorch_model_ema.pt \
    --gpus 0,1,2,3 --include-state
```

~45 min for 24 tasks x 50 episodes on 4 GPUs.

| flag | meaning |
|---|---|
| `--ckpt` / `--glob` / `--manifest` | checkpoint selection (several can be swept in sequence) |
| `--gpus 0,1,2,3` | bin-pack the 24 tasks into one lane per GPU by episode horizon, each lane with its own server and port; results merge into ONE report row |
| `--gpu` | single-GPU mode (ignored when `--gpus` is given) |
| `--include-state` | send proprio — **required** for the released kitchen checkpoint |
| `--tasks` | comma-separated subset of the 24 |
| `--n-episodes` | rollouts per task (default 50) |
| `--shards` | concurrent client processes per task |
| `--robocasa-python` | point at a sim venv outside the sibling default |

**`--gpus` is required in practice for video-DiT checkpoints.** One server cannot batch all 24
tasks' clients: measured OOM at 96 clients / 143 GB on an H200, versus ~60-80 GB at 24.

`--include-state` is the kitchen-specific trap — the released checkpoint conditions on proprio, and
omitting the flag degrades silently rather than erroring. (LIBERO is the opposite: stateless.)

## 3. Run the two processes by hand

```bash
# terminal 1 — policy server (glancewam .venv, GPU)
source .venv/bin/activate
bash examples/Robocasa_kitchen/eval_files/run_policy_server.sh <ckpt.pt> 6699

# terminal 2 — rollout client(s) (sim venv; PYTHONPATH handled by the launcher)
#   args: <ckpt_path> <task_name> <port> [num_shards]
bash examples/Robocasa_kitchen/eval_files/start_robocasa_kitchen_env.sh \
    <ckpt.pt> TurnOffMicrowave 6699 4
```

A checkpoint may carry a single stats key (`franka`,
the merged-mix embodiment tag), so `unnorm_key` **auto-resolves** — pass `unnorm_key=<key>` only if a
checkpoint stores stats for >1 dataset. Each shard runs a disjoint strided slice of the 50 episodes
and writes
`<model_root>/results/<task>/shard_<i>_of_<n>.json`; their union reproduces a full pass. Set
`save_video=1` for per-episode mp4s.

Direct single-shard invocation (debugging):

```bash
PYTHONPATH=$PWD MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=<egl_id> \
  ../robocasa-cosmos-policy/.venv/bin/python \
  examples/Robocasa_kitchen/eval_files/simulation_env.py \
    --args.pretrained-path <ckpt.pt> --args.task-name TurnOffMicrowave \
    --args.port 6699 --args.n-episodes 5   # --args.unnorm-key only needed for multi-dataset ckpts
```

## 4. Tasks

The 24 kitchen tasks (bare names or `robocasa/<Task>`) match `train_files/data_registry`:
`OpenDrawer CloseDrawer OpenSingleDoor CloseSingleDoor OpenDoubleDoor CloseDoubleDoor
PnP{CounterToCab,CabToCounter,CounterToSink,SinkToCounter,CounterToMicrowave,MicrowaveToCounter,CounterToStove,StoveToCounter}
TurnOn/Off{Stove,Microwave,SinkFaucet} TurnSinkSpout Coffee{SetupMug,ServeMug,PressButton}`.

The official protocol is 50 rollouts/task, `deterministic=True`, seeds {195,196,197}. This harness
defaults to `seed=195` with the cosmos per-episode seed formula; the sim-side render matches the
cosmos eval exactly (env kwargs are verbatim).
