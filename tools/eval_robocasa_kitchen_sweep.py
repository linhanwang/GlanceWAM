"""Sequential RoboCasa-kitchen (cosmos-policy) evaluation sweep across one or more checkpoints.

The kitchen analog of tools/eval_robocasa_tabletop_sweep.py — same server/client lifecycle,
discovery, and rolling-report machinery. Each checkpoint goes through:

  1. Launch the policy server (examples/Robocasa_kitchen/eval_files/run_policy_server.sh) in its
     own process group, with MAX_BATCH passed via env so concurrent client requests coalesce into
     one batched forward.
  2. Wait until the websocket port accepts connections (= model is loaded).
  3. Fan out the requested tasks as concurrent client sweeps
     (examples/Robocasa_kitchen/eval_files/start_robocasa_kitchen_env.sh), each splitting its
     task's n_episodes into `--shards` strided, per-episode-seeded partitions. All clients hit the
     ONE server, so the model loads once per server (not once per task).
  4. SIGTERM the server's process group (SIGKILL fallback).
  5. Read each task's per-shard JSON results, sum successes/episodes, and append a row to the
     markdown report (one column per task + Avg).

--gpus 1,2,4,5 fans ONE checkpoint across several GPUs: the 24 tasks are bin-packed by episode
horizon into one lane per GPU (own server, own port = --port + lane index), lanes run
concurrently, and results merge into a SINGLE report row. Required in practice for video-DiT
co-train checkpoints (SkyReelsV2*Cotrain): one server cannot batch all 24 tasks'
clients — measured OOM at 96 clients / 143 GB on an H200, vs ~60-80 GB at 24. See the skill doc.

Differences from the tabletop sweep (see .claude/skills/robocasa-kitchen-eval/SKILL.md):
- Tasks are the **24 kitchen task names** (bare, e.g. `TurnOffMicrowave`), not GR1 gym ids.
- **Per-task horizons.** The kitchen client derives its own `max_steps` per task (TASK_MAX_STEPS
  in simulation_env, 300-1000), so this orchestrator does NOT pass a global `max_episode_steps`.
- Client venv is the cosmos sim venv (`../robocasa-cosmos-policy/.venv`); `--robocasa-python`.
- Defaults: `n_action_steps=16` (action_horizon), `seed=195` (cosmos protocol). `unnorm_key`
  auto-resolves for a single-dataset checkpoint (pass `--unnorm-key` only for multi-dataset ckpts).

Checkpoints are selected from any combination of --ckpt / --glob / --manifest (additive, deduped).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob as globlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_FILES = REPO_ROOT / "examples" / "Robocasa_kitchen" / "eval_files"
SIM_ENV_PY = EVAL_FILES / "simulation_env.py"
SERVER_SH = EVAL_FILES / "run_policy_server.sh"
CLIENT_SH = EVAL_FILES / "start_robocasa_kitchen_env.sh"

DEFAULT_REPORT = REPO_ROOT / "examples" / "Robocasa_kitchen" / "eval_summary.md"
EVAL_DIR_NAME = "output_eval_robocasa_kitchen"

# The 24 cosmos-policy kitchen tasks (bare robosuite env names), matching
# examples/Robocasa_kitchen/train_files/data_registry/data_config.py _KITCHEN_TASKS.
KITCHEN_TASKS = [
    "CloseDoubleDoor",
    "CloseDrawer",
    "CloseSingleDoor",
    "CoffeePressButton",
    "CoffeeServeMug",
    "CoffeeSetupMug",
    "OpenDoubleDoor",
    "OpenDrawer",
    "OpenSingleDoor",
    "PnPCabToCounter",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "PnPCounterToSink",
    "PnPCounterToStove",
    "PnPMicrowaveToCounter",
    "PnPSinkToCounter",
    "PnPStoveToCounter",
    "TurnOffMicrowave",
    "TurnOffSinkFaucet",
    "TurnOffStove",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
    "TurnOnStove",
    "TurnSinkSpout",
]

# Named subset presets mirroring the train mixtures in
# examples/Robocasa_kitchen/train_files/data_registry/data_config.py.
# Pass e.g. `--tasks pnp` instead of the full comma-separated list.
TASK_PRESETS = {
    "24": list(KITCHEN_TASKS),
    "all": list(KITCHEN_TASKS),
    # robocasa_kitchen_pnp: the 8 counter<->cab/sink/stove/microwave pick-and-place tasks.
    "pnp": [t for t in KITCHEN_TASKS if t.startswith("PnP")],
}


def to_task(name: str) -> str:
    """Accept a bare task name (`TurnOffMicrowave`) or `robocasa/<Task>`; return the bare name."""
    return name.split("/", 1)[1] if "/" in name else name


def collect_checkpoints(ckpts, globs, manifest) -> list[Path]:
    paths: list[str] = []
    paths.extend(ckpts or [])
    for pattern in globs or []:
        matched = globlib.glob(pattern)
        if not matched:
            print(f"[warn] glob matched no files: {pattern}", file=sys.stderr)
        paths.extend(matched)
    if manifest:
        import yaml  # lazy import — only required when --manifest is used

        with open(manifest) as f:
            data = yaml.safe_load(f) or {}
        paths.extend(data.get("checkpoints", []))

    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        rp = str(Path(p).expanduser().resolve())
        if rp in seen:
            continue
        seen.add(rp)
        out.append(Path(rp))
    return out


def detect_include_state(ckpt: Path) -> bool | None:
    """Read datasets.vla_data.include_state from the checkpoint's saved run config.

    The train/eval state contract must match — a state-trained checkpoint evaluated stateless
    (or vice versa) silently degrades — so the orchestrator derives the flag from the experiment
    dir's config.yaml / config.full.yaml instead of trusting the operator. Returns None when no
    saved config is readable (caller falls back to --include-state).
    """
    import yaml  # lazy import, same as --manifest

    exp_dir = ckpt.parent.parent  # <exp>/checkpoints/steps_N_...pt -> <exp>
    for name in ("config.yaml", "config.full.yaml"):
        cfg_file = exp_dir / name
        if not cfg_file.is_file():
            continue
        try:
            cfg = yaml.safe_load(cfg_file.read_text())
            val = cfg["datasets"]["vla_data"]["include_state"]
        except Exception:
            continue
        if isinstance(val, str):
            val = val.strip().lower() in ("true", "1", "yes")
        return bool(val)
    return None


def wait_port_open(port: int, timeout: float) -> bool:
    """Return True once a TCP connect to localhost:port succeeds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


def wait_port_free(port: int, timeout: float = 60) -> bool:
    """Return True once a TCP connect to localhost:port fails (port is free)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                time.sleep(2)
        except OSError:
            return True
    return False


def kill_process_group(proc: subprocess.Popen, grace: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def parse_task(results_dir: Path, num_shards: int) -> dict:
    """Aggregate one task's shard JSONs into a success rate.

    Shards partition the task's episode_idx space disjointly, so the task rate is
    sum(successes) / sum(episodes) across shard files. Returns {mean, n_ok, n_shards, episodes}.
    """
    successes = episodes = 0
    n_ok = 0
    # Scope to THIS run's shard count so a stale shard_*_of_<other>.json from a prior run with a
    # different K isn't mixed in (the "(n/other)" inflation symptom).
    for f in sorted(results_dir.glob(f"shard_*_of_{num_shards}.json")):
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            print(f"  [warn] could not read {f.name}: {e}", file=sys.stderr)
            continue
        s, e_ = int(d.get("successes", 0)), int(d.get("episodes", 0))
        successes += s
        episodes += e_
        if e_ > 0:
            n_ok += 1
    mean = (successes / episodes) if episodes else None
    return {"mean": mean, "n_ok": n_ok, "n_shards": num_shards, "episodes": episodes}


def evaluate_one(
    ckpt: Path,
    tasks: list[str],
    port: int,
    gpu_id: int,
    server_timeout: float,
    max_batch: int,
    batch_wait_ms: float,
    shards: int,
    n_episodes: int,
    n_action_steps: int,
    seed: int,
    client_env_extra: dict[str, str],
    num_inference_timesteps: int | None = None,
    tag: str | None = None,
) -> dict[str, dict]:
    ckpt_dir = ckpt.parent
    eval_dir = ckpt_dir / EVAL_DIR_NAME
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Namespace outputs by checkpoint stem so multiple step-checkpoints of the SAME experiment
    # (which share one checkpoints/ dir) never collide on shard_*.json. When the flow-matching
    # Euler-step count is overridden, append it too so different step-settings of the SAME
    # checkpoint (e.g. a parallel nstep=1 vs nstep=2 comparison) don't clobber each other.
    # --tag does the same for eval-only ARMS of one checkpoint (goal degradations, staleness,
    # cadence): same weights, different inference-time env, so the stem alone would collide.
    ns = ckpt.stem if num_inference_timesteps is None else f"{ckpt.stem}__nstep{num_inference_timesteps}"
    if tag:
        ns = f"{ns}__{tag}"
    ckpt_eval_dir = eval_dir / ns

    if not wait_port_free(port, timeout=120):
        raise RuntimeError(f"port {port} still in use after 120s; aborting")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MAX_BATCH"] = str(max_batch)
    env["BATCH_WAIT_MS"] = str(batch_wait_ms)
    if num_inference_timesteps is not None:
        env["NUM_INFERENCE_TIMESTEPS"] = str(num_inference_timesteps)

    # ---- 1. start ONE batched server for the whole checkpoint (all tasks share it) ----
    server_proc = subprocess.Popen(
        ["bash", str(SERVER_SH), str(ckpt), str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"  server pid={server_proc.pid} (max_batch={max_batch}; waiting up to {server_timeout:.0f}s for port {port})")

    results: dict[str, dict] = {}
    clients: list[tuple[str, subprocess.Popen]] = []
    try:
        if not wait_port_open(port, timeout=server_timeout):
            raise RuntimeError(f"server failed to open port {port} within {server_timeout:.0f}s")
        print(f"  server ready on port {port}; launching {len(tasks)} task client sweep(s), {shards} shard(s) each")

        # ---- 2. fan out all tasks concurrently against the one server ----
        for task in tasks:
            task_dir = ckpt_eval_dir / task
            cenv = env.copy()
            cenv["results_dir"] = str(task_dir / "results")
            cenv["log_dir"] = str(task_dir / "logs")
            cenv["video_out_path"] = str(task_dir / "videos")
            cenv["host"] = "127.0.0.1"
            cenv["n_episodes"] = str(n_episodes)
            cenv["n_action_steps"] = str(n_action_steps)
            cenv["seed"] = str(seed)
            # NOTE: no max_episode_steps — the kitchen client derives it per task (TASK_MAX_STEPS).
            cenv.update(client_env_extra)
            proc = subprocess.Popen(
                ["bash", str(CLIENT_SH), str(ckpt), task, str(port), str(shards)],
                cwd=str(REPO_ROOT),
                env=cenv,
                start_new_session=True,
            )
            clients.append((task, proc))

        # ---- 3. wait for every task's client sweep ----
        for task, proc in clients:
            rc = proc.wait()
            if rc != 0:
                print(f"  [warn] task {task} client sweep exited with code {rc}", file=sys.stderr)
    finally:
        for _, proc in clients:
            kill_process_group(proc)
        print("  stopping server...")
        kill_process_group(server_proc)
        wait_port_free(port, timeout=60)

    # ---- 4. parse each task's shard JSONs ----
    for task in tasks:
        results[task] = parse_task(ckpt_eval_dir / task / "results", shards)
    return results


def task_horizons() -> dict[str, int]:
    """Per-task episode horizons, regex-read from simulation_env.py's TASK_MAX_STEPS.

    Parsed rather than imported: simulation_env imports robocasa/robosuite at module scope, which
    only exist in the sim venv — this orchestrator runs in the project .venv. Returns {} if the
    shape ever changes; callers fall back to an unweighted split.
    """
    try:
        blk = re.search(r"TASK_MAX_STEPS\s*[:=].*?\{(.*?)\n\}", SIM_ENV_PY.read_text(), re.S).group(1)
        return {m.group(1): int(m.group(2)) for m in re.finditer(r'"(\w+)"\s*:\s*(\d+)', blk)}
    except Exception:
        return {}


def bin_pack_tasks(tasks: list[str], n_lanes: int) -> list[list[str]]:
    """Split tasks across n_lanes, balancing each lane's summed episode horizon.

    Every task in a lane runs concurrently against that lane's server, and the sweep ends with the
    SLOWEST lane, so wall-clock tracks the per-lane horizon sum. Kitchen horizons span 300
    (CoffeePressButton) to 1000 (OpenDoubleDoor), so a naive round-robin leaves lanes lopsided and
    idles GPUs. Greedy longest-first packing gets max/min load ~1.03 on the 24-task/4-lane split.

    Note this balances TIME, not difficulty — per-lane success rates will still differ a lot (a
    measured 4-way split gave lane averages 0.77/0.75/0.69/0.49). That is expected and does not
    affect the merged row; do not read per-lane numbers as meaningful.
    """
    h = task_horizons()
    default = int(sum(h.values()) / len(h)) if h else 500
    lanes: list[list[str]] = [[] for _ in range(n_lanes)]
    load = [0] * n_lanes
    for t in sorted(tasks, key=lambda t: -h.get(t, default)):
        i = load.index(min(load))
        lanes[i].append(t)
        load[i] += h.get(t, default)
    return [lane for lane in lanes if lane]


def evaluate_multi_gpu(
    ckpt: Path,
    tasks: list[str],
    gpus: list[int],
    base_port: int,
    server_timeout: float,
    max_batch: int,
    **kw,
) -> dict[str, dict]:
    """Run one checkpoint across several GPUs: one server per GPU, a disjoint task subset each.

    Each lane is an independent evaluate_one (own server, own port = base_port + lane index, own
    CUDA_VISIBLE_DEVICES). Lanes run in threads — evaluate_one is subprocess-bound (it blocks in
    proc.wait()), so the GIL is never the constraint. Results merge into ONE report row, so a
    4-GPU sweep is indistinguishable from a 1-GPU sweep downstream.

    Why this exists: on a video-DiT co-train checkpoint the server is the bottleneck, and the
    single-GPU default (max_batch = ALL clients) OOMs — see --max-batch. Splitting tasks across
    GPUs cuts per-server batch AND multiplies throughput.
    """
    lanes = bin_pack_tasks(tasks, len(gpus))
    h = task_horizons()
    print(f"  {len(lanes)} lane(s) over GPU(s) {gpus} (horizon-balanced):")
    for i, lane in enumerate(lanes):
        load = sum(h.get(t, 0) for t in lane)
        print(
            f"    lane {i}: gpu={gpus[i]} port={base_port + i} horizon={load or '?'} n={len(lane)} :: {','.join(lane)}"
        )

    merged: dict[str, dict] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(lanes)) as ex:
        futs = {
            ex.submit(evaluate_one, ckpt, lane, base_port + i, gpus[i], server_timeout, max_batch=max_batch, **kw): i
            for i, lane in enumerate(lanes)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                merged.update(fut.result())
            except Exception as e:  # one lane dying must not strand the rest
                errors.append(f"lane {i} (gpu {gpus[i]}): {e}")
                print(f"  [warn] lane {i} on gpu {gpus[i]} failed: {e}", file=sys.stderr)
    if errors and not merged:
        raise RuntimeError("; ".join(errors))
    # A dead lane leaves its tasks absent; the report indexes results[t] for EVERY requested task,
    # so backfill them as empty. They render "—" (not 0.0) — a lane crash must not be averaged in
    # as if the policy had failed those tasks.
    missing = [t for t in tasks if t not in merged]
    for t in missing:
        merged[t] = {"mean": None, "n_ok": 0, "n_shards": kw.get("shards", 1), "episodes": 0}
    if errors:
        print(f"  [warn] {len(errors)} lane(s) failed; {len(missing)} task(s) will show '—'", file=sys.stderr)
    return merged


def split_ckpt(ckpt: Path) -> tuple[str, str]:
    """Return (exp_name, step) from .../<exp_name>/checkpoints/steps_<N>_pytorch_model.pt."""
    m = re.search(r"steps_(\d+)", ckpt.name)
    step = m.group(1) if m else "?"
    exp = ckpt.parent.parent.name if ckpt.parent.name == "checkpoints" else ckpt.parent.name
    return exp, step


def _fmt_cell(entry: dict) -> str:
    """Format one task's result as the success rate, or '—' if no shards reported.

    A `(n_ok/n_shards)` suffix is shown ONLY when some shards failed to report, so a crashed
    shard (which lowers the episode count) is flagged rather than silently folded into the number.
    """
    mean, n_ok, n_shards = entry["mean"], entry["n_ok"], entry["n_shards"]
    if mean is None:
        return "—"
    if n_ok == n_shards:
        return f"{mean:.3f}"
    return f"{mean:.3f} ({n_ok}/{n_shards} shards)"


def _write_header(f, tasks: list[str]) -> None:
    f.write("# RoboCasa kitchen (cosmos-policy) eval sweep\n\n")
    f.write("Each cell is the episode-weighted success rate over the task's strided shards (a single\n")
    f.write("pass over all episodes — shards just parallelize it). A `(n_ok/n_shards)` suffix appears\n")
    f.write("only if some shards failed to report. Avg is the unweighted mean across tasks.\n\n")
    header = ["Timestamp", "Exp", "Steps", "Avg", *tasks]
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join(["---"] * len(header)) + "|\n")


def _table_title(tasks: list[str]) -> str:
    """Heading for a task-set's table. The full 24-task set is the main table (no separate
    heading); narrower subsets get their own section so their column set doesn't misalign."""
    if tasks == TASK_PRESETS["pnp"]:
        return "pnp subset (robocasa_kitchen_pnp)"
    return f"{len(tasks)}-task subset"


def append_report(
    report_path: Path,
    tasks: list[str],
    ckpt: Path,
    results: dict[str, dict],
    num_inference_timesteps: int | None = None,
    tag: str | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    means = [results[t]["mean"] for t in tasks]
    valid = [v for v in means if v is not None]
    avg = sum(valid) / len(valid) if valid else None
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    cells = [_fmt_cell(results[t]) for t in tasks]
    avg_str = f"**{avg:.3f}**" if avg is not None else "—"
    exp, step = split_ckpt(ckpt)
    # Make the Euler-step override visible in the row so a step-count comparison on the same
    # checkpoint is legible (the Exp/Steps columns would otherwise be identical).
    if num_inference_timesteps is not None:
        exp = f"{exp} (nstep={num_inference_timesteps})"
    # Same for an eval-only arm tag: the Exp/Steps columns are identical across arms of one
    # checkpoint, so without this the report rows are indistinguishable.
    if tag:
        exp = f"{exp} [{tag}]"
    row = "| " + " | ".join([ts, exp, step, avg_str, *cells]) + " |"

    header_line = "| " + " | ".join(["Timestamp", "Exp", "Steps", "Avg", *tasks]) + " |"
    sep_line = "|" + "|".join(["---"] * (4 + len(tasks))) + "|"

    # Brand-new file: full preamble + this task set's table.
    if not report_path.exists():
        with report_path.open("w") as f:
            _write_header(f, tasks)
            f.write(row + "\n")
        return

    # Append the row to the table whose columns match THIS task set (matched by its exact header
    # line). A subset thus never lands in the 24-task table — each distinct task set keeps its own
    # table, created on first use.
    lines = report_path.read_text().splitlines()
    try:
        hidx = lines.index(header_line)
    except ValueError:
        hidx = -1
    if hidx >= 0:
        j = hidx + 2  # past header + separator
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        lines.insert(j, row)
        report_path.write_text("\n".join(lines) + "\n")
    else:
        block = [
            "",
            f"## {_table_title(tasks)}",
            "",
            "Same metric as the main table; columns are this run's task subset.",
            "",
            header_line,
            sep_line,
            row,
        ]
        with report_path.open("a") as f:
            f.write("\n".join(block) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", help="Explicit checkpoint paths")
    p.add_argument("--glob", nargs="*", help="Glob patterns (quote them so the shell doesn't expand)")
    p.add_argument("--manifest", help="YAML file with key 'checkpoints: [path, ...]'")
    p.add_argument(
        "--tasks",
        default=None,
        help="Task preset name (all/24/pnp) OR a comma-separated list of kitchen task names "
        "(bare, e.g. TurnOffMicrowave, or robocasa/<Task>). Default: all 24.",
    )
    p.add_argument("--port", type=int, default=6699, help="Server port. With --gpus, lane i uses --port + i.")
    p.add_argument("--gpu", type=int, default=0, help="Single-GPU mode. Ignored when --gpus is given.")
    p.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPUs to fan ONE checkpoint across, e.g. '1,2,4,5'. Tasks are split "
        "horizon-balanced into one lane per GPU (own server, own port = --port + lane index) and "
        "merged back into a SINGLE report row. Strongly recommended for video-DiT co-train "
        "checkpoints (SkyReelsV2*Cotrain), where one server cannot hold all 24 "
        "tasks' clients in a batch. Avoid GPUs listed in BAD_EGL_RENDER_GPUS (default empty, i.e. "
        "every GPU renders on itself) — the client launcher redirects their EGL rendering onto one "
        "fallback GPU, creating a render bottleneck.",
    )
    p.add_argument(
        "--server-timeout",
        type=float,
        default=900,
        help="Seconds to wait for the server port to open (model load can be slow).",
    )
    p.add_argument(
        "--shards",
        type=int,
        default=1,
        help="Strided client shards PER TASK (each runs a disjoint slice of the task's episodes). "
        "Total clients = len(tasks) x shards. Default 1 => one client per task (24 clients).",
    )
    p.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="Server-side batching: max requests coalesced per forward. Default = the clients ONE "
        "server must hold (all clients in single-GPU mode; the biggest lane's clients under "
        "--gpus). MEASURED: a SkyReelsV2 co-train server OOMs at batch 96 on a 143 GB H200 "
        "(VAE encode + 1.3B video DiT + flow-matching head per sample); batch 24 sits at "
        "~60-80 GB. Qwen*GR00T-2B checkpoints are far lighter. Lower this (or use --gpus) on "
        "OOM at the first batch.",
    )
    p.add_argument(
        "--batch-wait-ms",
        type=float,
        default=5.0,
        help="Max ms to wait for additional requests after the first arrives.",
    )
    p.add_argument(
        "--num-inference-timesteps",
        type=int,
        default=None,
        help="Override the flow-matching action head's Euler step count at inference (GR00T / PI "
        "heads). Unset => checkpoint default (typically 4). Outputs and the report row are "
        "namespaced by the value so a step-count comparison on one checkpoint doesn't collide.",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Short label for an eval-only ARM of a checkpoint (e.g. 'stale1chunk', 'goal_ds4'). "
        "Namespaces the output dir and is appended to the report's Exp cell as '[tag]'. Use it "
        "whenever the same weights are evaluated under different inference-time env knobs, or the "
        "rows are indistinguishable and the shard JSONs clobber each other.",
    )
    p.add_argument("--n-episodes", type=int, default=50, help="Rollouts per task (default 50).")
    p.add_argument(
        "--n-action-steps", type=int, default=16, help="Action chunk executed open-loop per query (default 16)."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=195,
        help="Base seed (cosmos protocol default 195); each episode gi is seeded seed*gi*256. Scene "
        "selection uses gi//10. Change it to evaluate a different fixed set of initial states.",
    )
    p.add_argument(
        "--unnorm-key",
        default=None,
        help="dataset_statistics.json key for action un-normalization. Omit to auto-resolve "
        "(single-dataset checkpoints, e.g. the released 'franka' key).",
    )
    p.add_argument("--save-video", action="store_true", help="Dump per-episode mp4s under each task's videos/.")
    p.add_argument(
        "--include-state",
        action="store_true",
        help="Send the raw 9-d proprio with each request. Normally UNNECESSARY: the orchestrator "
        "auto-detects datasets.vla_data.include_state from the checkpoint's saved run config. "
        "This flag is only a fallback for checkpoints with no readable config; passing it for a "
        "config-says-stateless checkpoint is a refused contract mismatch.",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Report markdown path (default examples/Robocasa_kitchen/eval_summary.md).",
    )
    p.add_argument(
        "--robocasa-python",
        default=None,
        help="Path to the kitchen sim venv python (py3.10). Sets ROBOCASA_python for the client "
        "subprocess. Default: ../robocasa-cosmos-policy/.venv/bin/python (set in the client script).",
    )
    p.add_argument(
        "--stop-on-error", action="store_true", help="Abort the whole sweep on first failure (default: continue)."
    )
    args = p.parse_args()

    if args.tasks and args.tasks.strip() in TASK_PRESETS:
        tasks = list(TASK_PRESETS[args.tasks.strip()])
    elif args.tasks:
        requested = [to_task(t.strip()) for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in requested if t not in KITCHEN_TASKS]
        if unknown:
            print(f"Unknown task(s): {unknown}\nValid tasks: {KITCHEN_TASKS}", file=sys.stderr)
            sys.exit(2)
        tasks = requested
    else:
        tasks = list(KITCHEN_TASKS)

    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""] if args.gpus else None
    if gpus is not None:
        if len(set(gpus)) != len(gpus):
            print(f"--gpus has duplicates: {gpus}", file=sys.stderr)
            sys.exit(2)
        if len(gpus) > len(tasks):
            print(f"--gpus lists {len(gpus)} GPUs but only {len(tasks)} task(s); using {len(tasks)}.")
            gpus = gpus[: len(tasks)]

    # max_batch is a PER-SERVER cap, so it must be sized by the clients ONE server sees. Under
    # --gpus that is the biggest lane, not the whole sweep — defaulting to the total would
    # reproduce the single-GPU OOM on every lane while using 1/N of each GPU's work.
    if gpus is not None:
        lane_sizes = [len(lane) for lane in bin_pack_tasks(tasks, len(gpus))]
        clients_per_server = max(lane_sizes) * args.shards
    else:
        clients_per_server = len(tasks) * args.shards
    total_clients = len(tasks) * args.shards
    max_batch = args.max_batch if args.max_batch is not None else clients_per_server

    client_env_extra: dict[str, str] = {}
    if args.robocasa_python:
        client_env_extra["ROBOCASA_python"] = args.robocasa_python
    if args.unnorm_key:
        client_env_extra["unnorm_key"] = args.unnorm_key
    if args.save_video:
        client_env_extra["save_video"] = "1"

    ckpts = collect_checkpoints(args.ckpt, args.glob, args.manifest)
    if not ckpts:
        print("No checkpoints selected. Pass --ckpt, --glob, or --manifest.", file=sys.stderr)
        sys.exit(2)

    nstep_note = (
        "" if args.num_inference_timesteps is None else f", num_inference_timesteps={args.num_inference_timesteps}"
    )
    where = f"gpus={gpus} (ports {args.port}..{args.port + len(gpus) - 1})" if gpus else f"gpu={args.gpu}"
    print(f"Sweep: {len(ckpts)} checkpoint(s), {len(tasks)} tasks, {where}{nstep_note}")
    print(
        f"       shards/task={args.shards}, total_clients={total_clients}, "
        f"clients/server={clients_per_server}, max_batch={max_batch}, report={args.report}"
    )
    for c in ckpts:
        print(f"  • {c}" if c.exists() else f"  [missing] {c}")

    failures: list[tuple[Path, str]] = []
    for i, ckpt in enumerate(ckpts, 1):
        print(f"\n=== [{i}/{len(ckpts)}] {ckpt} ===")
        if not ckpt.exists():
            print("  skipped (missing)")
            failures.append((ckpt, "missing"))
            continue
        cfg_state = detect_include_state(ckpt)
        if cfg_state is None:
            include_state = args.include_state
            if include_state:
                print("  include_state: no saved run config found — trusting --include-state")
        else:
            include_state = cfg_state
            if cfg_state and not args.include_state:
                print("  include_state: True (auto-detected from saved run config)")
            elif not cfg_state and args.include_state:
                print(
                    "  skipped: --include-state passed but the saved run config says the checkpoint "
                    "was trained stateless — refusing the mismatched eval"
                )
                failures.append((ckpt, "state contract mismatch"))
                continue
        ckpt_env_extra = dict(client_env_extra)
        if include_state:
            ckpt_env_extra["include_state"] = "1"
        common = dict(
            batch_wait_ms=args.batch_wait_ms,
            shards=args.shards,
            n_episodes=args.n_episodes,
            n_action_steps=args.n_action_steps,
            seed=args.seed,
            client_env_extra=ckpt_env_extra,
            num_inference_timesteps=args.num_inference_timesteps,
            tag=args.tag,
        )
        try:
            if gpus:
                results = evaluate_multi_gpu(
                    ckpt, tasks, gpus, args.port, args.server_timeout, max_batch=max_batch, **common
                )
            else:
                results = evaluate_one(
                    ckpt, tasks, args.port, args.gpu, args.server_timeout, max_batch=max_batch, **common
                )
            append_report(
                args.report, tasks, ckpt, results, num_inference_timesteps=args.num_inference_timesteps, tag=args.tag
            )
            valid = [results[t]["mean"] for t in tasks if results[t]["mean"] is not None]
            avg = sum(valid) / len(valid) if valid else None
            print(f"  Avg over {len(valid)}/{len(tasks)} tasks: {avg:.3f}" if avg is not None else "  Avg: —")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failures.append((ckpt, str(e)))
            if args.stop_on_error:
                raise

    print(f"\nReport: {args.report}")
    if failures:
        print(f"Failures: {len(failures)}")
        for c, why in failures:
            print(f"  {c}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()
