"""Sequential LIBERO evaluation sweep across one or more checkpoints.

The LIBERO analog of tools/eval_simpler_sweep.py. Each checkpoint goes through the
same lifecycle:

  1. Launch the policy server (examples/LIBERO/eval_files/run_policy_server.sh) in its
     own process group, with MAX_BATCH passed via env so concurrent client requests
     coalesce into one batched forward.
  2. Wait until the websocket port accepts connections (= model is loaded).
  3. Fan out the requested task suites as concurrent client sweeps
     (examples/LIBERO/eval_files/start_libero_env.sh), each splitting
     its suite into `--shards` strided (task, episode) partitions. All clients hit the
     ONE server, so the model loads once per checkpoint (not once per suite).
  4. SIGTERM the server's process group (SIGKILL fallback).
  5. Read each suite's per-shard JSON results, sum successes/episodes, and append a row
     to the markdown report (one column per suite + Avg).

Checkpoints are selected from any combination of:
  --ckpt path1 path2 ...
  --glob 'results/Checkpoints/*/checkpoints/steps_*_pytorch_model.pt'
  --manifest sweep.yaml      # YAML with key 'checkpoints: [path, ...]'

Concurrency: each suite is split into strided (task, episode) shards run as concurrent
client processes, each issuing one batched predict_action per action chunk. By DEFAULT
shards are horizon-weighted across suites via a total budget (DEFAULT_SHARD_BUDGET=32):
longer-horizon suites (libero_10) get more shards so all suites finish together and the
server batch stays full (no under-batched tail). --shards forces an equal per-suite split
instead; --shard-budget sets a custom weighted budget. --max-batch defaults to the total
shard count so one forward serves every shard. On an H200 the default 32 fits with
headroom; lower the budget / --max-batch on smaller GPUs.
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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_FILES = REPO_ROOT / "examples" / "LIBERO" / "eval_files"
SERVER_SH = EVAL_FILES / "run_policy_server.sh"
CLIENT_SH = EVAL_FILES / "start_libero_env.sh"

DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_REPORT = REPO_ROOT / "examples" / "LIBERO" / "eval_summary.md"
EVAL_DIR_NAME = "output_eval_libero"
# Default total shard budget, distributed across suites by horizon (see allocate_shards).
# Used when neither --shards nor --shard-budget is given. 32 fits an H200 with headroom.
DEFAULT_SHARD_BUDGET = 32

# Per-suite episode horizon (max_steps), mirroring examples/LIBERO/eval_files/eval_libero.py.
# Used by --shard-budget to weight shard counts by horizon so longer-horizon suites get more
# shards and all suites finish around the same time (no under-batched tail). It is a ceiling,
# not the actual per-episode time (successful episodes terminate early), so balancing is
# approximate but removes the bulk of the tail.
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def allocate_shards(suites: list[str], budget: int) -> dict[str, int]:
    """Distribute `budget` shards across suites proportional to each suite's max_steps.

    Largest-remainder apportionment with a floor of 1 shard per suite. When budget >= len(suites)
    the allocation sums exactly to budget; if budget < len(suites) every suite still gets 1 (the
    sum then exceeds budget — the caller sizes max_batch off the actual sum).
    """
    w = {s: SUITE_MAX_STEPS.get(s, 300) for s in suites}
    tw = sum(w.values())
    raw = {s: budget * w[s] / tw for s in suites}
    alloc = {s: max(1, int(raw[s])) for s in suites}
    short = budget - sum(alloc.values())
    if short > 0:
        # Hand out the remaining shards by largest fractional remainder.
        order = sorted(suites, key=lambda s: raw[s] - int(raw[s]), reverse=True)
        for i in range(short):
            alloc[order[i % len(order)]] += 1
    return alloc


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


def parse_suite(results_dir: Path, num_shards: int) -> dict:
    """Aggregate one suite's shard JSONs into a success rate.

    Shards partition the (task_id, episode_idx) space disjointly, so the suite rate is
    sum(successes) / sum(episodes) across shard files. Returns {mean, n_ok, n_shards,
    episodes}: n_ok = shards that reported >0 episodes, n_shards = expected shard count.
    """
    successes = episodes = 0
    n_ok = 0
    # Scope to THIS run's shard count: a results dir may retain shard_*_of_K.json from a
    # prior run with a different K (e.g. --shards 8 then 4); globbing bare shard_*.json
    # would mix them and inflate counts (the "(8/4)" symptom).
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
    suites: list[str],
    port: int,
    gpu_id: int,
    server_timeout: float,
    max_batch: int,
    batch_wait_ms: float,
    suite_shards: dict[str, int],
    num_trials_per_task: int,
    client_env_extra: dict[str, str],
) -> dict[str, dict]:
    ckpt_dir = ckpt.parent
    eval_dir = ckpt_dir / EVAL_DIR_NAME
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Namespace outputs by checkpoint stem so multiple step-checkpoints of the SAME
    # experiment (which share one checkpoints/ dir) never collide on shard_*.json when
    # swept concurrently across GPUs. (One fix is prefixing filenames
    # with the ckpt stem; here we add a ckpt-stem subdir under eval_dir.)
    ckpt_eval_dir = eval_dir / ckpt.stem

    if not wait_port_free(port, timeout=120):
        raise RuntimeError(f"port {port} still in use after 120s; aborting")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MAX_BATCH"] = str(max_batch)
    env["BATCH_WAIT_MS"] = str(batch_wait_ms)

    # ---- 1. start ONE batched server for the whole checkpoint (all suites share it) ----
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
        alloc_str = ", ".join(f"{s}={suite_shards[s]}" for s in suites)
        print(f"  server ready on port {port}; launching {len(suites)} suite client sweep(s): {alloc_str}")

        # ---- 2. fan out all suites concurrently against the one server ----
        for suite in suites:
            results_dir = ckpt_eval_dir / suite / "results"
            cenv = env.copy()
            cenv["results_dir"] = str(results_dir)
            cenv["log_dir"] = str(ckpt_eval_dir / suite / "logs")
            cenv["video_out_path"] = str(ckpt_eval_dir / suite / "videos")
            cenv["host"] = "127.0.0.1"
            cenv["num_trials_per_task"] = str(num_trials_per_task)
            cenv.update(client_env_extra)
            proc = subprocess.Popen(
                ["bash", str(CLIENT_SH), str(ckpt), suite, str(port), str(suite_shards[suite])],
                cwd=str(REPO_ROOT),
                env=cenv,
                start_new_session=True,
            )
            clients.append((suite, proc))

        # ---- 3. wait for every suite's client sweep ----
        for suite, proc in clients:
            rc = proc.wait()
            if rc != 0:
                print(f"  [warn] suite {suite} client sweep exited with code {rc}", file=sys.stderr)
    finally:
        # Kill any still-running client sweeps (e.g. on KeyboardInterrupt), then the server.
        for _, proc in clients:
            kill_process_group(proc)
        print("  stopping server...")
        kill_process_group(server_proc)
        wait_port_free(port, timeout=60)

    # ---- 4. parse each suite's shard JSONs ----
    for suite in suites:
        results[suite] = parse_suite(ckpt_eval_dir / suite / "results", suite_shards[suite])
    return results


def split_ckpt(ckpt: Path) -> tuple[str, str]:
    """Return (exp_name, step) from .../<exp_name>/checkpoints/steps_<N>_pytorch_model.pt."""
    m = re.search(r"steps_(\d+)", ckpt.name)
    step = m.group(1) if m else "?"
    exp = ckpt.parent.parent.name if ckpt.parent.name == "checkpoints" else ckpt.parent.name
    return exp, step


def _fmt_cell(entry: dict) -> str:
    """Format one suite's result as just the success rate, or '—' if no shards reported.

    Unlike sweeps that run each task multiple times and show a run count, LIBERO
    shards merely partition a single pass over the suite's episodes, so the shard count is
    not meaningful signal — we omit it. A `(n_ok/n_shards)` suffix is shown ONLY when some
    shards failed to report, so a crashed shard (which lowers the episode count) is flagged
    rather than silently folded into the number.
    """
    mean, n_ok, n_shards = entry["mean"], entry["n_ok"], entry["n_shards"]
    if mean is None:
        return "—"
    if n_ok == n_shards:
        return f"{mean:.3f}"
    return f"{mean:.3f} ({n_ok}/{n_shards} shards)"


def _write_header(f, suites: list[str]) -> None:
    f.write("# LIBERO eval sweep\n\n")
    f.write("Each cell is the episode-weighted success rate over the suite's strided shards (a single\n")
    f.write("pass over all episodes — shards just parallelize it, so no run/shard count is shown). A\n")
    f.write("`(n_ok/n_shards)` suffix appears only if some shards failed to report.\n\n")
    header = ["Timestamp", "Exp", "Steps", *suites, "Avg"]
    f.write("| " + " | ".join(header) + " |\n")
    f.write("|" + "|".join(["---"] * len(header)) + "|\n")


def append_report(report_path: Path, suites: list[str], ckpt: Path, results: dict[str, dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    means = [results[s]["mean"] for s in suites]
    valid = [v for v in means if v is not None]
    avg = sum(valid) / len(valid) if valid else None
    new_file = not report_path.exists()
    with report_path.open("a") as f:
        if new_file:
            _write_header(f, suites)
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        cells = [_fmt_cell(results[s]) for s in suites]
        avg_str = f"**{avg:.3f}**" if avg is not None else "—"
        exp, step = split_ckpt(ckpt)
        f.write("| " + " | ".join([ts, exp, step, *cells, avg_str]) + " |\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", help="Explicit checkpoint paths")
    p.add_argument("--glob", nargs="*", help="Glob patterns (quote them so the shell doesn't expand)")
    p.add_argument("--manifest", help="YAML file with key 'checkpoints: [path, ...]'")
    p.add_argument(
        "--suites",
        default=",".join(DEFAULT_SUITES),
        help=f"Comma-separated task suites to evaluate (default: {','.join(DEFAULT_SUITES)}).",
    )
    p.add_argument("--port", type=int, default=6694)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--server-timeout",
        type=float,
        default=900,
        help="Seconds to wait for the server port to open (model load can be slow).",
    )
    shard_grp = p.add_mutually_exclusive_group()
    shard_grp.add_argument(
        "--shards",
        type=int,
        default=None,
        help="Flat per-suite strided (task, episode) shards = concurrent client procs per suite. "
        "Total clients = len(suites) x shards. Use this to force an equal split; otherwise the "
        "horizon-weighted default applies.",
    )
    shard_grp.add_argument(
        "--shard-budget",
        type=int,
        default=None,
        help=f"Total shard budget distributed across suites proportional to each suite's max_steps "
        f"(horizon-weighted), so longer-horizon suites (e.g. libero_10) get more shards and all suites "
        f"finish together — removes the under-batched tail. THIS IS THE DEFAULT: with no shard flag the "
        f"budget is {DEFAULT_SHARD_BUDGET} (over the default 4 suites -> spatial=5, object=7, goal=7, "
        f"libero_10=13).",
    )
    p.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="Server-side batching: max requests coalesced per forward. Default = total shards across "
        "suites so one forward serves every shard. Drop on small GPUs (per the simpler-eval per-GPU table).",
    )
    p.add_argument(
        "--batch-wait-ms",
        type=float,
        default=5.0,
        help="Max ms to wait for additional requests after the first arrives.",
    )
    p.add_argument("--num-trials-per-task", type=int, default=50, help="Rollouts per task (default 50).")
    p.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Report markdown path (default examples/LIBERO/eval_summary.md).",
    )
    p.add_argument(
        "--libero-python",
        default=None,
        help="Path to the LIBERO client venv python (py3.10). Sets LIBERO_python for the client subprocess.",
    )
    p.add_argument("--libero-home", default=None, help="Path to the LIBERO repo. Sets LIBERO_HOME for the client.")
    p.add_argument(
        "--stop-on-error", action="store_true", help="Abort the whole sweep on first failure (default: continue)."
    )
    args = p.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    if not suites:
        print("No suites selected via --suites.", file=sys.stderr)
        sys.exit(2)

    # Per-suite shard counts. Default = horizon-weighted budget (DEFAULT_SHARD_BUDGET);
    # --shards forces an equal split; --shard-budget sets a custom weighted budget.
    if args.shards is not None:
        suite_shards = {s: args.shards for s in suites}
    else:
        budget = args.shard_budget if args.shard_budget is not None else DEFAULT_SHARD_BUDGET
        suite_shards = allocate_shards(suites, budget)
    total_shards = sum(suite_shards.values())
    max_batch = args.max_batch if args.max_batch is not None else total_shards

    client_env_extra: dict[str, str] = {}
    if args.libero_python:
        client_env_extra["LIBERO_python"] = args.libero_python
    if args.libero_home:
        client_env_extra["LIBERO_HOME"] = args.libero_home

    ckpts = collect_checkpoints(args.ckpt, args.glob, args.manifest)
    if not ckpts:
        print("No checkpoints selected. Pass --ckpt, --glob, or --manifest.", file=sys.stderr)
        sys.exit(2)

    shard_desc = ", ".join(f"{s}={suite_shards[s]}" for s in suites)
    print(f"Sweep: {len(ckpts)} checkpoint(s), suites={suites}, port={args.port}, gpu={args.gpu}")
    print(f"       shards [{shard_desc}] (total={total_shards}), max_batch={max_batch}, report={args.report}")
    for c in ckpts:
        print(f"  • {c}" if c.exists() else f"  [missing] {c}")

    failures: list[tuple[Path, str]] = []
    for i, ckpt in enumerate(ckpts, 1):
        print(f"\n=== [{i}/{len(ckpts)}] {ckpt} ===")
        if not ckpt.exists():
            print("  skipped (missing)")
            failures.append((ckpt, "missing"))
            continue
        try:
            results = evaluate_one(
                ckpt,
                suites,
                args.port,
                args.gpu,
                args.server_timeout,
                max_batch=max_batch,
                batch_wait_ms=args.batch_wait_ms,
                suite_shards=suite_shards,
                num_trials_per_task=args.num_trials_per_task,
                client_env_extra=client_env_extra,
            )
            append_report(args.report, suites, ckpt, results)
            pretty = ", ".join(f"{s}={_fmt_cell(results[s])}" for s in suites)
            print(f"  results: {pretty}")
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
