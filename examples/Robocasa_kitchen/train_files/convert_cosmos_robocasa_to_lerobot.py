#!/usr/bin/env python
"""Convert NVIDIA ``RoboCasa-Cosmos-Policy`` ``all_episodes`` per-episode HDF5 demos into
LeRobot v3.0, producing ONE dataset PER kitchen task (24 total),
``robocasa_<TaskName>_cosmos_lerobot`` (CamelCase task name == the RoboCasa eval ``--task_name``),
each carrying a ``success`` column for the dataloader filter.

Run in the dedicated converter venv (lerobot 0.4.4 + h5py + opencv), NOT the project ``.venv``::

    # one task (parallel-launch one process per task — each task ~54 eps, no aggregation needed):
    ~/envs/lerobot_convert/bin/python \
        examples/Robocasa_kitchen/train_files/convert_cosmos_robocasa_to_lerobot.py \
        --src /path/to/robocasa_cosmos --out-root results/Datasets --task PnPCounterToCab

    # all 24 tasks serially (single process):
    ... (omit --task)

This format differs from the LIBERO ``all_episodes`` converter:

* per-episode HDF5: datasets ``{primary,secondary,wrist}_images_jpeg`` (vlen-uint8, PIL/q95),
  ``proprio`` (T,9), ``actions`` (T,12); attrs ``success`` (bool), ``task_description`` (str).
* image    : ``cv2.imdecode`` (BGR) -> ``BGR2RGB``, NO flip (regen already stored ``flipud``-
             upright frames; the cosmos eval ``prepare_observation`` also flips, so train==eval).
             Note: PIL-encoded RGB at source, so the only swap needed is BGR->RGB from cv2.
* state(9) : cosmos-native order kept verbatim ``[gripper_qpos(2), eef_pos(3), eef_quat(4,xyzw)]``
             == ``prepare_observation``'s ``concat(gripper_qpos, eef_pos, eef_quat)`` (NO axis-angle
             conversion, unlike LIBERO) so the ported eval can feed proprio directly.
* action(7): ``actions[:, :7]`` = OSC pose delta(6) + gripper(1); drop the constant mobile-base
             tail ``actions[:, 7:12] == [0,0,0,0,-1]``. Gripper kept RAW (no {-1,+1}->{0,1} remap)
             so the model's output feeds straight to ``env.step`` after the 7->12 mobile-base pad.
* success  : HDF5 attr ``success`` -> extra ``success`` column patched into meta/episodes parquet.

Each episode is written once; the success-only training subset is just ``success==True`` (the
dataloader drops the rest), so no duplication. fps=20, robot_type=panda_omron, 224x224, v3.0.

One process converts one whole task (~54 eps), so the parallel launcher just fans out 24
processes — no shard/aggregate step (unlike the LIBERO converter, whose per-suite datasets
mix 10 tasks each and were big enough to shard).
"""

import argparse
import glob
import json
import re
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROBOT_TYPE = "panda_omron"
FPS = 20
RES = 224

FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (RES, RES, 3), "names": ["height", "width", "rgb"]},
    "observation.images.secondary_image": {
        "dtype": "video",
        "shape": (RES, RES, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {"dtype": "video", "shape": (RES, RES, 3), "names": ["height", "width", "rgb"]},
    "observation.state": {
        "dtype": "float32",
        "shape": (9,),
        "names": [
            "gripper_qpos_0",
            "gripper_qpos_1",
            "eef_x",
            "eef_y",
            "eef_z",
            "eef_quat_x",
            "eef_quat_y",
            "eef_quat_z",
            "eef_quat_w",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
    },
}

# gr00t_lerobot modality map (grouped, cosmos-native layout).
MODALITY = {
    "state": {
        "gripper_qpos": {"start": 0, "end": 2},
        "eef_pos": {"start": 2, "end": 5},
        "eef_quat": {"start": 5, "end": 9},
    },
    "action": {
        "eef_delta_pos": {"start": 0, "end": 3},
        "eef_delta_rot": {"start": 3, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "primary_image": {"original_key": "observation.images.image"},
        "secondary_image": {"original_key": "observation.images.secondary_image"},
        "wrist_image": {"original_key": "observation.images.wrist_image"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


def _ep_num(path: str) -> int:
    m = re.search(r"--ep=(\d+)--", path)
    return int(m.group(1)) if m else -1


def _list_tasks(src: Path) -> list[str]:
    return sorted(p.name for p in (src / "all_episodes").iterdir() if p.is_dir())


def _task_files(src: Path, task: str) -> list[str]:
    """One task's per-episode HDF5s, sorted by episode number (deterministic order)."""
    return sorted(glob.glob(f"{src}/all_episodes/{task}/*/*.hdf5"), key=_ep_num)


def _decode(jpeg_bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # no flip: frames already upright


def _convert_episode(ds: LeRobotDataset, path: str) -> bool:
    with h5py.File(path, "r") as f:
        task = f.attrs["task_description"]
        task = task.decode() if isinstance(task, bytes) else str(task)
        success = bool(f.attrs["success"]) if "success" in f.attrs else ("success=True" in path)
        pro = np.asarray(f["proprio"])  # (T, 9) = [gripper_qpos(2), eef_pos(3), eef_quat(4)]
        act = np.asarray(f["actions"])  # (T, 12) = [osc_pose(6), gripper(1), mobile_base(5)]
        prim = f["primary_images_jpeg"]
        sec = f["secondary_images_jpeg"]
        wrist = f["wrist_images_jpeg"]
        n = len(act)

        state = pro.astype(np.float32)  # keep native 9-dim order verbatim
        action = act[:, :7].astype(np.float32)  # drop constant mobile-base tail; gripper raw

        for t in range(n):
            ds.add_frame(
                {
                    "observation.images.image": _decode(prim[t]),
                    "observation.images.secondary_image": _decode(sec[t]),
                    "observation.images.wrist_image": _decode(wrist[t]),
                    "observation.state": state[t],
                    "action": action[t],
                    "task": task,
                }
            )
        ds.save_episode()
    return success


def _patch_success(out: Path, successes: list[bool]) -> None:
    """Add a per-episode ``success`` bool column to meta/episodes/*.parquet, keyed by episode_index."""
    files = sorted(out.glob("meta/episodes/*/*.parquet"))
    succ = np.asarray(successes, dtype=bool)
    patched = 0
    for fp in files:
        df = pd.read_parquet(fp)
        df["success"] = df["episode_index"].map(lambda i: bool(succ[int(i)]))
        df.to_parquet(fp, index=False)
        patched += len(df)
    assert patched == len(successes), f"patched {patched} != {len(successes)} episodes"


def _write_modality(out: Path) -> None:
    (out / "meta" / "modality.json").write_text(json.dumps(MODALITY, indent=4))


def convert_task(src: Path, out_root: Path, task: str, encoder_threads: int, limit) -> None:
    """Convert one whole kitchen task into its own dataset ``robocasa_<task>_cosmos_lerobot``."""
    files = _task_files(src, task)
    if limit:
        files = files[:limit]
    name = f"robocasa_{task}_cosmos"
    out = out_root / f"{name}_lerobot"
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=name,
        fps=FPS,
        features=FEATURES,
        root=str(out),
        robot_type=ROBOT_TYPE,
        use_videos=True,
        encoder_threads=encoder_threads,
    )
    # episode_index == order of save_episode calls == order of `files`, so the returned
    # success flags line up with episode_index directly (no recompute needed).
    successes = [_convert_episode(ds, path) for path in files]
    ds.finalize()
    _patch_success(out, successes)
    _write_modality(out)
    n_succ = int(np.sum(successes))
    print(f"[{task}] DONE -> {out}  ({n_succ} success / {len(successes)} total)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="cosmos-policy robocasa source dir")
    ap.add_argument("--out-root", type=Path, default=Path("results/Datasets"))
    ap.add_argument("--task", type=str, default=None, help="convert only this task (default: all 24)")
    ap.add_argument("--encoder-threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="convert only first N eps per task (smoke test)")
    args = ap.parse_args()

    tasks = [args.task] if args.task else _list_tasks(args.src)
    for task in tasks:
        convert_task(args.src, args.out_root, task, args.encoder_threads, args.limit)


if __name__ == "__main__":
    main()
