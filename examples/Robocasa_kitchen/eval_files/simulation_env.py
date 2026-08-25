"""RoboCasa kitchen (cosmos-policy, PandaOmron single-arm) evaluation against a glancewam
websocket policy server.

Ported from cosmos-policy's
``cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py`` — env construction
(``create_robocasa_env``), observation prep (``prepare_observation``), the episode loop, and the
per-task ``TASK_MAX_STEPS`` horizons are kept verbatim so the eval distribution matches the data
the checkpoint was converted from. The in-process cosmos model is swapped for the glancewam
``PolicyWarper`` websocket client; sharding + per-shard JSON results mirror
the RoboCasa kitchen task suite.

Runs in the ``robocasa-cosmos-policy`` sim venv (py3.10, robosuite==1.5.1 / mujoco==3.2.6);
glancewam rides in via ``PYTHONPATH`` (see ``start_robocasa_kitchen_env.sh``). The client does NO
CUDA compute — the policy runs server-side.
"""

from __future__ import annotations

import ast
import itertools
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import robocasa  # noqa: F401  (registers the PandaMobile kitchen tasks)
import robosuite
import tyro
from robocasa.utils.dataset_registry import MULTI_STAGE_TASK_DATASETS, SINGLE_STAGE_TASK_DATASETS

from examples.Robocasa_kitchen.eval_files.model2robocasa_kitchen_interface import PolicyWarper

# Fixed OSC_POSE controller config shipped next to this file (copied from cosmos-policy).
CONTROLLER_CONFIGS_PATH = str(Path(__file__).resolve().parent / "robocasa_controller_configs.pkl")

# Per-task episode horizons — verbatim from cosmos-policy run_robocasa_eval.TASK_MAX_STEPS.
TASK_MAX_STEPS = {
    # Pick and place
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPSinkToCounter": 500,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    # Door
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    # Drawer
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    # Stove
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    # Sink
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    # Coffee
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    # Microwave
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
}

MOBILE_BASE_TAIL = np.array([0.0, 0.0, 0.0, 0.0, -1.0])  # 7->12 pad (base unused, gripper-open convention)
NUM_STEPS_WAIT = 10  # dummy steps to let objects settle before the first policy query

TEST_SCENES = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))  # official held-out eval scenes (robocasa eval_utils)
ALL_LAYOUTS = range(10)  # LayoutType 0-9
ALL_STYLES = range(12)  # StyleType 0-11


def resolve_scene_pool(spec: str) -> Optional[List[tuple]]:
    """Resolve ``--layout_and_style_ids`` into a list of (layout, style) combos.

    ``""`` -> None (env samples any scene, INCLUDING the test 5); ``"test"`` -> the 5 held-out eval
    scenes; ``"train"`` -> all 120 combos minus the test 5 (115 training scenes); anything else is
    parsed as a literal tuple string like the default.
    """
    if not spec:
        return None
    if spec == "test":
        return list(TEST_SCENES)
    if spec == "train":
        return [c for c in itertools.product(ALL_LAYOUTS, ALL_STYLES) if c not in set(TEST_SCENES)]
    return [tuple(x) for x in ast.literal_eval(spec)]


def prepare_observation(obs: Dict, flip_images: bool = True) -> Dict:
    """Extract the 3 cameras + 9-d proprio from a robosuite obs dict, matching training.

    RoboCasa renders images upside-down, so ``flip_images`` (default True) applies ``np.flipud``
    — the training data was stored already-upright, so eval must flip to match.
    """

    def _cam(key):
        img = obs[key]
        return np.flipud(img) if flip_images else img

    return {
        "primary_image": _cam("robot0_agentview_left_image"),
        "secondary_image": _cam("robot0_agentview_right_image"),
        "wrist_image": _cam("robot0_eye_in_hand_image"),
        # cosmos-native order kept verbatim: [gripper_qpos(2), eef_pos(3), eef_quat(4, xyzw)].
        "proprio": np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])),
    }


def create_robocasa_env(
    task_name: str,
    seed: Optional[int],
    episode_idx: Optional[int],
    env_img_res: int,
    robots: str,
    obj_instance_split: str,
    layout_and_style_ids_str: str,
    randomize_cameras: bool,
):
    """Build a robosuite RoboCasa env with the official cosmos eval kwargs. A fresh env is made per
    episode so the scene is deterministically selected by ``episode_idx // 10`` (5 held-out scenes)."""
    if layout_and_style_ids_str:
        all_ids = ast.literal_eval(layout_and_style_ids_str)
        if episode_idx is not None:
            scene = (all_ids[(episode_idx // 10) % len(all_ids)],)
        else:
            scene = all_ids
    else:
        scene = None

    with open(CONTROLLER_CONFIGS_PATH, "rb") as fh:
        controller_configs = pickle.load(fh)

    env = robosuite.make(
        env_name=task_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=env_img_res,
        camera_heights=env_img_res,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=None,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=scene,
        translucent_robot=False,
    )
    return env


def run_episode(env, model, task_description, max_steps, flip_images, collect_frames):
    """One rollout: warmup, then open-loop chunks until success or ``max_steps``.

    The open-loop chunk length is owned by the interface (``model.n_action_steps``), so the queue
    simply drains whatever ``get_action`` returns before requerying."""
    # Per-episode reset of any policy-side state (e.g. the goal-image cotrain's H_g refresh
    # cycle). No-op for stateless policies (reset just re-sets the task description).
    if hasattr(model, "reset"):
        model.reset(task_description)
    obs = None
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape))

    success = False
    frames: List[np.ndarray] = []
    action_dim = env.action_dim
    queue: List[np.ndarray] = []
    length = max_steps
    for step in range(max_steps):
        observation = prepare_observation(obs, flip_images)
        if collect_frames:
            frames.append(observation["primary_image"])
        if not queue:  # requery when the open-loop chunk is exhausted
            queue = list(model.get_action(observation, task_description))  # (n_action_steps, 7)
        act = queue.pop(0)
        if act.shape[-1] == 7 and action_dim == 12:  # pad mobile-base tail
            act = np.concatenate([act, MOBILE_BASE_TAIL])
        obs, _reward, _done, _info = env.step(act)
        if env._check_success():
            success = True
            length = step + 1
            break
    return success, length, frames


def _resolve_task(name: str) -> str:
    """Accept a bare task name (``TurnOffMicrowave``) or ``robocasa/<Task>``."""
    return name.split("/", 1)[1] if "/" in name else name


def _save_video(frames: List[np.ndarray], path: Path) -> None:
    try:
        import imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), [np.asarray(f, dtype=np.uint8) for f in frames], fps=20)
    except Exception as e:
        print(f"[kitchen] video save skipped ({path.name}): {e}", flush=True)


@dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 6699
    task_name: str = "TurnOffMicrowave"  # bare task or robocasa/<Task>
    pretrained_path: str = ""
    unnorm_key: Optional[str] = None
    n_episodes: int = 50
    n_action_steps: int = 16  # open-loop steps executed per policy query (== action_horizon)
    seed: int = 195

    # Parallel sharding: strided slice of [0, n_episodes) so shards are disjoint and their union
    # reproduces a full pass. Scene selection uses the GLOBAL episode index (episode_idx // 10).
    num_shards: int = 1
    shard_idx: int = 0
    results_out: str = ""

    # Output
    video_out_path: Optional[str] = None
    save_video: bool = False

    # Env / obs (verbatim cosmos eval defaults)
    env_img_res: int = 224
    robots: str = "PandaMobile"  # alias for PandaOmron
    obj_instance_split: str = "B"  # held-out test objects ("A" = training objects)
    layout_and_style_ids: str = "((1,1),(2,2),(4,4),(6,9),(7,10))"  # 5 test scenes; also "train" / "test" / ""
    # "cycle" = official protocol (episode_idx//10 walks the pool: 10 eps per scene, first
    # n_episodes/10 scenes only). "random" = seeded per-episode draw from the WHOLE pool — use this
    # with layout_and_style_ids="train" so 50 eps sample 50 of the 115 training scenes instead of 5.
    scene_sampling: str = "cycle"
    randomize_cameras: bool = False
    flip_images: bool = True
    use_jpeg: bool = True
    include_state: bool = False
    resize_size: tuple = (224, 224)


def main(args: Args) -> None:
    task = _resolve_task(args.task_name)
    all_tasks = {**SINGLE_STAGE_TASK_DATASETS, **MULTI_STAGE_TASK_DATASETS}
    if task not in all_tasks:
        raise ValueError(f"Unknown RoboCasa task '{task}'. Available: {sorted(all_tasks)}")
    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        raise ValueError(f"bad sharding: shard_idx={args.shard_idx} num_shards={args.num_shards}")

    max_steps = TASK_MAX_STEPS.get(task, 500)
    episode_indices = list(range(args.shard_idx, args.n_episodes, args.num_shards))
    scene_pool = resolve_scene_pool(args.layout_and_style_ids)
    print(
        f"[kitchen] task={task} shard {args.shard_idx}/{args.num_shards}: "
        f"{len(episode_indices)}/{args.n_episodes} eps, max_steps={max_steps}, "
        f"scenes={'any' if scene_pool is None else len(scene_pool)} ({args.scene_sampling}), "
        f"obj split {args.obj_instance_split}",
        flush=True,
    )

    model = PolicyWarper(
        policy_ckpt_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
        use_jpeg=args.use_jpeg,
        include_state=args.include_state,
    )

    successes: List[bool] = []
    scenes: List[tuple] = []
    t0 = time.time()
    for gi in episode_indices:
        if scene_pool is not None and args.scene_sampling == "random":
            # Seeded per-GLOBAL-index draw: shard-invariant, and covers the whole pool.
            lay, sty = scene_pool[np.random.default_rng(args.seed + gi).integers(len(scene_pool))]
            scene_str, scene_ep_idx = f"(({lay},{sty}),)", None
        else:
            # Official protocol, bit-identical to the original: episode_idx//10 walks the pool.
            scene_str, scene_ep_idx = (repr(tuple(scene_pool)) if scene_pool is not None else ""), gi
        env = create_robocasa_env(
            task,
            seed=args.seed * gi * 256,  # cosmos per-episode seed formula (gi=0 -> 0)
            episode_idx=scene_ep_idx,
            env_img_res=args.env_img_res,
            robots=args.robots,
            obj_instance_split=args.obj_instance_split,
            layout_and_style_ids_str=scene_str,
            randomize_cameras=args.randomize_cameras,
        )
        env.reset()
        ep_meta = env.get_ep_meta()
        task_description = ep_meta["lang"]
        scenes.append((int(ep_meta.get("layout_id", -1)), int(ep_meta.get("style_id", -1))))
        success, length, frames = run_episode(
            env,
            model,
            task_description,
            max_steps,
            args.flip_images,
            collect_frames=(args.save_video and args.video_out_path is not None),
        )
        try:
            env.close()
        except Exception:
            pass
        successes.append(success)
        n_ok = int(np.sum(successes))
        print(
            f"  gi={gi} scene={scenes[-1]} success={success} len={length} | cumulative={n_ok}/{len(successes)} "
            f"({100 * n_ok / len(successes):.0f}%)",
            flush=True,
        )
        if args.save_video and args.video_out_path:
            _save_video(frames, Path(args.video_out_path) / f"{task}_ep{gi}_success={success}.mp4")

    total = len(successes)
    n_ok = int(np.sum(successes))
    rate = n_ok / total if total else 0.0
    print(
        f"[kitchen] {task} shard {args.shard_idx}/{args.num_shards}: "
        f"rate {rate:.3f} ({n_ok}/{total}) in {time.time() - t0:.1f}s",
        flush=True,
    )

    if args.results_out:
        per_scene: Dict[str, List[int]] = {}  # "(layout,style)" -> [successes, episodes]
        for scene, ok in zip(scenes, successes, strict=True):
            entry = per_scene.setdefault(str(scene), [0, 0])
            entry[0] += int(ok)
            entry[1] += 1
        out = Path(args.results_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "task": task,
                    "env_name": args.task_name,
                    "shard_idx": args.shard_idx,
                    "num_shards": args.num_shards,
                    "obj_instance_split": args.obj_instance_split,
                    "layout_and_style_ids": args.layout_and_style_ids,
                    "scene_sampling": args.scene_sampling,
                    "successes": n_ok,
                    "episodes": total,
                    "rate": rate,
                    "episode_indices": episode_indices,
                    "success_flags": [bool(s) for s in successes],
                    "scene_ids": [list(s) for s in scenes],
                    "per_scene": per_scene,
                },
                indent=2,
            )
        )
        print(f"[kitchen] wrote shard results to {out}", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
