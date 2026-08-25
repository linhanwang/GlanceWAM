import dataclasses
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Co-locate MuJoCo's EGL renderer on the server's physical GPU. start_libero_env.sh passes
# the *EGL device index* (an index into eglQueryDevicesEXT, which is NOT the nvidia-smi /
# CUDA order on some hosts) as GLANCEWAM_EGL_DEVICE_ID. We promote it to MUJOCO_EGL_DEVICE_ID
# here — crucially AFTER the `from libero...` import above, whose robosuite binding_utils
# asserts MUJOCO_EGL_DEVICE_ID is a substring of CUDA_VISIBLE_DEVICES (which the permuted
# EGL index would fail). robosuite's egl_context reads MUJOCO_EGL_DEVICE_ID lazily at first
# render, so setting it now (before any OffScreenRenderEnv) still takes effect.
_egl_id = os.environ.get("GLANCEWAM_EGL_DEVICE_ID")
if _egl_id and not os.environ.get("MUJOCO_EGL_DEVICE_ID"):
    os.environ["MUJOCO_EGL_DEVICE_ID"] = _egl_id
    logging.info("Set MUJOCO_EGL_DEVICE_ID=%s (EGL index for the target GPU) from GLANCEWAM_EGL_DEVICE_ID", _egl_id)

from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Parallel sharding: partition the flattened (task_id, episode_idx) work list across `num_shards` client
    # processes; this client runs only the strided slice `[shard_idx::num_shards]`. All shards share one batched
    # policy server (see start_libero_env.sh). num_shards=1 reproduces the original full single-client sweep.
    #################################################################################################################
    num_shards: int = 1  # total number of parallel client shards hitting the same server
    shard_idx: int = 0  # this client's shard index, in [0, num_shards)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    results_out: str = ""  # optional path to dump a JSON results summary (for shard-aware aggregation)

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path,  # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    # Validate sharding args.
    if not (1 <= args.num_shards):
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_idx < args.num_shards):
        raise ValueError(f"shard_idx must be in [0, {args.num_shards}), got {args.shard_idx}")

    # Build this shard's assigned work by striding the flattened (task_id, episode_idx) list. The global
    # index gi = task_id * num_trials_per_task + episode_idx is partitioned by gi % num_shards == shard_idx,
    # so each shard touches a mix of every task (balanced load) and the partitions are disjoint. The episode
    # init state is keyed by episode_idx and `seed` (both constant across shards), so a strided run is
    # bit-identical per-episode to the original serial sweep.
    assigned: dict[int, list[int]] = {}
    gi = 0
    for task_id in range(num_tasks_in_suite):
        for episode_idx in range(args.num_trials_per_task):
            if gi % args.num_shards == args.shard_idx:
                assigned.setdefault(task_id, []).append(episode_idx)
            gi += 1

    logging.info(
        f"[shard {args.shard_idx}/{args.num_shards}] assigned {sum(len(v) for v in assigned.values())} "
        f"episodes across {len(assigned)} tasks"
    )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    per_task_results: dict[int, dict] = {}
    for task_id in tqdm.tqdm(sorted(assigned)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(assigned[task_id]):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API --> two images provided here --> check training
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                }

                start_time = time.time()

                response = client_model.step(example=example_dict, step=step)

                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Record per-task results and free the env before moving to the next task.
        per_task_results[task_id] = {
            "task": str(task_description),
            "successes": task_successes,
            "episodes": task_episodes,
        }
        try:
            env.close()
        except Exception:
            pass

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    rate = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    logging.info(f"Total success rate: {rate}")
    logging.info(f"Total episodes: {total_episodes}")
    # Machine-parseable line for shard-aware aggregation (works even without --results-out).
    logging.info(
        f">> SHARD_RESULT shard={args.shard_idx}/{args.num_shards} "
        f"successes={total_successes} episodes={total_episodes} rate={rate:.4f}"
    )

    if args.results_out:
        summary = {
            "task_suite": args.task_suite_name,
            "shard_idx": args.shard_idx,
            "num_shards": args.num_shards,
            "successes": total_successes,
            "episodes": total_episodes,
            "rate": rate,
            "per_task": {str(k): v for k, v in sorted(per_task_results.items())},
        }
        out_path = pathlib.Path(args.results_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        logging.info(f"Wrote shard results to {out_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
