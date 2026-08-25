# Copyright 2025 glancewam community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket

# --- Thread cap. MUST be set before `import torch`, which sizes its intra-op pool (and the
# OpenMP runtime behind it) from the host core count. That default is catastrophic when
# several policy servers share a many-core box: measured on a 480-core host, five
# unconstrained servers came up with ~643 threads EACH, putting ~2400 busy-waiting threads on
# 480 cores (load average 15000+). The sim clients — which actually drive the eval loop — were
# starved to ~17% CPU and GPU utilisation sat at ~0%: the sweep was CPU-thrashing, not
# computing. Capping it took a stalled RoboCasa-kitchen run to 114 s/episode.
# The server does almost no CPU math (the model runs on GPU; per-request CPU work is a msgpack
# decode plus one image stitch/resize), so a small pool costs nothing. setdefault, so an
# explicit environment setting still wins; override with GLANCEWAM_SERVER_THREADS.
_SERVER_THREADS = os.environ.get("GLANCEWAM_SERVER_THREADS", "8")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _SERVER_THREADS)

import torch

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from glancewam.model.framework.base_framework import baseframework

# Backstop for any path where torch was already imported before this module (the env vars
# above only bind at OpenMP init); torch.set_num_threads re-sizes the pool at runtime.
torch.set_num_threads(int(_SERVER_THREADS))


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained(  # TODO should auto detect framework from model path
        args.ckpt_path,
    )

    if args.stop_after_step is not None:
        try:
            vla.config.framework.video_noise.stop_after_step = int(args.stop_after_step)
        except Exception:
            vla.config.framework.video_noise["stop_after_step"] = int(args.stop_after_step)
        logging.info("Override framework.video_noise.stop_after_step = %s", args.stop_after_step)

    if args.num_sampling_step is not None:
        try:
            vla.config.framework.video_noise.num_sampling_step = int(args.num_sampling_step)
        except Exception:
            vla.config.framework.video_noise["num_sampling_step"] = int(args.num_sampling_step)
        logging.info("Override framework.video_noise.num_sampling_step = %s", args.num_sampling_step)

    if args.num_inference_timesteps is not None:
        n = int(args.num_inference_timesteps)
        action_model = getattr(vla, "action_model", None)
        if action_model is not None and hasattr(action_model, "num_inference_timesteps"):
            action_model.num_inference_timesteps = n
            try:
                vla.config.framework.action_model.num_inference_timesteps = n
            except Exception:
                pass
            logging.info("Override action_model.num_inference_timesteps = %s (flow-matching Euler steps)", n)
        else:
            logging.warning(
                "--num-inference-timesteps=%s requested but vla.action_model.num_inference_timesteps not found; ignored",
                n,
            )

    if args.use_bf16:  # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "127.0.0.1"
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # Goal-image cotrain (GlanceWAM): advertise H_g (goal horizon, native
    # rows) so the client auto-enables the stateful goal cadence (hold a generated goal for
    # H_g, refresh, feed the shrinking goal-time diff). Absent for every other framework.
    metadata = {"env": "simpler_env"}
    goal_horizon = getattr(vla, "_default_goal_horizon", None)
    if goal_horizon is not None and getattr(vla, "goal_enabled", False):
        metadata["goal_horizon_rows"] = float(goal_horizon)
        logging.info("Advertising goal_horizon_rows=%s (goal-image cotrain cadence).", goal_horizon)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        max_batch=args.max_batch,
        batch_wait_ms=args.batch_wait_ms,
        metadata=metadata,
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--max-batch",
        type=int,
        default=1,
        help="Max number of concurrent infer requests coalesced into one forward pass. 1 disables batching.",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=0.0,
        help="Max ms to wait for additional requests after the first one arrives, before starting the forward pass.",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=None,
        help="VAM only: tap the K-th DiT block hidden state after K Karras-7 AB2 iterations "
        "(mimic-video's eval/bridge recipe). Unset → framework default (1, mimic-video's best macro).",
    )
    parser.add_argument(
        "--num-sampling-step",
        type=int,
        default=None,
        help="VAM only: total number of Karras-7 AB2 schedule iterations. "
        "Unset → framework default (35, matches mimic-video).",
    )
    parser.add_argument(
        "--num-inference-timesteps",
        type=int,
        default=None,
        help="Flow-matching action heads (GR00T / PI): number of Euler integration steps in "
        "predict_action(). Unset → checkpoint config default (typically 4). Lower = faster, less accurate.",
    )
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
