#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert a GlanceWAM research checkpoint (``steps_*_pytorch_model[_ema].pt`` + run ``config.yaml`` +
``dataset_statistics.json``) into a LeRobot ``pretrained_model`` directory (config, safetensors weights,
pre/post-processors with normalization stats and the prefilled UMT5 text table).

Example:
    python -m lerobot_policy_glancewam.convert_checkpoint \\
        --checkpoint <run_dir>/checkpoints/steps_15000_pytorch_model_ema.pt \\
        --run-dir <run_dir> \\
        --dataset-dir <datasets>/libero_10_no_noops_1.0.0_lerobot [--dataset-dir ...] \\
        --output outputs/glancewam_libero_lerobot
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .configuration_glancewam import GlanceWAMConfig
from .modeling_glancewam import GlanceWAMPolicy
from .processor_glancewam import UMT5TextTableProcessorStep, make_glancewam_pre_post_processors

# Frozen, base-reconstructable modules the research checkpoints may carry; never part of the policy.
_DROP_PREFIXES = ("backbone.text_encoder.", "backbone.vae.", "_vae_encoder_eager.", "_vae_encoder_compiled.")


def _read_prompts(dataset_dirs: list[Path]) -> list[str]:
    import pandas as pd

    prompts: list[str] = []
    for d in dataset_dirs:
        df = pd.read_parquet(d / "meta" / "tasks.parquet")
        tasks = df["task"].tolist() if "task" in df.columns else df.index.tolist()
        prompts.extend(str(t) for t in tasks)
    return list(dict.fromkeys(prompts))


def _stats_from_glancewam(stats: dict, use_state: bool) -> dict[str, dict[str, torch.Tensor]]:
    """GR00T min/max stats -> LeRobot stats. Masked action dims get (min, max) = (-1, 1), i.e. identity."""
    out: dict[str, dict[str, torch.Tensor]] = {}
    a = stats["action"]
    amin = torch.tensor(a["min"], dtype=torch.float32)
    amax = torch.tensor(a["max"], dtype=torch.float32)
    mask = torch.tensor(a.get("mask", [True] * len(a["min"])), dtype=torch.bool)
    amin = torch.where(mask, amin, torch.full_like(amin, -1.0))
    amax = torch.where(mask, amax, torch.full_like(amax, 1.0))
    out[ACTION] = {"min": amin, "max": amax}
    if use_state and "state" in stats:
        out[OBS_STATE] = {
            "min": torch.tensor(stats["state"]["min"], dtype=torch.float32),
            "max": torch.tensor(stats["state"]["max"], dtype=torch.float32),
        }
    return out


def build_config(
    run_cfg: dict, *, camera_keys: list[str], image_hw: tuple[int, int], use_state: bool, state_dim: int
) -> GlanceWAMConfig:
    fw = run_cfg["framework"]
    wm, vc, gc, am = (
        fw["world_model"],
        fw.get("video_cotrain", {}),
        fw.get("goal_conditioning", {}),
        fw["action_model"],
    )
    ds = run_cfg["datasets"]["vla_data"]
    action_dim = int(am["action_dim"])
    input_features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_hw)) for k in camera_keys}
    if use_state:
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    goal_range = ds.get("goal_frame_range", [1, int(ds["future_frame_idx"])])
    return GlanceWAMConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        chunk_size=int(am["action_horizon"]),
        n_action_steps=int(am["action_horizon"]),
        base_model_path=wm["base_wm"],
        vae_input_size=tuple(int(x) for x in wm["vae_input_size"]),
        camera_order=camera_keys,
        camera_layout=str(wm.get("camera_concat", "side_by_side")),
        extract_layers=[int(t) for t in wm.get("extract_layers", [-1])],
        fps=int(wm.get("fps", 24)),
        text_max_length=int(wm.get("text_max_length", 512)),
        num_history_frames=int(wm.get("num_history_frames", 1)),
        future_frame_idx=int(ds["future_frame_idx"]),
        goal_frame_range=(int(goal_range[0]), int(goal_range[1])),
        video_loss_weight=float(vc.get("lambda", 1.0)),
        sigma_sampling=str(vc.get("sigma_sampling", "uniform")),
        clean_prefix_isolation=bool(vc.get("clean_prefix_isolation", True)),
        goal_dropout_p=float(gc.get("dropout_p", 0.1)),
        goal_gen_steps=int(gc.get("goal_gen_steps", 10)),
        goal_h_embed_dim=int(gc.get("h_embed_dim", 16)),
        action_model_type=str(am.get("action_model_type", "DiT-B")),
        action_hidden_size=int(am.get("hidden_size", 1024)),
        num_action_head_layers=int(am["diffusion_model_cfg"]["num_layers"]),
        action_head_dropout=float(am["diffusion_model_cfg"].get("dropout", 0.2)),
        num_target_vision_tokens=int(am.get("num_target_vision_tokens", 32)),
        max_seq_len=int(am.get("max_seq_len", 1024)),
        add_pos_embed=bool(am.get("add_pos_embed", True)),
        noise_beta_alpha=float(am.get("noise_beta_alpha", 1.5)),
        noise_beta_beta=float(am.get("noise_beta_beta", 1.0)),
        noise_s=float(am.get("noise_s", 0.999)),
        num_timestep_buckets=int(am.get("num_timestep_buckets", 1000)),
        num_inference_timesteps=int(am.get("num_inference_timesteps", 4)),
        repeated_diffusion_steps=int(am.get("repeated_diffusion_steps", 8)),
        t_loss_weighting=bool(am.get("t_loss_weighting", False)),
        use_state=use_state,
        action_head_state_dim=int(am.get("state_dim", 0) or 0),
        compile_vae=bool(wm.get("compile_vae", True)),
        sdpa_backend=str(run_cfg.get("trainer", {}).get("sdpa_backend", "cudnn") or "auto"),
        # GR00T checkpoints ship min/max stats only and were trained MIN_MAX; the config default is QUANTILES.
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
    )


def convert(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir)
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    stats_all = json.loads((run_dir / "dataset_statistics.json").read_text())
    if len(stats_all) != 1 and args.embodiment is None:
        raise ValueError(
            f"dataset_statistics.json has {len(stats_all)} embodiments, pass --embodiment: {list(stats_all)}"
        )
    stats = stats_all[args.embodiment or next(iter(stats_all))]

    camera_keys = [f"{OBS_IMAGES}.{c}" for c in args.cameras]
    state_dim = len(stats["state"]["min"]) if "state" in stats else 0
    config = build_config(
        run_cfg,
        camera_keys=camera_keys,
        image_hw=(args.image_size, args.image_size),
        use_state=args.use_state,
        state_dim=state_dim,
    )
    config.device = args.device

    logging.info("Building the policy (random init; the checkpoint carries every trained weight).")
    policy = GlanceWAMPolicy(config, load_pretrained_backbone=False)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(_DROP_PREFIXES)}
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    logging.info(f"Loaded {len(state_dict)} tensors.")

    prompts = _read_prompts([Path(d) for d in args.dataset_dir])
    logging.info(f"{len(prompts)} task prompts to prefill.")
    dataset_stats = _stats_from_glancewam(stats, args.use_state)
    preprocessor, postprocessor = make_glancewam_pre_post_processors(config, dataset_stats, prompts=prompts)
    text_step = next(s for s in preprocessor.steps if isinstance(s, UMT5TextTableProcessorStep))
    text_step.encode(prompts, torch.device(args.device))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    preprocessor.save_pretrained(out)
    postprocessor.save_pretrained(out)
    logging.info(f"Saved LeRobot checkpoint to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="steps_*_pytorch_model[_ema].pt")
    parser.add_argument(
        "--run-dir", required=True, help="directory holding config.yaml and dataset_statistics.json"
    )
    parser.add_argument(
        "--dataset-dir", action="append", default=[], help="LeRobot v3 dataset dir(s) for the task prompts"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cameras", nargs="+", default=["image", "wrist_image"], help="camera names in stitching order"
    )
    parser.add_argument(
        "--image-size", type=int, default=256, help="camera frame size declared in the input features"
    )
    parser.add_argument("--use-state", action="store_true")
    parser.add_argument("--embodiment", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    convert(args)


if __name__ == "__main__":
    main()
