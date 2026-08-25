# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].

import inspect
import json
import warnings
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset

from glancewam.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset, ModalityConfig
from glancewam.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    EmbodimentTag,
)
from glancewam.dataloader.gr00t_lerobot.transform.video import (
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from glancewam.dataloader.t5_cache import T5CacheReader


def collate_fn(batch):
    return batch


def _ensure_video_resize(transforms, video_keys, image_size):
    # Several per-robot data_configs ship with VideoResize commented out, which
    # under lerobot v3 silently feeds native-res frames into the VLM. If
    # `data_cfg.image_size` is set and the chain has no VideoResize yet,
    # prepend ToTensor → Resize → ToNumpy so every robot yields frames at the
    # requested resolution.
    if image_size is None or not video_keys:
        return
    if any(isinstance(t, VideoResize) for t in transforms.transforms):
        return
    h, w = int(image_size[0]), int(image_size[1])
    transforms.transforms = [
        VideoToTensor(apply_to=list(video_keys)),
        VideoResize(apply_to=list(video_keys), height=h, width=w, interpolation="linear"),
        VideoToNumpy(apply_to=list(video_keys)),
    ] + list(transforms.transforms)


def _reject_legacy_future_window_keys(data_name: str, data_cfg) -> None:
    """Fail loud on the pre-2026-07-16 video-window keys.

    All four are gone. The window is now ONE frame at ``future_frame_idx`` NATIVE rows, and the
    history half is strided by ``hist_fps`` alone:

    - ``num_future_frames`` / ``future_frame_mode`` — the window was denominated in STRIDED frames
      while ``action_horizon`` was in native rows, so the two silently drifted (kitchen trained a
      t+48 window against a 16-row chunk for weeks). ``future_frame_mode='all'`` (the dense
      window) no longer exists at all; only the single end-of-window frame is trained.
    - ``video_fps`` / ``train_fps`` — once the future frame is addressed directly, this only ever
      strided the HISTORY half, duplicating ``hist_fps`` and silently overriding it. Since
      ``hist_fps`` is what the VAE-cache precompute keys on, the two disagreeing meant the cache
      and the live-decode path could build different windows.

    Ignoring any of them would be silent and wrong: a stale ``num_future_frames: 12`` beside a
    defaulted ``future_frame_idx`` trains a t+0 window instead of t+48, and a stale ``video_fps``
    beside ``num_history_frames > 1`` drops the history stride to 1 — neither says a word.
    """
    if data_cfg is None:
        return
    removed = {
        "num_future_frames": (
            "future_frame_idx = num_future_frames * round(native_fps / video_fps), "
            "e.g. nff=4 @ video_fps=5 on a 20 Hz dataset -> future_frame_idx: 16"
        ),
        "future_frame_mode": "drop it — only the single end-of-window frame is trained now",
        "video_fps": "drop it, or set hist_fps to the same value if num_history_frames > 1",
        "train_fps": "drop it, or set hist_fps to the same value if num_history_frames > 1",
    }
    hits = [k for k in removed if data_cfg.get(k, None) is not None]
    if hits:
        detail = "".join(f"    - {k}: {removed[k]}\n" for k in hits)
        raise ValueError(
            f"[{data_name}] datasets.vla_data sets removed key(s) {', '.join(hits)}.\n"
            f"  The video window is now ONE frame at `future_frame_idx`, in the same NATIVE rows "
            f"as action_horizon; `hist_fps` alone strides the history half.\n{detail}"
        )


def _parse_future_frame_range(data_name: str, data_cfg, eval_split: str) -> tuple[int, int] | None:
    """Validate `datasets.vla_data.future_frame_range` ([min, max] inclusive, native rows).

    When set, each training sample draws H ~ U{min..max} and both the predicted future
    frame and the shipped future-state row move to t+H (§8 state-conditioning,
    below). `future_frame_idx` must still be > 0 — it
    keeps defining the video window's future SLOT (and the eval-time default horizon);
    the range only overrides its value per sample. Forced off on the eval split, which
    already drops the future frame entirely.
    """
    rng = data_cfg.get("future_frame_range", None) if data_cfg is not None else None
    if rng is None or eval_split == "eval":
        return None
    rng = list(rng)
    if len(rng) != 2 or int(rng[0]) < 1 or int(rng[0]) > int(rng[1]):
        raise ValueError(
            f"[{data_name}] datasets.vla_data.future_frame_range must be [min, max] with "
            f"1 <= min <= max (native rows), got {rng}"
        )
    if int(data_cfg.get("future_frame_idx", 0)) <= 0:
        raise ValueError(
            f"[{data_name}] future_frame_range needs future_frame_idx > 0: the fixed index "
            f"defines the video window's future slot (and the eval default); the range only "
            f"re-draws its value per sample."
        )
    return int(rng[0]), int(rng[1])


def _parse_goal_frame_range(data_name: str, data_cfg, eval_split: str) -> tuple[int, int] | None:
    """Validate `datasets.vla_data.goal_frame_range` ([min, max] inclusive, native rows).

    The goal-image cotrain (GlanceWAM; below,
    G0) ships a SECOND future frame — a hindsight "goal" the action head conditions on,
    sampled per training sample at an offset g ~ U{min..max} native rows ahead (the
    (0, H_g] cadence-rule distribution, so min >= 1). It is APPENDED after the fixed
    future slot (the video target @future_frame_idx), so `future_frame_idx` must be > 0.
    Forced off on the eval split (predict_action generates the goal). Incompatible with
    the VAE cache (a per-sample random goal frame cannot be precomputed)."""
    rng = data_cfg.get("goal_frame_range", None) if data_cfg is not None else None
    if rng is None or eval_split == "eval":
        return None
    rng = list(rng)
    if len(rng) != 2 or int(rng[0]) < 1 or int(rng[0]) > int(rng[1]):
        raise ValueError(
            f"[{data_name}] datasets.vla_data.goal_frame_range must be [min, max] with "
            f"1 <= min <= max (native rows), got {rng}"
        )
    if int(data_cfg.get("future_frame_idx", 0)) <= 0:
        raise ValueError(
            f"[{data_name}] goal_frame_range needs future_frame_idx > 0: the goal frame is "
            f"appended AFTER the fixed video-target future slot."
        )
    return int(rng[0]), int(rng[1])


def _parse_multi_horizon(data_name: str, data_cfg, eval_split: str) -> tuple[list[int], list[tuple[int, int]]] | None:
    """Validate the multi-frame video-window keys:
    `datasets.vla_data.future_frame_idxs` (ascending video-target offsets, native rows)
    and the OPTIONAL `datasets.vla_data.goal_frame_ranges` (E8.0 two-horizon design,
    one contiguous hindsight band per target,
    band i = (prev_max, idx_i]). The window becomes [obs, f@idx_0.., goal@band_0..].

    Two consumers:
      * E8.0 goal cotrain — both keys, matched entry-for-entry.
      * a dense-future variant — `future_frame_idxs` alone, giving the dense
        9-frame video window its Wan2.2 video expert denoises, with no goal frames.

    Forced off on the eval split (predict_action generates the goals / needs no future
    window). Incompatible with the single-goal keys, the §8 random future window, and
    the VAE cache."""
    if data_cfg is None:
        return None
    idxs = data_cfg.get("future_frame_idxs", None)
    ranges = data_cfg.get("goal_frame_ranges", None)
    if idxs is None and ranges is None:
        return None
    if eval_split == "eval":
        return None
    if idxs is None:
        raise ValueError(f"[{data_name}] goal_frame_ranges needs future_frame_idxs (got ranges={ranges}).")
    idxs = [int(i) for i in idxs]
    # goal_frame_ranges omitted = a dense video window with no hindsight goal frames.
    ranges = [] if ranges is None else [(int(r[0]), int(r[1])) for r in ranges]
    if len(idxs) < 2 or (ranges and len(idxs) != len(ranges)):
        raise ValueError(
            f"[{data_name}] multi-horizon needs >= 2 video targets, matched to goal_frame_ranges "
            f"when those are set: future_frame_idxs={idxs}, goal_frame_ranges={ranges or None}. "
            f"For a single horizon use future_frame_idx + goal_frame_range."
        )
    if idxs != sorted(set(idxs)) or idxs[0] < 1:
        raise ValueError(f"[{data_name}] future_frame_idxs must be ascending unique positive rows, got {idxs}")
    if idxs[-1] != int(data_cfg.get("future_frame_idx", 0)):
        raise ValueError(
            f"[{data_name}] future_frame_idx ({data_cfg.get('future_frame_idx', 0)}) must equal "
            f"future_frame_idxs[-1] ({idxs[-1]}) — downstream consumers (default goal horizon, "
            f"span checks) read the scalar key."
        )
    prev_hi = 0
    for i, ((lo, hi), idx) in enumerate(zip(ranges, idxs[: len(ranges)], strict=True)):
        if hi != idx or lo != prev_hi + 1:
            raise ValueError(
                f"[{data_name}] goal_frame_ranges[{i}]={list(ranges[i])} must be the contiguous band "
                f"({prev_hi}, {idx}] = [{prev_hi + 1}, {idx}] matching future_frame_idxs[{i}]."
            )
        prev_hi = hi
    if data_cfg.get("goal_frame_range", None) is not None or data_cfg.get("future_frame_range", None) is not None:
        raise ValueError(
            f"[{data_name}] future_frame_idxs/goal_frame_ranges are incompatible with the "
            f"single-goal goal_frame_range and the §8 future_frame_range."
        )
    return idxs, ranges


def _warn_video_action_span_mismatch(
    data_name: str,
    future_frame_idx: int,
    action_horizon: int | None,
    data_cfg,
    future_frame_range: tuple[int, int] | None = None,
) -> None:
    """Warn when an ACTION-CONDITIONED run predicts a frame the action chunk doesn't reach.

    Both quantities are native rows, so the check is a direct comparison — the unit mismatch that
    made the old assert necessary is gone with the ``(num_future_frames x stride)`` arithmetic.

    Deliberately a warning, and only for ``action_conditioning``: a mismatch is a real defect
    ONLY there, because the action token then describes a fraction of the interval the DiT is
    asked to predict. With conditioning off, pairing a long pixel-prediction window with a short
    chunk is a legitimate (and measured-good) recipe — the 0.680 kitchen champion trained a t+48
    window against a 16-row chunk, so hard-failing it would forbid our own best result.
    """
    if not action_horizon or not future_frame_idx or data_cfg is None:
        return
    if not bool(data_cfg.get("action_conditioning", False)):
        return
    if future_frame_range is not None:
        lo, hi = future_frame_range
        if hi != int(action_horizon) or lo != int(action_horizon):
            warnings.warn(
                f"[{data_name}] action_conditioning is ON with a per-sample random video window "
                f"(future_frame_range=[{lo}, {hi}] native rows) while the action chunk is fixed at "
                f"{int(action_horizon)} rows: samples with H > {int(action_horizon)} leave the tail "
                f"of the predicted interval action-marginalized. Expected when the window is driven "
                f"by state@t+H conditioning (§8) — the state token covers any H.",
                UserWarning,
                stacklevel=2,
            )
        return
    if future_frame_idx != int(action_horizon):
        warnings.warn(
            f"[{data_name}] action_conditioning is ON and the video window does not match the "
            f"action chunk: the DiT predicts the frame at t+{future_frame_idx} native rows, but "
            f"the chunk a_t..a_t+{int(action_horizon) - 1} only carries the scene to "
            f"t+{int(action_horizon)} ({future_frame_idx / int(action_horizon):.1f}x apart). The "
            f"action token therefore describes part of the predicted interval, leaving the rest "
            f"action-marginalized. Set future_frame_idx == action_horizon ({action_horizon}) "
            f"unless the mismatch is intentional.",
            UserWarning,
            stacklevel=2,
        )


def read_native_fps(dataset_dir: Path | str) -> float:
    """Read the dataset's recording fps from ``meta/info.json`` (top-level
    ``fps``). LeRobot v3 stores one global fps per dataset (e.g. 20 for LIBERO,
    20 for RoboCasa kitchen)."""
    info_path = Path(dataset_dir) / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    fps = info.get("fps")
    if fps is None:
        raise ValueError(f"no top-level 'fps' in {info_path}")
    return float(fps)


def resolve_frame_stride(dataset_dir: Path | str, hist_fps: float | None, num_frames: int) -> int:
    """Stride (in native frames) between consecutive history frames so the
    window samples at ~``hist_fps`` Hz.

    ``stride = round(native_fps / hist_fps)``, clamped to ``>= 1``. Returns 1
    (contiguous window at the dataset's native rate) when ``hist_fps`` is
    unset/non-positive or when the history is a single frame.
    """
    if hist_fps is None or hist_fps <= 0 or num_frames <= 1:
        return 1
    return max(1, int(round(read_native_fps(dataset_dir) / float(hist_fps))))


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    eval_split: str = "train",
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    # Single source of truth for chunk length: if data_cfg provides
    # `action_horizon`, override the data_config's action_indices so that
    # ModalityConfig.delta_indices (and thus the per-sample action tensor
    # length) matches the model's expected chunk. Set this from YAML via
    # `action_horizon: ${framework.action_model.action_horizon}`.
    action_horizon = data_cfg.get("action_horizon", None) if data_cfg is not None else None
    if action_horizon is not None:
        data_config.action_indices = list(range(int(action_horizon)))
    modality_config = data_config.modality_config()
    # Multi-frame history: widen ONLY the video modality's delta_indices to
    # [-(N-1), ..., 0] so torchcodec returns N consecutive past frames per
    # sample (out-of-range indices at episode start get clamped to frame 0
    # by get_video — equivalent to first-frame repetition, which matches
    # tools/precompute_vae_for_lerobot.py's pad scheme). state / language /
    # action keep their original [0]-length delta_indices; widening them
    # would change per-sample shapes everywhere downstream.
    # Future-window pixels are needed by Video-Action Models (mimic-video
    # Algorithm 2): a fixed number of frames AFTER the current step that the
    # frozen video DiT denoises over.  We widen the same `video` modality's
    # delta_indices to cover both [-(N_past-1), …, 0] AND [1, …, N_fut].  The
    # returned image list is therefore length N_past + N_fut; the framework
    # reads `num_history_frames` to know where the split is.  Frames past the
    # end of an episode get clamped to the last valid index by torchcodec's
    # get_video, equivalent to last-frame repetition — matches mimic-video's
    # padding scheme at episode tails.
    num_history_frames = int(data_cfg.get("num_history_frames", 1)) if data_cfg is not None else 1
    # Video co-training predicts ONE future frame, at t+future_frame_idx NATIVE rows from the
    # obs frame (the cosmos-policy conditioning trick: the framework tiles it across one
    # temporal VAE group so it lands on exactly one noised latent). 0/unset = no future frame.
    #
    # This is denominated in NATIVE rows, exactly like action_horizon, so the two are directly
    # comparable and independently settable. It replaces the pre-2026-07-16
    # (num_future_frames x round(native_fps/video_fps)) pair, which denominated the video window
    # in STRIDED frames and the action chunk in NATIVE rows: every live recipe had silently
    # drifted (kitchen paired a t+48 video window with a 16-row chunk, LIBERO t+48 with 8).
    future_frame_idx = int(data_cfg.get("future_frame_idx", 0)) if data_cfg is not None else 0
    if future_frame_idx < 0:
        raise ValueError(f"[{data_name}] datasets.vla_data.future_frame_idx must be >= 0, got {future_frame_idx}")
    _reject_legacy_future_window_keys(data_name, data_cfg)
    future_frame_range = _parse_future_frame_range(data_name, data_cfg, eval_split)
    goal_frame_range = _parse_goal_frame_range(data_name, data_cfg, eval_split)
    multi_horizon = _parse_multi_horizon(data_name, data_cfg, eval_split)
    # The held-out action-MSE eval scores obs-frame action prediction only — predict_action
    # never consumes the video future window. Drop the future frame on the eval split so each
    # window decodes just the obs frame(s) (and tiles by the action chunk), making the
    # co-training eval path identical to (and as fast as) the action-only recipe instead of
    # live-decoding the full obs+future window per window.
    if eval_split == "eval":
        future_frame_idx = 0
    # History sampling rate (Hz): when set, the history half is sampled every
    # `stride` native frames (stride = round(native_fps / hist_fps)) so the
    # window spans a fixed wall-clock duration regardless of the dataset's fps.
    hist_fps = data_cfg.get("hist_fps", None) if data_cfg is not None else None
    hist_stride = resolve_frame_stride(Path(data_root_dir) / data_name, hist_fps, num_history_frames)
    if (num_history_frames > 1 or future_frame_idx > 0) and "video" in modality_config:
        video_cfg = modality_config["video"]
        # History deltas [-(N-1)*s, …, -s, 0] (stride s), then the single future frame at
        # +future_frame_idx native rows. Both halves are now in native rows end-to-end.
        hist_deltas = list(range(-(num_history_frames - 1) * hist_stride, 1, hist_stride))
        # Goal-image cotrain: append the hindsight goal frame(s) the action head conditions
        # on, drawn per-sample at t+g (g ~ U{band}) and overridden in get_video. The stored
        # values are placeholders (band maxes); they land LAST in delta order so they pack
        # as the trailing frames of `sample["image"]` ([obs..., futures..., goals...]),
        # which GlanceWAM reads as im[-n_goal:]. E8.0 multi-horizon ships
        # one video target + one goal band per horizon.
        if multi_horizon is not None:
            future_deltas = list(multi_horizon[0])
            goal_deltas = [r[1] for r in multi_horizon[1]]
        else:
            future_deltas = [future_frame_idx] if future_frame_idx > 0 else []
            goal_deltas = [goal_frame_range[1]] if goal_frame_range is not None else []
        modality_config["video"] = ModalityConfig(
            delta_indices=hist_deltas + future_deltas + goal_deltas,
            modality_keys=video_cfg.modality_keys,
        )
        _warn_video_action_span_mismatch(data_name, future_frame_idx, action_horizon, data_cfg, future_frame_range)
    image_aug = bool(data_cfg.get("image_aug", False)) if data_cfg is not None else False
    image_size = data_cfg.get("image_size", None) if data_cfg is not None else None
    transform_params = inspect.signature(data_config.transform).parameters
    kwargs = {}
    if image_aug and "image_aug" in transform_params:
        kwargs["image_aug"] = True
        if image_size is not None and "image_size" in transform_params:
            kwargs["image_size"] = tuple(image_size)
    transforms = data_config.transform(**kwargs)
    _ensure_video_resize(transforms, getattr(data_config, "video_keys", None), image_size)
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(
            f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default"
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,  # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
        eval_split=eval_split,
    )

    # Attach a T5 cache reader so _pack_sample can return cached `lang_embed`
    # alongside the raw `lang` string. The framework decides whether to use it.
    # Per-dataset, opens lazily; missing prompts at runtime are a hard error
    # (not a silent fallback) — see T5CacheReader.lookup.
    # `attach_embeds: false` keeps the cache "enabled" for config purposes but
    # skips attaching the reader, so workers stop decompressing + shipping the
    # (max_length, 4096) fp16 rows per sample. Only valid when the framework
    # serves text embeds itself (e.g. the frameworks' eager
    # resident_text_table); the framework hard-errors on uncovered prompts.
    t5_cache_cfg = data_cfg.get("t5_cache", None) if data_cfg is not None else None
    if t5_cache_cfg is not None and t5_cache_cfg.get("enabled", False) and t5_cache_cfg.get("attach_embeds", True):
        # The on-disk cache dir is keyed by (model_id, max_length), so the reader
        # needs both. Defaults match tools/precompute_umt5_for_lerobot.py.
        dataset._t5_cache = T5CacheReader(
            dataset_path,
            t5_model_id=t5_cache_cfg.get("model_id", "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"),
            max_length=int(t5_cache_cfg.get("max_length", 512)),
        )

    return dataset


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    eval_split: str = "train",
    **kwargs: dict,
) -> "LeRobotMixtureDataset | ConcatDataset":
    """
    Get a LeRobotMixtureDataset object.

    `eval_split` selects the held-out action-MSE monitor's split:
    - "train" (default): excludes the held-out episodes when
      `data_cfg.eval_holdout_episodes > 0`; returns the weighted
      `LeRobotMixtureDataset` as usual.
    - "eval": returns a `ConcatDataset` over per-task tiled held-out windows
      (ordered, exhaustive — bypasses the mixture sampler so every window is
      scored exactly once).
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_LeRobotSingleDataset(
                    Path(data_root_dir),
                    d_name,
                    robot_type,
                    delete_pause_frame=delete_pause_frame,
                    data_cfg=data_cfg,
                    eval_split=eval_split,
                ),
                d_weight,
            )
        )

    # Rung-3 AWR guard: with advantage_label_id set, at least one dataset in the mix
    # must have actually loaded its sidecar. Otherwise (typo'd label id, wrong data
    # root) every sample would carry A=0 -> all weights 1 and the "AWR" run silently
    # degenerates into the filtered-BC baseline with no error anywhere.
    if data_cfg.get("advantage_label_id", "") and not any(ds._advantage is not None for ds, _ in dataset_mixture):
        raise FileNotFoundError(
            f"advantage_label_id={data_cfg.get('advantage_label_id')!r}: no dataset in mix "
            f"{data_mix!r} has a matching sidecar under glancewam_cache/advantage/ — typo, or "
            f"label_advantages.py not run against this data root?"
        )

    if eval_split == "eval":
        # Exhaustive, ordered coverage of every tiled held-out window; no
        # weighted sampling. Each single dataset's `_all_steps` is already the
        # tiled-window list, so a sequential pass over the concat enumerates
        # exactly the windows to score.
        return ConcatDataset([ds for ds, _ in dataset_mixture])

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/LIBERO/train_files/config_cotrain_baseline_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    for task_id in ["all"]:
        vla_dataset_cfg.task_id = task_id
        print(f"Testing Task ID: {task_id}")
        dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )

    cfg.output_dir = "./results/debug"
    output_dir = Path(cfg.output_dir)
    dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    from tqdm import tqdm

    count = 0
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        if count > 100:
            break
        count += 1
        pass
