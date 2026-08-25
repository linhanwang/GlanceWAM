# Copyright 2025 glancewam community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
GlanceWAM’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from glancewam.dataloader import build_dataloader
from glancewam.model.framework.base_framework import build_framework
from glancewam.model.framework.share_tools import apply_config_compat
from glancewam.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from glancewam.training.trainer_utils.ema import EMAModel
from glancewam.training.trainer_utils.hf_upload import upload_run_to_hf
from glancewam.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True
# TF32 tensor cores for the FP32 matmuls that remain under bf16 autocast (residual/
# norm math, any fp32-fallback GEMMs). Big win on A100 (Ampere), negligible-precision
# cost next to bf16 training noise; on H200 the bf16 path already dominates so this is
# just a safe default. Honors an explicit override via env if a run needs strict fp32.
if os.environ.get("GLANCEWAM_DISABLE_TF32", "0") in ("0", "", "false", "False"):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# SDPA backend selection is honored inside each framework via the
# ``torch.nn.attention.sdpa_kernel`` context manager scoped around the
# attention-heavy forward (DiT for video frameworks).  The trainer here only
# logs the user's choice so it shows up in the run header; the actual dispatch
# happens in the framework (see e.g. CosmoPredict2PI_v3_flex._run_backbone_denoise_and_tap).
#
# Why context manager and not process-wide toggles: on torch 2.10 the
# `enable_*_sdp(bool)` toggles apply a stricter `can_use_*` check than the
# forced `sdpa_kernel([...])` path.  Empirically (2026-05-23 H200, Cosmos DiT,
# T_lat=4) the toggle path silently rejects cuDNN and falls to math fallback
# (~400 ms vs 222 ms), whereas the context manager dispatches cuDNN reliably.
def _log_sdpa_backend_choice(cfg) -> None:
    backend = "auto"
    if cfg is not None:
        backend = cfg.get("trainer", {}).get("sdpa_backend", "auto") if hasattr(cfg, "get") else "auto"
    accelerator.print(f">> [*] SDPA: framework will dispatch via sdpa_kernel(backend={backend!r}).")


# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir, collate_fn=None) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py, collate_fn=collate_fn)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    scheduler_total_steps = cfg.trainer.get("scheduler_total_steps", None) or cfg.trainer.max_train_steps
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=scheduler_total_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self._micro_step = 0
        self.total_batch_size = self._calculate_total_batch_size()

        # EMA (optional, default off). Built in prepare_training() after the model
        # is wrapped; lives on rank 0 only. `_ema_resume_path` is set by
        # _init_checkpointing when resuming so the shadow continues accumulating.
        self.ema = None
        self._ema_resume_path = None

        # Held-out action-MSE monitor. Built on
        # rank 0 only in prepare_training when datasets.vla_data.eval_holdout_episodes > 0.
        self.eval_dataloader = None

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_ema()

        self._init_eval_dataloader()

        self._init_wandb()

    def _init_eval_dataloader(self):
        """Build the held-out tiled eval set on rank 0.

        No-op unless `datasets.vla_data.eval_holdout_episodes > 0`. Built on the
        Built on EVERY rank: `predict_action` must be called collective-symmetrically
        (compile_dit recompiles / DeepSpeed touches desync if only rank 0 runs the
        forward → deadlock). All ranks iterate the same deterministic windows and run
        the forward; only rank 0 keeps the metric (see `_eval_on_holdout`).
        """
        if int(self.config.datasets.vla_data.get("eval_holdout_episodes", 0)) <= 0:
            return

        from glancewam.dataloader import build_eval_dataloader

        unwrapped = self.accelerator.unwrap_model(self.model)
        collate_fn = unwrapped.build_collate_fn() if hasattr(unwrapped, "build_collate_fn") else None
        self.eval_dataloader = build_eval_dataloader(self.config, collate_fn=collate_fn)
        if self.eval_dataloader is not None:
            logger.info(
                f"✅ Held-out eval-MSE monitor: {len(self.eval_dataloader.dataset)} tiled windows, "
                f"batch={int(self.config.trainer.get('eval_batch_size', 64))}"
            )

    def _init_ema(self):
        """Build the EMA shadow on rank 0 if `trainer.ema.enabled` (default off).

        Valid to keep on rank 0 only: under DeepSpeed ZeRO-2 parameters are
        replicated across ranks (only optimizer state / gradients are partitioned),
        so the unwrapped model's live params on rank 0 are the full weights.
        """
        ema_cfg = self.config.trainer.get("ema", None)
        if not (ema_cfg and ema_cfg.get("enabled", False)):
            return
        if not self.accelerator.is_main_process:
            return

        unwrapped = self.accelerator.unwrap_model(self.model)
        trainable = [(n, p) for n, p in unwrapped.named_parameters() if p.requires_grad]
        self.ema = EMAModel(
            trainable,
            decay=float(ema_cfg.get("decay", 0.999)),
            warmup=bool(ema_cfg.get("warmup", True)),
            device=str(ema_cfg.get("device", "cuda")),
        )
        logger.info(
            f">> [*] EMA enabled: decay={self.ema.decay}, warmup={self.ema.warmup}, "
            f"device={self.ema.device}, start_step={int(ema_cfg.get('start_step', 0))}, "
            f"tracking {len(self.ema.shadow)} trainable tensors."
        )

        # Resume: continue accumulation from a saved EMA sibling if present.
        if self._ema_resume_path and os.path.isfile(self._ema_resume_path):
            ema_sd = torch.load(self._ema_resume_path, map_location="cpu")
            self.ema.load_state_dict(ema_sd)
            logger.info(f">> [*] EMA shadow restored from {self._ema_resume_path}")

    def _update_ema(self):
        """Pull live trainable params into the EMA shadow at a grad-accum boundary."""
        if self.ema is None:
            return
        ema_cfg = self.config.trainer.get("ema", None)
        start_step = int(ema_cfg.get("start_step", 0)) if ema_cfg else 0
        if self.completed_steps < start_step:
            return
        self.ema.update(self.accelerator.unwrap_model(self.model), self.completed_steps)

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb_dir = getattr(self.config, "wandb_dir", None) or os.path.join(self.config.output_dir, "wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(
                name=self.config.run_id,
                dir=wandb_dir,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                self._ema_resume_path = self._ema_sibling_path(resume_from_checkpoint)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    @staticmethod
    def _ema_sibling_path(checkpoint_file):
        """Map a checkpoint file path to its EMA sibling (insert `_ema` before suffix).

        steps_5_pytorch_model.pt        -> steps_5_pytorch_model_ema.pt
        steps_5_model.safetensors       -> steps_5_model_ema.safetensors
        """
        p = Path(checkpoint_file)
        return str(p.with_name(p.stem + "_ema" + p.suffix))

    def _write_state_dict(self, state_dict, path, save_format):
        """Write a state_dict to `path` in the configured format."""
        if save_format == "safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, path)
        elif save_format == "pt":
            torch.save(state_dict, path)
        else:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            suffix = "_model.safetensors" if save_format == "safetensors" else "_pytorch_model.pt"

            state_dict = self.accelerator.get_state_dict(self.model)
            state_dict = self.filter_savable_state_dict(self.accelerator.unwrap_model(self.model), state_dict)
            self._write_state_dict(state_dict, checkpoint_path + suffix, save_format)

            # Save the EMA model alongside the live one (identical key set, so it
            # loads via from_pretrained by just pointing the path at this file).
            if self.ema is not None:
                model_dtype = next(self.accelerator.unwrap_model(self.model).parameters()).dtype
                ema_state_dict = self.ema.merge_into(state_dict, dtype=model_dtype)
                self._write_state_dict(ema_state_dict, self._ema_sibling_path(checkpoint_path + suffix), save_format)

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if (
            self.completed_steps == 1 or self.completed_steps % self.config.trainer.logging_frequency == 0
        ) and self.accelerator.is_main_process:
            metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, g in enumerate(self.optimizer.param_groups):
                name = g.get("name", f"group_{i}")
                metrics[f"learning_rate/{name}"] = last_lrs[i]
            metrics["learning_rate"] = last_lrs[0]
            metrics["epoch"] = round(
                self.completed_steps * self.accelerator.gradient_accumulation_steps / len(self.vla_train_dataloader),
                2,
            )
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        # trainer.gc_interval > 0: disable automatic Python GC and collect
        # manually every N optimizer steps (Megatron/torchtitan pattern).
        # CPython's allocation-count-triggered gen-2 collections scan the whole
        # model + compiled-artifact object graph and land mid-step as ~0.4 s
        # stalls (measured on the SkyReels backbone at B=64: fwd+bwd std 53 ms -> 3 ms
        # with GC controlled). 0 (default) leaves GC untouched.
        gc_interval = int(self.config.trainer.get("gc_interval", 0) or 0)
        if gc_interval > 0:
            import gc

            gc.collect()
            gc.disable()
            logger.info(f">> [*] trainer.gc_interval={gc_interval}: automatic GC off, manual collect at that cadence.")

        # trainer.step_timing: per-step sync-bracketed phase walls (fwd/bwd/opt
        # + framework sub-phases via GLANCEWAM_STEP_TIMING) appended as JSONL to
        # output_dir/step_timing_rank{r}.jsonl on EVERY rank. Diagnostic only —
        # the phase syncs cost a few ms/step. Default off.
        self._step_timing = bool(self.config.trainer.get("step_timing", False))
        self._timing_fh = None
        if self._step_timing:
            os.environ["GLANCEWAM_STEP_TIMING"] = "1"
            timing_path = os.path.join(self.config.output_dir, f"step_timing_rank{self.accelerator.process_index}.jsonl")
            self._timing_fh = open(timing_path, "a", buffering=1)
            logger.info(f">> [*] trainer.step_timing=True: per-step phase walls -> {timing_path}")

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()
            self._last_data_time = t_end_data - t_start_data

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            is_boundary = step_metrics.pop("_is_boundary", True)

            # Live timing feedback refreshes every micro-batch; the postfix is
            # cheap and independent of the optimizer-step counter.
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            # Everything below is per-OPTIMIZER-STEP work. Under gradient
            # accumulation `completed_steps` is constant across the N micro-batches
            # of a window, so eval/log/save MUST be gated on the boundary — else
            # `completed_steps % eval_interval == 0` (true for all N micro-batches
            # while completed_steps==0) fires the eval N times and pins the bar at 0.
            if not is_boundary:
                continue

            progress_bar.update(1)
            self.completed_steps += 1
            self._update_ema()
            if gc_interval > 0 and self.completed_steps % gc_interval == 0:
                import gc

                gc.collect()

            if self.completed_steps == 1 or self.completed_steps % self.config.trainer.eval_interval == 0:
                _t_eval = time.perf_counter()
                step_metrics = self.eval_action_model(step_metrics)
                if self.accelerator.is_main_process:
                    _eval_dt = time.perf_counter() - _t_eval
                    step_metrics["eval/eval_time_s"] = _eval_dt
                    logger.info(f">> [eval] action-MSE monitor took {_eval_dt:.1f}s at step {self.completed_steps}.")

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    @contextlib.contextmanager
    def _ema_weights_applied(self, model):
        """Temporarily load the EMA shadow into `model`, restoring live weights on exit.

        Backs up and swaps only the tracked (trainable) params, on rank 0 — the
        only rank holding the shadow. Used to score the EMA weights during the
        in-loop action-eval without disturbing the optimization state.
        """
        backup = {}
        try:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    shadow = self.ema.shadow.get(name)
                    if shadow is None:
                        continue
                    backup[name] = param.detach().clone()
                    param.data.copy_(shadow.to(device=param.device, dtype=param.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup:
                        param.data.copy_(backup[name])

    def eval_action_model(self, step_metrics: dict | None = None) -> float:
        """Action-MSE eval. Routes to the held-out monitor when enabled, else the
        legacy in-distribution train-batch score.

        When `datasets.vla_data.eval_holdout_episodes > 0`, score a fixed,
        held-out, fully-tiled set of episodes and log `eval/action_mse` (replacing
        `mse_score`) Otherwise (default)
        keep the original behaviour exactly.
        """
        if int(self.config.datasets.vla_data.get("eval_holdout_episodes", 0)) > 0:
            return self._eval_on_holdout(step_metrics)
        return self._eval_on_train_batch(step_metrics)

    def _eval_on_train_batch(self, step_metrics: dict | None = None) -> float:
        """Legacy in-distribution action-eval on the current training batch."""
        batch = self._get_next_batch()
        # When the framework supplies a worker-side collate, the dataloader
        # yields a pre-tokenised mapping (BatchFeature, not a dict subclass)
        # that smuggles the raw samples under `_raw_examples`. predict_action
        # still consumes the raw List[dict].
        examples = batch if isinstance(batch, list) else batch["_raw_examples"]
        actions = [example["action"] for example in examples]
        model = self.accelerator.unwrap_model(self.model)
        output_dict = model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)

        if self.accelerator.is_main_process:
            actions = np.array(actions)
            normalized_actions = np.asarray(output_dict["normalized_actions"])
            step_metrics["mse_score"] = float(((normalized_actions - actions) ** 2).mean())

            # Score the EMA weights on the same batch (rank 0 holds the shadow and
            # is the only rank that needs the number). Swap EMA in, predict, restore.
            # predict_action runs on the unwrapped module (no ZeRO collectives), so
            # this extra rank-0-only forward is safe; eval is infrequent so the cost
            # (one forward + a transient param backup) is negligible.
            if self.ema is not None:
                with self._ema_weights_applied(model):
                    ema_out = model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)
                ema_actions = np.asarray(ema_out["normalized_actions"])
                step_metrics["mse_score_ema"] = float(((ema_actions - actions) ** 2).mean())

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _score_eval_windows(self, model) -> float:
        """Mean squared error over ALL tiled held-out windows, in normalized
        action space, flattened across chunk steps and action dims.

        Runs on EVERY rank — `predict_action` must be collective-symmetric or
        compile_dit/DeepSpeed desync and deadlock. Every rank iterates the same
        deterministic windows with the same per-batch seed, so all ranks compute
        the identical number; only rank 0 logs it (see `_eval_on_holdout`)."""
        sq_sum, count = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in self.eval_dataloader:
                examples = batch if isinstance(batch, list) else batch["_raw_examples"]
                torch.manual_seed(0)
                out = model.predict_action(examples=examples)
                actions = np.asarray([e["action"] for e in examples], dtype=np.float32)
                pred = np.asarray(out["normalized_actions"], dtype=np.float32)
                diff = (pred - actions) ** 2
                sq_sum += float(diff.sum())
                count += diff.size
        return sq_sum / max(count, 1)

    def _eval_on_holdout(self, step_metrics: dict | None = None) -> float:
        """Held-out action-MSE monitor.

        Two forwards, mirroring the legacy `mse_score` / `mse_score_ema` structure:
        - BASE (live weights) runs on ALL ranks — the FIRST forward on the eval path
          compiles the compile_dit graph / sets up any cross-rank state, which under
          DeepSpeed must happen on every rank together. A rank-0-only first forward
          desyncs and deadlocks. Only rank 0 keeps the (identical) number.
        - EMA (rank 0 only) reuses what the all-ranks base forward established (a
          weight-value swap does not recompile — guards key on shape/dtype), so it
          is safe rank-0-only; the other ranks wait at the trailing barrier. Emitted
          only when EMA is enabled.
        """
        if self.eval_dataloader is not None:
            model = self.accelerator.unwrap_model(self.model)
            was_training = model.training
            model.eval()
            try:
                # Base: all ranks (compiles the eval path symmetrically).
                mse = self._score_eval_windows(model)
                if self.accelerator.is_main_process:
                    step_metrics["eval/action_mse"] = mse
                # EMA: rank 0 only, reusing the now-established eval path.
                if self.ema is not None and self.accelerator.is_main_process:
                    with self._ema_weights_applied(model):
                        step_metrics["eval/action_mse_ema"] = self._score_eval_windows(model)
            finally:
                if was_training:
                    model.train()

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        timing_on = getattr(self, "_step_timing", False)
        if timing_on:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # DeepSpeed engine handles gradient accumulation internally; accelerator.accumulate()
        # is incompatible with ZeRO-2. Gradient clipping is configured in ds_config.yaml.
        # No torch.autocast: DeepSpeed already runs the model in bf16.
        output_dict = self.model.forward(batch_vla)
        action_loss = output_dict["action_loss"]
        if timing_on:
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        self.accelerator.backward(action_loss)
        if timing_on:
            torch.cuda.synchronize()
            t2 = time.perf_counter()

        self.optimizer.step()
        self.optimizer.zero_grad()
        if timing_on:
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            rec = {
                "micro": self._micro_step + 1,
                "data": round(getattr(self, "_last_data_time", -1.0), 4),
                "fwd": round(t1 - t0, 4),
                "bwd": round(t2 - t1, 4),
                "opt": round(t3 - t2, 4),
                "load1": round(os.getloadavg()[0], 1),
            }
            for k, v in (output_dict.get("timing") or {}).items():
                rec[k] = round(v, 4)
            self._timing_fh.write(json.dumps(rec) + "\n")

        self._micro_step += 1
        is_boundary = self._micro_step % self.accelerator.gradient_accumulation_steps == 0
        if is_boundary:
            self.lr_scheduler.step()

        # `action_dit_loss` is the action-DiT loss, kept directly comparable ACROSS
        # frameworks. Co-training frameworks (e.g. CotrainBaseline) overload
        # output_dict["action_loss"] with the COMBINED objective (action + λ·video) — that
        # is what must be backpropped — but they also return the pure "action_only_loss".
        # Log that pure component as action_dit_loss so it lines up with action-only runs
        # (which return only "action_loss"); the combined objective is logged as total_loss.
        action_dit_loss = output_dict.get("action_only_loss", action_loss)
        step_metrics = {
            "action_dit_loss": action_dit_loss.detach(),
            "_is_boundary": is_boundary,
        }
        # For co-training frameworks, also log the combined objective (total_loss) and the
        # video split. `action_only_loss` itself is NOT logged — it equals action_dit_loss
        # above, so it would be a duplicate. (The framework still returns it; that's what
        # action_dit_loss is read from.)
        if "action_only_loss" in output_dict:
            step_metrics["total_loss"] = action_loss.detach()
        if "video_loss" in output_dict:
            step_metrics["video_loss"] = output_dict["video_loss"].detach()
        # E6 auxiliary depth supervision (framework.depth_supervision.enabled=True). Logged
        # UNWEIGHTED — total_loss already carries loss_weight * depth_loss — so the curve is
        # comparable across weights and readable as "is the DPT head learning at all?".
        if "depth_loss" in output_dict:
            step_metrics["depth_loss"] = output_dict["depth_loss"].detach()
        # AWR weight distribution (rung-3 collapse check): logged only when the
        # framework runs with framework.awr.enabled=True.
        for awr_key in ("awr_w_mean", "awr_w_p95", "awr_w_min", "awr_w_max"):
            if awr_key in output_dict:
                step_metrics[awr_key] = output_dict[awr_key].detach()
        # Rung-12a lift-off guard: ‖W‖ of the zero-init action-conditioning embedder —
        # must leave 0 within the first optimizer steps; pinned at 0 ⇒ wiring bug.
        if "action_cond_w_norm" in output_dict:
            step_metrics["action_cond_w_norm"] = output_dict["action_cond_w_norm"].detach()
        # §8.3 state head: flow-matching loss of the future-state DiT (named to line up
        # with action_dit_loss), plus the §8.2 state-conditioning embedder's lift-off
        # guard (same semantics as action_cond_w_norm above).
        if "state_loss" in output_dict:
            step_metrics["state_dit_loss"] = output_dict["state_loss"].detach()
        if "state_cond_w_norm" in output_dict:
            step_metrics["state_cond_w_norm"] = output_dict["state_cond_w_norm"].detach()
        # Pre-clip global grad norm computed by DeepSpeed at each boundary.
        if is_boundary and hasattr(self.model, "get_global_grad_norm"):
            gn = self.model.get_global_grad_norm()
            if gn is not None:
                step_metrics["grad_norm"] = float(gn)
        return step_metrics

    def _finalize_training(self):
        """Training end processing."""
        # Barrier before the rank-0-only save / wandb.finish so other ranks
        # don't sit on a NCCL collective while rank 0 does slow local work.
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            final_name = "model.safetensors" if save_format == "safetensors" else "pytorch_model.pt"
            state_dict = self.accelerator.get_state_dict(self.model)
            state_dict = self.filter_savable_state_dict(self.accelerator.unwrap_model(self.model), state_dict)
            self._write_state_dict(state_dict, os.path.join(final_checkpoint, final_name), save_format)

            if self.ema is not None:
                model_dtype = next(self.accelerator.unwrap_model(self.model).parameters()).dtype
                ema_state_dict = self.ema.merge_into(state_dict, dtype=model_dtype)
                self._write_state_dict(
                    ema_state_dict,
                    self._ema_sibling_path(os.path.join(final_checkpoint, final_name)),
                    save_format,
                )

            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

            # wandb.finish() has been observed to deadlock on this SLURM cluster
            # (parent <-> wandb-service IPC hangs indefinitely while the Go
            # uploader is healthy). Cap it so the HF upload step downstream
            # actually runs.
            t = threading.Thread(target=wandb.finish, daemon=True)
            t.start()
            t.join(timeout=180)
            if t.is_alive():
                logger.warning("wandb.finish() exceeded 180s; abandoning and continuing.")


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    _log_sdpa_backend_choice(cfg)

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    # If the framework offers a worker-safe collate (pre-tokenised batches),
    # wire it into the DataLoader so build_qwenvl_inputs runs in DataLoader
    # workers and overlaps with the previous step's GPU compute.
    collate_fn = vla.build_collate_fn() if hasattr(vla, "build_collate_fn") else None
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir, collate_fn=collate_fn)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    if accelerator.is_main_process:
        upload_run_to_hf(cfg)

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/config_cotrain_baseline_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
