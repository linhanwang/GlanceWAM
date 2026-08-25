import json
import os
from pathlib import Path

import numpy as np
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def _vla_worker_init(worker_id: int) -> None:
    """Cap intra-op threads to 1 in each DataLoader worker.

    The lerobot transforms use torch/numpy ops that otherwise spawn ~ncores
    intra-op threads *per worker*; with many workers this oversubscribes the
    CPU and slows decode under load. Benchmarked: with this cap, 8->16 workers
    nearly doubles loader throughput (83->162 samples/s) and collapses the
    tail (p90 5s->0.6s); without it, more workers thrash. Only the workers are
    capped — the main process (model compute) is untouched.
    """
    import torch

    torch.set_num_threads(1)


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")


def build_dataloader(
    cfg, dataset_py="lerobot_datasets_oxe", collate_fn=None
):  # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from glancewam.dataloader.lerobot_datasets import collate_fn as default_collate_fn
        from glancewam.dataloader.lerobot_datasets import get_vla_dataset

        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn if collate_fn is not None else default_collate_fn,
            num_workers=cfg.datasets.vla_data.get("num_workers", 8),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=cfg.datasets.vla_data.get("prefetch_factor", 4),
            shuffle=True,
            worker_init_fn=_vla_worker_init,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader

    raise ValueError(f"Unknown datasets.vla_data.dataset_py {dataset_py!r} (GlanceWAM supports 'lerobot_datasets').")


def build_eval_dataloader(cfg, collate_fn=None):
    """Held-out tiled eval set for the action-MSE monitor.

    Returns None when disabled — i.e. dataset is not `lerobot_datasets` or
    `datasets.vla_data.eval_holdout_episodes <= 0` (the default). Ordered,
    non-shuffled pass over every tiled held-out window (a ConcatDataset); the
    caller scores it on rank 0 only.
    """
    vla_dataset_cfg = cfg.datasets.vla_data
    if vla_dataset_cfg.dataset_py != "lerobot_datasets":
        return None
    if int(vla_dataset_cfg.get("eval_holdout_episodes", 0)) <= 0:
        return None

    from glancewam.dataloader.lerobot_datasets import collate_fn as default_collate_fn
    from glancewam.dataloader.lerobot_datasets import get_vla_dataset

    eval_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg, mode="val", eval_split="eval")
    if len(eval_dataset) == 0:
        logger.warning("Held-out eval set is empty (eval_holdout_episodes too large?); disabling eval-MSE monitor.")
        return None

    return DataLoader(
        eval_dataset,
        batch_size=int(cfg.trainer.get("eval_batch_size", 64)),
        collate_fn=collate_fn if collate_fn is not None else default_collate_fn,
        num_workers=int(vla_dataset_cfg.get("eval_num_workers", 4)),
        pin_memory=True,
        persistent_workers=False,
        shuffle=False,
        worker_init_fn=_vla_worker_init,
    )
