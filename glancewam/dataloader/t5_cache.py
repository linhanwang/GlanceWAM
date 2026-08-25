"""Read-side helper for the T5 precompute cache produced by
``tools/precompute_t5_for_lerobot.py``.

Layout (per dataset, zarr v3 sharded):

    <dataset_dir>/glancewam_cache/t5/<safe_t5_id>_L<max_length>/
      manifest.json       — provenance, n_unique_prompts
      prompts.json        — list[str], same order as embeddings rows
      embeddings/         — sharded (N_unique, max_length, 1024) fp16
      n_tokens/           — single-shard (N_unique,) int32

The constructor resolves ``<safe_t5_id>_L<max_length>`` from the requested
model id + max_length, then lazily opens the array per worker pid (forked
DataLoader workers must not share zarr's internal state).

Random reads decompress one inner chunk (default 4 rows ≈ 4 MiB Blosc-LZ4),
not the whole shard — that's the point of v3 sharding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import zarr
import zarr.storage


def safe_model_id(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def t5_cache_subdir(t5_model_id: str, max_length: int) -> Path:
    """Path relative to a dataset root that holds the v3 T5 cache."""
    return Path("glancewam_cache") / "t5" / f"{safe_model_id(t5_model_id)}_L{max_length}"


class T5CacheReader:
    """Lazy, fork-safe reader for the per-dataset T5 cache (zarr v3 sharded)."""

    def __init__(
        self,
        dataset_dir: Path | str,
        t5_model_id: str = "nvidia/Cosmos-Predict2-2B-Video2World",
        max_length: int = 512,
    ):
        self.cache_dir = Path(dataset_dir) / t5_cache_subdir(t5_model_id, max_length)
        if not self.cache_dir.exists():
            raise FileNotFoundError(
                f"T5 cache not found at {self.cache_dir}. " f"Run tools/precompute_t5_for_lerobot.py first."
            )

        manifest_path = self.cache_dir / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        if self.manifest.get("cache_kind") != "t5":
            raise ValueError(
                f"manifest.json at {self.cache_dir} is not a t5 cache " f"(kind={self.manifest.get('cache_kind')})"
            )
        t5_meta = self.manifest["t5"]
        if int(t5_meta["max_length"]) != max_length:
            raise ValueError(f"T5 cache max_length mismatch: requested {max_length}, cache says {t5_meta['max_length']}")
        if t5_meta.get("model_id") != t5_model_id:
            raise ValueError(
                f"T5 cache model_id mismatch: requested {t5_model_id}, cache says {t5_meta.get('model_id')}"
            )

        self.max_length: int = int(t5_meta["max_length"])
        self.embed_dim: int = int(t5_meta["embed_dim"])

        prompts = json.loads((self.cache_dir / "prompts.json").read_text())
        self.prompts: list[str] = prompts
        # Pre-built lookup avoids re-hashing the same prompt across __getitem__ calls.
        self._prompt_to_idx: dict[str, int] = {s: i for i, s in enumerate(prompts)}

        # `n_tokens` is (N,) int32 ≈ ~80 KB for 20k prompts; eager-load is fine.
        # Open the group on a temporary local store, copy out the small array,
        # then drop the handle so workers re-open after fork.
        root = zarr.open_group(zarr.storage.LocalStore(str(self.cache_dir)), mode="r")
        n_tok_arr = root["n_tokens"]
        assert isinstance(n_tok_arr, zarr.Array)
        # `np.asarray` narrows zarr's union-typed __getitem__ return for `ty`;
        # always an ndarray at runtime for a 1-D zarr Array slice.
        self.n_tokens: np.ndarray = np.asarray(n_tok_arr[:])
        if self.n_tokens.shape[0] != len(prompts):
            raise RuntimeError(
                f"n_tokens length {self.n_tokens.shape[0]} != #prompts {len(prompts)} at {self.cache_dir}"
            )

        # Lazy per-pid handle to the `embeddings` array.
        self._embeds: zarr.Array | None = None
        self._embeds_pid: int | None = None

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._embeds is not None and self._embeds_pid == pid:
            return
        root = zarr.open_group(zarr.storage.LocalStore(str(self.cache_dir)), mode="r")
        arr = root["embeddings"]
        assert isinstance(arr, zarr.Array)
        self._embeds = arr
        self._embeds_pid = pid

    def has(self, prompt: str) -> bool:
        return prompt in self._prompt_to_idx

    def lookup(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(embed (max_length, embed_dim) fp16, mask (max_length,) bool)``."""
        idx = self._prompt_to_idx.get(prompt)
        if idx is None:
            raise KeyError(
                f"prompt not in T5 cache at {self.cache_dir}: {prompt!r}. "
                f"Cache covers {len(self.prompts)} prompts; rerun precompute if dataset changed."
            )
        self._ensure_open()
        assert self._embeds is not None
        # `np.asarray` narrows zarr 3's union-typed slice return for `ty`;
        # always (max_length, embed_dim) fp16 at runtime.
        embed = np.asarray(self._embeds[idx])
        n = int(self.n_tokens[idx])
        mask = np.zeros(self.max_length, dtype=np.bool_)
        mask[:n] = True
        return embed, mask
