"""Offline UMT5 precompute for LeRobot v3 datasets, for the SkyReels-V2 (Wan2.1) path.

Sibling of ``precompute_t5_for_lerobot.py`` with three deltas:
  - Encoder is ``UMT5EncoderModel`` (umt5-xxl, d_model=4096) loaded from the
    SkyReels/Wan diffusers repo's ``text_encoder`` subfolder, instead of the
    Cosmos-Predict2 T5 (1024-d).
  - Padded positions are ZEROED before writing (``embeds * attention_mask``),
    matching ``_SkyReelsV2DF_Interface._encode_text`` exactly — Wan-family DiT
    cross-attn takes no attention mask, so pads must be zero vectors. The
    framework consumes ``lang_embed`` directly without re-masking.
  - embed_dim is read from the encoder config (4096) rather than hardcoded.

The cache layout/manifest is the standard ``glancewam_cache/t5/`` zarr-v3 sharded
format (cache_kind="t5"), so the existing ``T5CacheReader`` + dataloader
``t5_cache`` config keys consume it unchanged — just point
``--datasets.vla_data.t5_cache.model_id`` at the same ``--umt5-model-id``.

For each (data_root, dataset_name): walk meta/tasks.parquet, dedupe the task
strings, UMT5-encode each unique string once at max_length=512, write

    <data_root>/<dataset_name>/glancewam_cache/t5/<safe_model_id>_L<max_length>/
      manifest.json     provenance (model id, revision, max_length, dtype, shard/chunk shape)
      prompts.json      list[str], same order as embeddings rows
      embeddings/       sharded (N_unique, L, 4096) fp16, padding zeroed
      n_tokens/         single-shard (N_unique,) int32

Resume/atomicity semantics are identical to the T5 tool (shard file existence
== complete; ``--force`` wipes and re-encodes).

Usage:
    python tools/precompute_umt5_for_lerobot.py \\
        --data-root-dir results/Datasets \\
        --dataset-name libero_10_no_noops_1.0.0_lerobot \\
        --umt5-model-id Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers \\
        --device cuda:0 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import zarr
import zarr.storage  # explicit so static analyzers see LocalStore
from zarr.codecs import BloscCodec

CACHE_VERSION = 2  # zarr-v3 sharded layout (same family as the T5 cache)

# Per-row size at max_length=512, embed_dim=4096, fp16 ≈ 4 MiB (4x the T5 rows).
# Inner chunk = 4 rows ≈ 16 MiB; shard = 1024 rows. Robot-instruction sets are
# tiny (tens of prompts), so everything lands in one shard file anyway.
CHUNK_ROWS = 4
SHARD_ROWS = 1024


def safe_model_id(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def cache_subdir(model_id: str, max_length: int) -> Path:
    """Per-model, per-max-length cache directory under glancewam_cache/t5/."""
    return Path("glancewam_cache") / "t5" / f"{safe_model_id(model_id)}_L{max_length}"


def load_unique_prompts(dataset_dir: Path) -> list[str]:
    """Return the deduped, sorted list of task strings from meta/tasks.parquet."""
    tasks_pq = dataset_dir / "meta" / "tasks.parquet"
    if not tasks_pq.exists():
        raise FileNotFoundError(f"tasks.parquet not found at {tasks_pq}")
    df = pd.read_parquet(tasks_pq)
    if df.index.name == "task":
        prompts = df.index.astype(str).tolist()
    elif "task" in df.columns:
        prompts = df["task"].astype(str).tolist()
    elif "task_index" in df.columns:
        # LeRobot v3 layout where the task string is the (unnamed) index and
        # task_index is the integer column (e.g. RoboCasa GR00T sim datasets).
        prompts = df.index.astype(str).tolist()
    else:
        raise RuntimeError(f"Unexpected schema in {tasks_pq}: columns={df.columns.tolist()}, index={df.index.name}")
    # Sorted + deduped so reruns produce the same row order.
    return sorted(set(prompts))


def load_umt5(model_id: str, device: str):
    """Load the UMT5 tokenizer + encoder in bf16, locked to eval/no-grad."""
    from transformers import T5TokenizerFast, UMT5EncoderModel

    tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.to(device)
    cfg_dict = getattr(encoder.config, "to_dict", lambda: {})()
    revision = cfg_dict.get("_commit_hash", None) or cfg_dict.get("transformers_version", None)
    embed_dim = int(encoder.config.d_model)
    return tokenizer, encoder, revision, embed_dim


@torch.no_grad()
def encode_prompts_batch(prompts: list[str], tokenizer, encoder, max_length: int, device: str):
    """Encode -> (embeds [B, L, 4096] fp16 numpy with padding zeroed, n_tokens [B] int32 numpy).

    Mirrors ``_SkyReelsV2DF_Interface._encode_text``: max_length padding,
    truncation, special tokens, then padded positions multiplied to zero.
    """
    toks = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device)
    out = encoder(input_ids=toks.input_ids, attention_mask=toks.attention_mask).last_hidden_state
    out = out * toks.attention_mask.unsqueeze(-1)
    n_tokens = toks.attention_mask.sum(dim=1).to(torch.int32).cpu().numpy()
    embeds = out.to(torch.float16).cpu().numpy()
    return embeds, n_tokens


def build_manifest(
    dataset_name: str,
    model_id: str,
    revision: str | None,
    max_length: int,
    embed_dim: int,
    n_unique: int,
) -> dict:
    return {
        "version": CACHE_VERSION,
        # Keep cache_kind="t5" so T5CacheReader (the generic text-cache reader)
        # accepts it; the model_id + embed_dim fields disambiguate UMT5.
        "cache_kind": "t5",
        "dataset_name": dataset_name,
        "layout": "zarr_v3_sharded",
        "t5": {
            "model_id": model_id,
            "subfolder": "text_encoder",
            "encoder_class": "UMT5EncoderModel",
            "revision": revision,
            "max_length": max_length,
            "embed_dim": embed_dim,
            "dtype": "float16",
            "padding_zeroed": True,
            "chunks": [CHUNK_ROWS, max_length, embed_dim],
            "shards": [SHARD_ROWS, max_length, embed_dim],
            "compressor": {"name": "blosc", "cname": "lz4", "clevel": 1, "shuffle": "bitshuffle"},
        },
        "n_unique_prompts": n_unique,
    }


def manifest_matches(existing: dict, desired: dict) -> bool:
    """Same kind, version, model_id, max_length, and embed_dim → cache is reusable."""
    if not existing:
        return False
    if existing.get("version") != desired["version"] or existing.get("cache_kind") != desired["cache_kind"]:
        return False
    et5, dt5 = existing.get("t5", {}), desired["t5"]
    return all(et5.get(k) == dt5.get(k) for k in ("model_id", "subfolder", "max_length", "embed_dim", "dtype"))


def shard_exists(arr_dir: Path, shard_idx: int, n_extra_dims: int) -> bool:
    """A v3 sharded array's data file lives at ``<arr_dir>/c/<i0>/<i1>/.../<iN>``."""
    rest = "/".join(["0"] * n_extra_dims)
    p = arr_dir / "c" / str(shard_idx)
    if rest:
        p = p / rest
    return p.exists()


def precompute_for_dataset(
    data_root: Path,
    dataset_name: str,
    model_id: str,
    max_length: int,
    batch_size: int,
    device: str,
    force: bool,
    shared_encoder=None,
) -> tuple | None:
    """Encode one dataset. Returns the (tokenizer, encoder, revision, embed_dim)
    tuple so the multi-GB UMT5 encoder is loaded once across datasets."""
    dataset_dir = data_root / dataset_name
    cache_dir = dataset_dir / cache_subdir(model_id, max_length)

    prompts = load_unique_prompts(dataset_dir)
    n_unique = len(prompts)
    if n_unique == 0:
        print(f"[{dataset_name}] no unique prompts; skipping")
        return shared_encoder
    n_shards = (n_unique + SHARD_ROWS - 1) // SHARD_ROWS
    print(f"[{dataset_name}] {n_unique} unique prompts → {n_shards} shard(s)")

    if force and cache_dir.exists():
        shutil.rmtree(cache_dir)

    # embed_dim is needed for the manifest check; defer the (expensive) encoder
    # load until we know there is work, except when a manifest must be compared.
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        if shared_encoder is None:
            shared_encoder = load_umt5(model_id, device)
        _, _, revision, embed_dim = shared_encoder
        desired_manifest = build_manifest(dataset_name, model_id, revision, max_length, embed_dim, n_unique)
        existing = json.loads(manifest_path.read_text())
        if not manifest_matches(existing, desired_manifest):
            raise RuntimeError(
                f"Existing manifest at {manifest_path} is incompatible with requested config. "
                f"Use --force to wipe, or delete the dir."
            )
        if existing.get("n_unique_prompts") != n_unique:
            print(
                f"[{dataset_name}] WARNING: tasks.parquet now has {n_unique} prompts but cached manifest "
                f"says {existing.get('n_unique_prompts')}; rebuilding from scratch."
            )
            shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False))

    # Resume: which shards already have a file on disk?
    emb_dir = cache_dir / "embeddings"
    todo_shards: list[int] = [s for s in range(n_shards) if not shard_exists(emb_dir, s, n_extra_dims=2)]
    if not todo_shards:
        print(f"[{dataset_name}] all {n_shards} shard(s) already encoded")
        return shared_encoder

    if shared_encoder is None:
        shared_encoder = load_umt5(model_id, device)
    tokenizer, encoder, revision, embed_dim = shared_encoder

    store = zarr.storage.LocalStore(str(cache_dir))
    root = zarr.open_group(store=store, mode="a")
    blosc = BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")
    if "embeddings" not in root:
        root.create_array(
            name="embeddings",
            shape=(n_unique, max_length, embed_dim),
            chunks=(CHUNK_ROWS, max_length, embed_dim),
            shards=(SHARD_ROWS, max_length, embed_dim),
            dtype="float16",
            compressors=blosc,
        )
    if "n_tokens" not in root:
        root.create_array(
            name="n_tokens",
            shape=(n_unique,),
            chunks=(n_unique,),
            shards=(n_unique,),
            dtype="int32",
        )
    emb = root["embeddings"]
    n_tok = root["n_tokens"]
    assert isinstance(emb, zarr.Array) and isinstance(n_tok, zarr.Array)

    print(f"[{dataset_name}] encoding {len(todo_shards)}/{n_shards} shard(s)  batch={batch_size}")
    t0 = time.perf_counter()
    for s in todo_shards:
        row_lo = s * SHARD_ROWS
        row_hi = min((s + 1) * SHARD_ROWS, n_unique)
        n_rows = row_hi - row_lo
        buf = np.empty((n_rows, max_length, embed_dim), dtype=np.float16)
        ntok_buf = np.empty((n_rows,), dtype=np.int32)
        for j0 in range(0, n_rows, batch_size):
            j1 = min(j0 + batch_size, n_rows)
            batch_prompts = prompts[row_lo + j0 : row_lo + j1]
            embeds, ntoks = encode_prompts_batch(batch_prompts, tokenizer, encoder, max_length, device)
            buf[j0:j1] = embeds
            ntok_buf[j0:j1] = ntoks
        emb[row_lo:row_hi] = buf
        n_tok[row_lo:row_hi] = ntok_buf
        dt = time.perf_counter() - t0
        print(f"[{dataset_name}]   shard {s + 1}/{n_shards}  ({n_rows} rows)  {dt:.1f}s elapsed")

    manifest = build_manifest(dataset_name, model_id, revision, max_length, embed_dim, n_unique)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{dataset_name}] done.  cache: {cache_dir}")
    return shared_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root-dir", type=Path, required=True, help="e.g. results/Datasets")
    ap.add_argument(
        "--dataset-name",
        nargs="+",
        required=True,
        help="One or more dataset directory names (may be nested paths) under --data-root-dir.",
    )
    ap.add_argument("--umt5-model-id", default="Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true", help="Wipe existing cache dir and re-encode.")
    args = ap.parse_args()

    shared_encoder = None
    for name in args.dataset_name:
        shared_encoder = precompute_for_dataset(
            data_root=args.data_root_dir.expanduser(),
            dataset_name=name,
            model_id=args.umt5_model_id,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
            force=args.force,
            shared_encoder=shared_encoder,
        )


if __name__ == "__main__":
    main()
