#!/usr/bin/env bash
# Precompute the UMT5 text cache for the RoboCasa KITCHEN SkyReelsV2-GR00T recipe.
#
# and examples/LIBERO/train_files/run_precompute_umt5_skyreels.sh.
# SkyReelsV2-GR00T needs two caches; only the text one is SkyReels-specific:
#   - text : UMT5-XXL (umt5-xxl, d_model=4096) last_hidden_state, [512, 4096] fp16
#            with padded positions ZEROED, built by tools/precompute_umt5_for_lerobot.py
#            and keyed under the Skywork repo id. Served through the standard
#            t5_cache reader (cache_kind="t5"); training with RESIDENT_TEXT_TABLE=True
#            sets skip_text_encoder=True so the ~11 GB UMT5-XXL never loads on the ranks.
#   - vae  : NOT built here (the kitchen run live-decodes; no VAE cache exists).
#
# The text cache is prompt-only (camera-independent), so it needs no camera/VAE
# flags. max_length=512 matches Wan-family pretraining and the t5_cache reader.
#
# Kitchen data layout: the 24 per-task lerobot v3 datasets live DIRECTLY under
# <DATA_ROOT> (unlike tabletop, whose episodes are one level deeper under the
# PhysicalAI-... subdir).
#
# Output (per dataset, under <DATA_ROOT>/<dataset>/glancewam_cache/):
#   t5/Skywork_SkyReels-V2-DF-1.3B-540P-Diffusers_L512/
# Idempotent + resumable: re-run without --force to fill gaps.
#
# Usage:
#   bash examples/Robocasa_kitchen/train_files/run_precompute_umt5_skyreels.sh
#   CUDA_VISIBLE_DEVICES=0 bash examples/Robocasa_kitchen/train_files/run_precompute_umt5_skyreels.sh
#   FORCE=1 bash examples/Robocasa_kitchen/train_files/run_precompute_umt5_skyreels.sh   # wipe + re-encode

set -euo pipefail

# === GPU (text tool runs on a single device) ===
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# === LD_LIBRARY_PATH (same as the training run scripts) ===
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
VENV_NV_LIB="${REPO_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cu12/lib"
export LD_LIBRARY_PATH="${VENV_NV_LIB}:${LD_LIBRARY_PATH:-}"

# === Run knobs (override via env) ===
# Kitchen datasets live directly under this dir (matches the training launcher's
# --datasets.vla_data.data_root_dir).
DATA_ROOT="${DATA_ROOT:-results/Datasets}/robocasa_cosmos_kitchen"
# Defaults below = the 24 kitchen datasets. Override DATASETS with a space-separated list
# to cover others.
# A missing cache is NOT a slow fallback: training with RESIDENT_TEXT_TABLE=True sets
# skip_text_encoder=True, so the ranks never build UMT5 and the miss surfaces mid-step as
# `AttributeError: 'NoneType' object has no attribute 'parameters'` in _encode_text.
# The tool is idempotent/resumable, so re-running over already-cached datasets is a no-op.
if [ -n "${DATASETS:-}" ]; then
  read -r -a DATASETS <<< "${DATASETS}"
else
DATASETS=(
  robocasa_CloseDoubleDoor_cosmos_lerobot
  robocasa_CloseDrawer_cosmos_lerobot
  robocasa_CloseSingleDoor_cosmos_lerobot
  robocasa_CoffeePressButton_cosmos_lerobot
  robocasa_CoffeeServeMug_cosmos_lerobot
  robocasa_CoffeeSetupMug_cosmos_lerobot
  robocasa_OpenDoubleDoor_cosmos_lerobot
  robocasa_OpenDrawer_cosmos_lerobot
  robocasa_OpenSingleDoor_cosmos_lerobot
  robocasa_PnPCabToCounter_cosmos_lerobot
  robocasa_PnPCounterToCab_cosmos_lerobot
  robocasa_PnPCounterToMicrowave_cosmos_lerobot
  robocasa_PnPCounterToSink_cosmos_lerobot
  robocasa_PnPCounterToStove_cosmos_lerobot
  robocasa_PnPMicrowaveToCounter_cosmos_lerobot
  robocasa_PnPSinkToCounter_cosmos_lerobot
  robocasa_PnPStoveToCounter_cosmos_lerobot
  robocasa_TurnOffMicrowave_cosmos_lerobot
  robocasa_TurnOffSinkFaucet_cosmos_lerobot
  robocasa_TurnOffStove_cosmos_lerobot
  robocasa_TurnOnMicrowave_cosmos_lerobot
  robocasa_TurnOnSinkFaucet_cosmos_lerobot
  robocasa_TurnOnStove_cosmos_lerobot
  robocasa_TurnSinkSpout_cosmos_lerobot
)
fi

# UMT5 text encoder — keyed on the Skywork repo id (must match the training run's
# --datasets.vla_data.t5_cache.model_id). max_length must match text_max_length.
UMT5_MODEL_ID="${UMT5_MODEL_ID:-Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers}"
UMT5_MAX_LENGTH="${UMT5_MAX_LENGTH:-512}"
UMT5_BATCH_SIZE="${UMT5_BATCH_SIZE:-16}"

FORCE="${FORCE:-0}"

cd "${REPO_ROOT}"

echo "========================================================================"
echo "[precompute umt5 robocasa kitchen] data_root=${DATA_ROOT}"
echo "[precompute umt5 robocasa kitchen] datasets=${#DATASETS[@]}"
echo "[precompute umt5 robocasa kitchen] model=${UMT5_MODEL_ID}  L=${UMT5_MAX_LENGTH}  gpu=${CUDA_VISIBLE_DEVICES}"
echo "========================================================================"

python -u tools/precompute_umt5_for_lerobot.py \
    --data-root-dir "${DATA_ROOT}" \
    --dataset-name "${DATASETS[@]}" \
    --umt5-model-id "${UMT5_MODEL_ID}" \
    --max-length "${UMT5_MAX_LENGTH}" \
    --batch-size "${UMT5_BATCH_SIZE}" \
    --device "cuda:0" \
    ${FORCE:+$([ "${FORCE}" = "1" ] && echo --force)}

echo "[precompute umt5 robocasa kitchen] done."
