# Precompute the UMT5 text cache for the LIBERO 4-in-1 SkyReelsV2-GR00T recipe.
#
# This is the TEXT half of the SkyReels precompute. SkyReelsV2-GR00T needs two
# caches; only the text one is SkyReels-specific:
#   - text : UMT5-XXL (umt5-xxl, d_model=4096) last_hidden_state, [512, 4096] fp16
#            with padded positions ZEROED, built by tools/precompute_umt5_for_lerobot.py
#            and keyed under the Skywork repo id. Served through the standard
#            t5_cache reader (cache_kind="t5"); training sets skip_text_encoder=True
#            so the ~11 GB UMT5-XXL never loads on the ranks.
#   - vae  : NOT built here. The SkyReels VAE is the SAME Wan2.1 VAE as
#            Cosmos-Predict2/2.5 (identical weights + norm stats), so the run
#            reuses the existing side-by-side 224x448 / 1-frame cache keyed under
#            nvidia/Cosmos-Predict2-2B-Video2World. Build it (once) with:
#              NUM_FRAMES=1 bash examples/LIBERO/train_files/run_precompute_cosmopredict2_5.sh
#            (or SKIP_VAE handling there) if you have not already.
#
# The text cache is prompt-only (camera-independent), so it needs no camera/VAE
# flags. max_length=512 matches the shipped configs (framework.world_model.text_max_length) and
# Wan-family pretraining. The unique-prompt set per LIBERO suite is tiny (tens of
# strings → one shard), so a single GPU finishes in seconds.
#
# Output (per dataset, under <DATA_ROOT>/<dataset>/glancewam_cache/):
#   t5/Skywork_SkyReels-V2-DF-1.3B-540P-Diffusers_L512/
# Idempotent + resumable: re-run without --force to fill gaps.
#
# Usage:
#   bash examples/LIBERO/train_files/run_precompute_umt5_skyreels.sh
#   CUDA_VISIBLE_DEVICES=0 bash examples/LIBERO/train_files/run_precompute_umt5_skyreels.sh
#   FORCE=1 bash examples/LIBERO/train_files/run_precompute_umt5_skyreels.sh   # wipe + re-encode

set -euo pipefail

# === GPU (text tool runs on a single device) ===
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# === LD_LIBRARY_PATH (same as the training run scripts) ===
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
VENV_NV_LIB="${REPO_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cu12/lib"
export LD_LIBRARY_PATH="${VENV_NV_LIB}:${LD_LIBRARY_PATH:-}"

# === Run knobs (override via env) ===
DATA_ROOT="${DATA_ROOT:-results/Datasets}"
# The tool is idempotent/resumable: datasets whose cache already exists are skipped.
# Override DATASETS with a space-separated list to build others.
if [ -n "${DATASETS:-}" ]; then
  read -r -a DATASETS <<< "${DATASETS}"
else
DATASETS=(
  libero_object_no_noops_1.0.0_lerobot
  libero_goal_no_noops_1.0.0_lerobot
  libero_spatial_no_noops_1.0.0_lerobot
  libero_10_no_noops_1.0.0_lerobot
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
echo "[precompute umt5] data_root=${DATA_ROOT}"
echo "[precompute umt5] datasets=${DATASETS[*]}"
echo "[precompute umt5] model=${UMT5_MODEL_ID}  L=${UMT5_MAX_LENGTH}  gpu=${CUDA_VISIBLE_DEVICES}"
echo "========================================================================"

python -u tools/precompute_umt5_for_lerobot.py \
    --data-root-dir "${DATA_ROOT}" \
    --dataset-name "${DATASETS[@]}" \
    --umt5-model-id "${UMT5_MODEL_ID}" \
    --max-length "${UMT5_MAX_LENGTH}" \
    --batch-size "${UMT5_BATCH_SIZE}" \
    --device "cuda:0" \
    ${FORCE:+$([ "${FORCE}" = "1" ] && echo --force)}

echo "[precompute umt5] done."
