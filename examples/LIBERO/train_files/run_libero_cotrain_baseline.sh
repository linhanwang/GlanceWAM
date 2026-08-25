#!/usr/bin/env bash
# SkyReels-V2 DF + GR00T head — VIDEO + ACTION CO-TRAINING on LIBERO (4-in-1).
#
# One forward pass jointly trains the GR00T action head (on the clean obs frame) and the
# SkyReels DF video objective (on the noised future frame); both losses backprop into the
# unfrozen DiT theta. See config_cotrain_baseline_libero.yaml and
# glancewam/model/framework/wam/CotrainBaseline.py.
#
# PREREQUISITE — the UMT5 text cache (RESIDENT_TEXT_TABLE=True by default):
#   bash examples/LIBERO/train_files/run_precompute_umt5_skyreels.sh
#
# Defaults: from-scratch on the LIBERO demos (mixture libero_all) from the stock
# SkyReels-V2 DF-1.3B checkpoint, 4 GPUs x bsz 16 x grad-accum 2 = global batch 128, 20k steps.
#
# Environment overrides:
#   CUDA_VISIBLE_DEVICES   GPUs to use (default 0-3; NUM_PROCESSES is derived from it)
#   DATA_ROOT              lerobot v3 dataset root (default results/Datasets)
#   DATA_MIX               dataset mixture (default libero_all)
#   BASE_WM                backbone checkpoint; must contain "skyreels" so the world-model
#                          factory routes to the SkyReelsV2DF wrapper
#   RUN_ID                 checkpoint dir name (default: auto-tagged)
#   MAX_TRAIN_STEPS        default 20000
#   PRETRAINED_CHECKPOINT  load weights only; steps restart at 0
#   IS_RESUME=True         continue the latest checkpoint of the SAME run_id

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# Force torch's bundled cuBLAS/cuBLASLt to win over any system CUDA toolkit on LD_LIBRARY_PATH.
export LD_LIBRARY_PATH="${REPO_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"

# torchcodec needs ffmpeg 6 .so files; redirect to PyAV's wheel-bundled libs via the shim.
FFMPEG6_SHIM="${REPO_ROOT}/.venv/lib/ffmpeg6_shim"
AV_LIBS="${REPO_ROOT}/.venv/lib/python3.11/site-packages/av.libs"
[ -d "${FFMPEG6_SHIM}" ] || python "${REPO_ROOT}/tools/setup_ffmpeg6_shim.py"
export LD_LIBRARY_PATH="${FFMPEG6_SHIM}:${AV_LIBS}:${LD_LIBRARY_PATH}"

# === Paths ===
config_yaml=./examples/LIBERO/train_files/config_cotrain_baseline_libero.yaml
data_root_dir="${DATA_ROOT:-results/Datasets}"
run_root_dir=./results/Checkpoints

# === Recipe ===
base_wm="${BASE_WM:-Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers}"
data_mix="${DATA_MIX:-libero_all}"
max_train_steps="${MAX_TRAIN_STEPS:-20000}"
save_interval="${SAVE_INTERVAL:-5000}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-16}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"

action_horizon=8
repeated_diffusion_steps="${RDS:-8}"
extract_layers="${EXTRACT_LAYERS:-[19]}"
num_history_frames="${NUM_HISTORY_FRAMES:-1}"
# The ONE future frame the DiT predicts, at t+future_frame_idx NATIVE rows — same units as
# action_horizon and independent of it. Runs are tagged w<future_frame_idx>.
future_frame_idx="${FUTURE_FRAME_IDX:-${action_horizon}}"
# 224x448 = the two LIBERO cameras (primary | wrist) stitched side-by-side.
vae_input_size="[224,448]"

# Video-loss weight: total = action + lambda * video.
cotrain_lambda="${COTRAIN_LAMBDA:-1.0}"
# logitnormal (shift=5) matches the LIBERO video LoRA finetune, so video_loss curves are
# directly comparable. `uniform` instead gives SkyReels FoPP marginals.
sigma_sampling="${SIGMA_SAMPLING:-logitnormal}"
clean_prefix_isolation="${CLEAN_PREFIX_ISOLATION:-True}"
# torch.compile on the DiT blocks and on the frozen Wan VAE encoder. compile_vae is ~1.6x on
# the encode at ~0.2% systematic latent drift (bf16 conv lowering) — self-consistent across
# train/eval and far below the flow-matching target floor. COMPILE_VAE=False for a bit-exact
# encode.
compile_dit="${COMPILE_DIT:-True}"
compile_vae="${COMPILE_VAE:-True}"

# Appearance-only image aug (color jitter + JPEG q95, no geometric ops). Geometry-free, so
# eval needs no counterpart. Tags the run _appaug.
image_aug="${IMAGE_AUG:-False}"

# EMA of the trainable weights: every checkpoint also writes a *_ema.* sibling with an
# identical key set, so eval can A/B averaged vs. live weights by path alone.
ema_enabled="${EMA:-True}"
ema_decay="${EMA_DECAY:-0.999}"
ema_device="${EMA_DEVICE:-cuda}"

# Held-out action-MSE monitor: holds out N whole episodes PER TASK and scores them every
# EVAL_INTERVAL steps. 0 = disabled.
eval_holdout_episodes="${EVAL_HOLDOUT_EPISODES:-0}"
eval_interval="${EVAL_INTERVAL:-250}"
eval_batch_size="${EVAL_BATCH_SIZE:-32}"

base_lr="${BASE_LR:-1e-4}"
action_lr="${ACTION_LR:-${base_lr}}"
num_warmup_steps="${NUM_WARMUP_STEPS:-1000}"
# Cosine horizon; 100000 is a near-constant anneal over a 20k from-scratch run.
scheduler_total_steps="${SCHEDULER_TOTAL_STEPS:-100000}"

pretrained_checkpoint="${PRETRAINED_CHECKPOINT-}"
is_resume="${IS_RESUME:-False}"
extra_args=()
if [ "${is_resume}" = "True" ] || [ "${is_resume}" = "true" ]; then
  extra_args+=(--trainer.is_resume true)
fi
if [ -n "${pretrained_checkpoint}" ]; then
  extra_args+=(--trainer.pretrained_checkpoint "${pretrained_checkpoint}")
fi

# RESIDENT_TEXT_TABLE=True serves UMT5 embeds from a GPU-resident per-prompt table
# eager-prefilled from the on-disk cache: bit-exact (UMT5 is frozen), UMT5-XXL never loaded
# (~11 GB freed), and the live per-step text encode disappears. False = fully live.
resident_text_table="${RESIDENT_TEXT_TABLE:-True}"
t5_cache_model_id=Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
if [ "${resident_text_table}" = "True" ] || [ "${resident_text_table}" = "true" ]; then
  skip_text_encoder=True; t5_cache_enabled=True
else
  skip_text_encoder=False; t5_cache_enabled=False
fi

# === Run id ===
case "${base_wm}" in
  Skywork/SkyReels-V2-DF-1.3B*) wm_tag=stock ;;
  *) wm_tag=$(basename "${base_wm}") ;;
esac
run_id_base=libero_4in1_CotrainBaseline_${wm_tag}_w${future_frame_idx}
[ "${data_mix}" = "libero_all" ] || run_id_base="${run_id_base}_${data_mix}"
case "${image_aug}" in True|true) run_id_base="${run_id_base}_appaug" ;; esac
case "${ema_enabled}" in True|true) run_id_base="${run_id_base}_ema" ;; esac
run_id="${RUN_ID:-${run_id_base}}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLANCEWAM_TORCHCODEC_CACHE_MAX=400
NUM_PROCESSES=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')

accelerate launch \
  --config_file glancewam/config/deepspeed/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port 29528 \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  glancewam/training/train.py \
  --config_yaml ${config_yaml} \
  --trainer.learning_rate.base ${base_lr} \
  --trainer.learning_rate.action_model ${action_lr} \
  --trainer.num_warmup_steps ${num_warmup_steps} \
  --framework.world_model.base_wm ${base_wm} \
  --framework.world_model.extract_layers "${extract_layers}" \
  --framework.world_model.vae_input_size "${vae_input_size}" \
  --framework.world_model.camera_concat side_by_side \
  --framework.world_model.skip_text_encoder ${skip_text_encoder} \
  --framework.world_model.skip_vae False \
  --framework.world_model.resident_text_table ${resident_text_table} \
  --framework.world_model.truncate_at_extract False \
  --framework.world_model.compile_dit ${compile_dit} \
  --framework.world_model.compile_vae ${compile_vae} \
  --framework.world_model.num_history_frames ${num_history_frames} \
  --framework.video_cotrain.enabled True \
  --framework.video_cotrain.lambda ${cotrain_lambda} \
  --framework.video_cotrain.sigma_sampling ${sigma_sampling} \
  --framework.video_cotrain.clean_prefix_isolation ${clean_prefix_isolation} \
  --framework.qwenvl.base_vlm ${base_wm} \
  --framework.action_model.action_horizon ${action_horizon} \
  --framework.action_model.repeated_diffusion_steps ${repeated_diffusion_steps} \
  --datasets.vla_data.t5_cache.enabled ${t5_cache_enabled} \
  --datasets.vla_data.t5_cache.model_id ${t5_cache_model_id} \
  --datasets.vla_data.t5_cache.attach_embeds False \
  --datasets.vla_data.future_frame_idx ${future_frame_idx} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --datasets.vla_data.num_workers 16 \
  --datasets.vla_data.image_aug ${image_aug} \
  --datasets.vla_data.eval_holdout_episodes ${eval_holdout_episodes} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project GlanceWAM_LIBERO \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.scheduler_total_steps ${scheduler_total_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.eval_interval ${eval_interval} \
  --trainer.eval_batch_size ${eval_batch_size} \
  --trainer.sdpa_backend cudnn \
  --trainer.gc_interval 500 \
  --trainer.ema.enabled ${ema_enabled} \
  --trainer.ema.decay ${ema_decay} \
  --trainer.ema.device ${ema_device} \
  --is_debug False \
  "${extra_args[@]}"
