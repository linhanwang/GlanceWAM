#!/usr/bin/env bash
# SkyReelsV2-GR00T **GOAL-IMAGE co-training** launcher — LIBERO 4-in-1.
#
# The low level of the goal-image planner — LIBERO
# sibling of examples/Robocasa_kitchen/train_files/run_robocasa_kitchen_glancewam.sh.
# Clone of the 0628 LIBERO cotrain champion's FROM-SCRATCH recipe (stock SkyReels init,
# libero_all demos, stateless, no appearance aug, 0.4 s chunks) with ONE addition: the action
# head conditions on a GOAL IMAGE. See config_glancewam_libero.yaml and
# wam/GlanceWAM.py.
#
# What differs from the plain LIBERO cotrain launcher (run_libero_cotrain_baseline.sh):
#   - framework GlanceWAM (goal conditioning) + goal_frame_range dataloader knob.
#   - FROM-SCRATCH defaults (the plain launcher currently defaults to the rung-1 finetune leg):
#     no pretrained_checkpoint, base/action LR 1e-4, 100k-step cosine, warmup 5000.
#   - The window ships a SECOND future frame: the video branch trains at the FIXED horizon
#     FUTURE_FRAME_IDX (H_g, the goal-generation horizon), while the action head conditions on
#     a hindsight GOAL frame drawn per sample at g ~ U(0, H_g] (GOAL_FRAME_MAX = H_g).
#   - DEFAULT H_g = 2.4 s = 48 native rows (20 Hz), the plan's minimum goal horizon — LIBERO
#     episodes are short (median 105 frames), so the kitchen's 3 s/60 would clamp the video
#     target to the episode end too often. A KNOB (Q1). action_horizon stays 8.
#
# NOTE eval: the policy server auto-advertises H_g; the LIBERO eval client (model2libero_interface)
# runs the stateful goal-refresh cadence when it sees it. No-goal control: GLANCEWAM_GOAL_DROP=1
# in the SERVER environment.
#
# REQUIRES the UMT5 text cache over libero_all (RESIDENT_TEXT_TABLE=True; set False to skip):
#   bash examples/LIBERO/train_files/run_precompute_umt5_skyreels.sh

# Reference hardware: 4xH200. 4 GPUs x per-device bsz 16 x grad-accum 2 = global batch 128.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Force torch's bundled cuBLAS/cuBLASLt to win over any system CUDA toolkit on LD_LIBRARY_PATH.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
VENV_NV_LIB="${REPO_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cu12/lib"
export LD_LIBRARY_PATH="${VENV_NV_LIB}:${LD_LIBRARY_PATH:-}"

# torchcodec needs ffmpeg 6 .so files; redirect to PyAV's wheel-bundled libs via the shim.
FFMPEG6_SHIM="${REPO_ROOT}/.venv/lib/ffmpeg6_shim"
AV_LIBS="${REPO_ROOT}/.venv/lib/python3.11/site-packages/av.libs"
[ -d "${FFMPEG6_SHIM}" ] || python "${REPO_ROOT}/tools/setup_ffmpeg6_shim.py"
export LD_LIBRARY_PATH="${FFMPEG6_SHIM}:${AV_LIBS}:${LD_LIBRARY_PATH}"

# === Paths ===
config_yaml=./examples/LIBERO/train_files/config_glancewam_libero.yaml
data_root_dir="${DATA_ROOT:-results/Datasets}"
run_root_dir="${RUN_ROOT_DIR:-./results/Checkpoints}"

# === Run knobs (override the config) ===
base_wm="${BASE_WM:-Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers}"
data_mix="${DATA_MIX:-libero_all}"
max_train_steps="${MAX_TRAIN_STEPS:-20000}"
# Continue an existing run: IS_RESUME=True picks up the latest checkpoint in this run_id's
# checkpoints/ dir (loads live + EMA weights, sets completed_steps, advances the LR cosine).
# Pin RUN_ID to the existing run so _get_latest_checkpoint finds it; optimizer state is not
# saved (warm restart). Extend to N by raising MAX_TRAIN_STEPS.
is_resume="${IS_RESUME:-False}"
eval_holdout_episodes="${EVAL_HOLDOUT_EPISODES:-0}"
eval_interval="${EVAL_INTERVAL:-500}"
eval_batch_size="${EVAL_BATCH_SIZE:-32}"
action_horizon="${ACTION_HORIZON:-8}"
extract_layers="${EXTRACT_LAYERS:-[19]}"
num_history_frames="${NUM_HISTORY_FRAMES:-1}"
# Video horizon Hv = H_g (the goal-generation horizon), NATIVE rows (20 Hz -> 48 = 2.4 s).
# The video branch predicts one frame at t+future_frame_idx; the goal head conditions on a
# frame sampled uniformly within it. DEFAULT H_g = 2.4 s = 48 (Q1; kitchen used 3 s = 60 but
# LIBERO episodes are shorter — see the config header).
future_frame_idx="${FUTURE_FRAME_IDX:-48}"
# Goal offset range g ~ U{GOAL_FRAME_MIN..GOAL_FRAME_MAX} native rows = the (0, H_g] cadence-rule
# distribution. Max defaults to H_g so it never exceeds the video-target horizon.
goal_frame_min="${GOAL_FRAME_MIN:-1}"
goal_frame_max="${GOAL_FRAME_MAX:-${future_frame_idx}}"
# Per-sample goal-token dropout (keeps a no-goal fallback in-distribution for the eval control arm).
goal_dropout_p="${GOAL_DROPOUT_P:-0.1}"
# Video-DiT sampler steps for generating the goal latent at inference (eval-only).
goal_gen_steps="${GOAL_GEN_STEPS:-10}"
# Sinusoidal width for the goal-time diff embed (g = rows between goal and obs); 0 = off.
goal_h_embed_dim="${GOAL_H_EMBED_DIM:-16}"
image_aug="${IMAGE_AUG:-False}"
cotrain_lambda="${COTRAIN_LAMBDA:-1.0}"
sigma_sampling="${SIGMA_SAMPLING:-logitnormal}"
clean_prefix_isolation="${CLEAN_PREFIX_ISOLATION:-True}"
compile_dit="${COMPILE_DIT:-True}"
compile_vae="${COMPILE_VAE:-True}"
repeated_diffusion_steps="${RDS:-8}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-16}"
# 224x448 = two 224x224 cameras stitched side-by-side (primary | wrist).
vae_input_size="${VAE_INPUT_SIZE:-[224,448]}"
base_lr="${BASE_LR:-1e-4}"
action_lr="${ACTION_LR:-1e-4}"
scheduler_total_steps="${SCHEDULER_TOTAL_STEPS:-100000}"
num_warmup_steps="${NUM_WARMUP_STEPS:-5000}"
ema_enabled="${EMA:-True}"
ema_decay="${EMA_DECAY:-0.999}"
ema_device="${EMA_DEVICE:-cuda}"
save_interval="${SAVE_INTERVAL:-5000}"
resident_text_table="${RESIDENT_TEXT_TABLE:-True}"
t5_cache_model_id=Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
if [ "${resident_text_table}" = "True" ] || [ "${resident_text_table}" = "true" ]; then
  skip_text_encoder=True; t5_cache_enabled=True
else
  skip_text_encoder=False; t5_cache_enabled=False
fi

# Tag by backbone init + window so runs never clobber each other.
case "${base_wm}" in
  Skywork/SkyReels-V2-DF-1.3B*) wm_tag=stock ;;
  *) wm_tag=$(basename "${base_wm}") ;;
esac
run_id_base="libero_4in1_GlanceWAM_${wm_tag}_w${future_frame_idx}_ah${action_horizon}_g${goal_frame_max}_blre4"
# Only tag a non-default cosine (mirrors the kitchen launcher), so baseline run_ids are unchanged.
if [ "${scheduler_total_steps}" != "100000" ] || [ "${num_warmup_steps}" != "5000" ]; then
  run_id_base="${run_id_base}_sched${scheduler_total_steps}wu${num_warmup_steps}"
fi
if [ "${image_aug}" = "True" ] || [ "${image_aug}" = "true" ]; then run_id_base="${run_id_base}_appaug"; fi
case "${data_mix}" in
  libero_all) ;;
  *) run_id_base="${run_id_base}_${data_mix#libero_}" ;;
esac
if [ "${ema_enabled}" = "True" ] || [ "${ema_enabled}" = "true" ]; then run_id_base="${run_id_base}_ema"; fi
run_id="${RUN_ID:-${run_id_base}}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLANCEWAM_TORCHCODEC_CACHE_MAX=400

NUM_PROCESSES=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"

accelerate launch \
  --config_file glancewam/config/deepspeed/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port "${MAIN_PROCESS_PORT:-29529}" \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  glancewam/training/train.py \
  --config_yaml ${config_yaml} \
  --trainer.learning_rate.base ${base_lr} \
  --trainer.learning_rate.action_model ${action_lr} \
  --trainer.scheduler_total_steps ${scheduler_total_steps} \
  --trainer.num_warmup_steps ${num_warmup_steps} \
  --framework.world_model.base_wm ${base_wm} \
  --framework.world_model.extract_layers "${extract_layers}" \
  --framework.world_model.vae_input_size "${vae_input_size}" \
  --framework.world_model.camera_concat side_by_side \
  --framework.world_model.skip_text_encoder ${skip_text_encoder} \
  --framework.world_model.skip_vae False \
  --framework.world_model.resident_text_table ${resident_text_table} \
  --datasets.vla_data.t5_cache.enabled ${t5_cache_enabled} \
  --datasets.vla_data.t5_cache.model_id ${t5_cache_model_id} \
  --datasets.vla_data.t5_cache.attach_embeds False \
  --framework.world_model.truncate_at_extract False \
  --framework.world_model.compile_dit ${compile_dit} \
  --framework.world_model.compile_vae ${compile_vae} \
  --framework.world_model.num_history_frames ${num_history_frames} \
  --framework.video_cotrain.enabled True \
  --framework.video_cotrain.lambda ${cotrain_lambda} \
  --datasets.vla_data.future_frame_idx ${future_frame_idx} \
  --datasets.vla_data.goal_frame_range "[${goal_frame_min},${goal_frame_max}]" \
  --framework.video_cotrain.sigma_sampling ${sigma_sampling} \
  --framework.video_cotrain.clean_prefix_isolation ${clean_prefix_isolation} \
  --framework.goal_conditioning.enabled True \
  --framework.goal_conditioning.dropout_p ${goal_dropout_p} \
  --framework.goal_conditioning.goal_gen_steps ${goal_gen_steps} \
  --framework.goal_conditioning.h_embed_dim ${goal_h_embed_dim} \
  --framework.qwenvl.base_vlm ${base_wm} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --datasets.vla_data.num_workers 16 \
  --datasets.vla_data.image_aug ${image_aug} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project GlanceWAM_LIBERO \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.is_resume ${is_resume} \
  --trainer.save_interval ${save_interval} \
  --datasets.vla_data.eval_holdout_episodes ${eval_holdout_episodes} \
  --trainer.eval_interval ${eval_interval} \
  --trainer.eval_batch_size ${eval_batch_size} \
  --trainer.sdpa_backend cudnn \
  --trainer.step_timing "${STEP_TIMING:-False}" \
  --trainer.gc_interval 500 \
  --trainer.ema.enabled ${ema_enabled} \
  --trainer.ema.decay ${ema_decay} \
  --trainer.ema.device ${ema_device} \
  --framework.action_model.action_horizon ${action_horizon} \
  --framework.action_model.repeated_diffusion_steps ${repeated_diffusion_steps} \
  --is_debug False
