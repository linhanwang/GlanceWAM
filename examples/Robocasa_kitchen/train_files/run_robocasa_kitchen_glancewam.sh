#!/usr/bin/env bash
# SkyReelsV2-GR00T **GOAL-IMAGE co-training** launcher — RoboCasa KITCHEN (cosmos-policy 24 tasks).
#
# The low level of the goal-image planner. Clone of
# run_robocasa_kitchen_cotrain_baseline.sh (the plain-cotrain fBC 0.680 champion's
# FROM-SCRATCH pretraining recipe) with ONE addition: the action head conditions on a GOAL
# IMAGE. See config_glancewam_robocasa_kitchen.yaml and
# wam/GlanceWAM.py.
#
# What differs from the plain cotrain launcher:
#   - framework GlanceWAM (goal conditioning)
#     (this framework does not use the rung-12a video-conditioning token).
#   - The window ships a SECOND future frame: the video branch trains at the FIXED horizon
#     FUTURE_FRAME_IDX (H_g, the goal-generation horizon), while the action head conditions on
#     a hindsight GOAL frame drawn per sample at g ~ U(0, H_g] (GOAL_FRAME_MAX = H_g), the
#     distribution it faces mid-cycle at inference (the plan's cadence rule).
#   - DEFAULT H_g = 3 s = 60 native rows (20 Hz). A KNOB (Q1): 4 s = 80. action_horizon stays 16.
#
# Everything else mirrors the champion (logitnormal σ, appearance aug, state, t-loss-weighting,
# EMA, compile_dit/vae, resident UMT5 table). Every knob is env-overridable.

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
config_yaml=./examples/Robocasa_kitchen/train_files/config_glancewam_robocasa_kitchen.yaml
data_root_dir="${DATA_ROOT:-results/Datasets}/robocasa_cosmos_kitchen"
run_root_dir="${RUN_ROOT_DIR:-./results/Checkpoints}"

# === Run knobs (override the config) ===
base_wm="${BASE_WM:-Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers}"
# E3(b) — DF clean-corner A/B. Loads ONLY the DiT weights from
# another repo; the Wan VAE, UMT5, tokenizer and scheduler still come from base_wm, and so does
# every other knob below. TRANSFORMER_FROM=Wan-AI/Wan2.1-T2V-1.3B-Diffusers is the non-DF arm:
# the model SkyReels-V2-DF-1.3B was diffusion-forcing-finetuned from, same architecture, its 825
# transformer tensors an exact subset of SkyReels' 830 (the 5 extra are the fps embedding, off
# for a Wan config). Empty = the DF arm, i.e. the unchanged recipe.
transformer_from="${TRANSFORMER_FROM:-}"
data_mix="${DATA_MIX:-robocasa_kitchen_all}"
# DATA-EFFICIENCY ablation: train on only N demo episodes per task (deterministic,
# seed-stable per-task subset; the eval-holdout split and the dataset normalization stats
# are untouched, so arms stay comparable). 0 = off = the full ~50 successful demos/task.
max_episodes_per_task="${MAX_EPISODES_PER_TASK:-0}"
episode_subsample_seed="${EPISODE_SUBSAMPLE_SEED:-42}"
max_train_steps="${MAX_TRAIN_STEPS:-20000}"
eval_holdout_episodes="${EVAL_HOLDOUT_EPISODES:-0}"
eval_interval="${EVAL_INTERVAL:-500}"
eval_batch_size="${EVAL_BATCH_SIZE:-64}"
action_horizon="${ACTION_HORIZON:-16}"
extract_layers="${EXTRACT_LAYERS:-[19]}"
num_history_frames="${NUM_HISTORY_FRAMES:-1}"
# Video horizon Hv = H_g (the goal-generation horizon), NATIVE rows (20 Hz -> 60 = 3 s). The
# video branch predicts one frame at t+future_frame_idx; the goal head conditions on a frame
# sampled uniformly within it. DEFAULT H_g = 3 s = 60 (Q1; 4 s = 80).
future_frame_idx="${FUTURE_FRAME_IDX:-60}"
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
goal_window_args=(--datasets.vla_data.goal_frame_range "[${goal_frame_min},${goal_frame_max}]")
pretrained_checkpoint="${PRETRAINED_CHECKPOINT:-}"
reload_modules="${RELOAD_MODULES:-}"
freeze_modules="${FREEZE_MODULES:-}"
include_state="${INCLUDE_STATE:-True}"
image_aug="${IMAGE_AUG:-True}"
t_loss_weighting="${T_LOSS_WEIGHTING:-True}"
cotrain_lambda="${COTRAIN_LAMBDA:-1.0}"
sigma_sampling="${SIGMA_SAMPLING:-logitnormal}"
clean_prefix_isolation="${CLEAN_PREFIX_ISOLATION:-True}"
compile_dit="${COMPILE_DIT:-True}"
compile_vae="${COMPILE_VAE:-True}"
# Activation checkpointing on the DiT blocks (~30% slower, frees the per-layer
# activation store). OFF by default = every 1.3B run unchanged. Required for the
# DF-14B backbone: its ZeRO-2 states alone are ~77 GB/GPU at 4 ranks.
grad_ckpt="${GRAD_CKPT:-False}"
repeated_diffusion_steps="${RDS:-12}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-16}"
camera_concat="${CAMERA_CONCAT:-side_by_side}"
if [ "${camera_concat}" = "side_by_side" ]; then
  vae_input_size="${VAE_INPUT_SIZE:-[224,672]}"
else
  vae_input_size="${VAE_INPUT_SIZE:-[224,336]}"
fi
ema_enabled="${EMA:-True}"
ema_decay="${EMA_DECAY:-0.999}"
ema_device="${EMA_DEVICE:-cuda}"
save_interval="${SAVE_INTERVAL:-5000}"
# LR schedule (cosine_with_min_lr). Default is the ANNEAL recipe: cosine completes within the
# 20k-step run (goal-cotrain peaks ~10k = 1.45 epochs under flat LR; the historical 100k-planned
# schedule never decayed below ~0.82x peak LR). Historical flat-LR recipe:
# SCHEDULER_TOTAL_STEPS=100000 NUM_WARMUP_STEPS=5000 MAX_TRAIN_STEPS=40000.
scheduler_total_steps="${SCHEDULER_TOTAL_STEPS:-20000}"
num_warmup_steps="${NUM_WARMUP_STEPS:-1000}"
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
# A swapped DiT is a different backbone, so it must show up in the run id.
if [ -n "${transformer_from}" ]; then wm_tag="dit-$(basename "${transformer_from}")"; fi
if [ "${action_horizon}" != "${future_frame_idx}" ]; then ah_tag="_ah${action_horizon}"; else ah_tag=""; fi
if [ "${scheduler_total_steps}" != "100000" ] || [ "${num_warmup_steps}" != "5000" ]; then
  sched_tag="_sched${scheduler_total_steps}wu${num_warmup_steps}"
else
  sched_tag=""
fi
run_id_base="robocasa_kitchen_GlanceWAM_${wm_tag}_w${future_frame_idx}${ah_tag}_g${goal_frame_max}_blre4${sched_tag}"
# Bidirectional-attention ablation (no 3-class mask) must never clobber a masked run.
if [ "${clean_prefix_isolation}" != "True" ] && [ "${clean_prefix_isolation}" != "true" ]; then
  run_id_base="${run_id_base}_bidir"
fi
if [ "${camera_concat}" = "side_by_side" ]; then run_id_base="${run_id_base}_sbs"; fi
if [ "${image_aug}" = "True" ] || [ "${image_aug}" = "true" ]; then run_id_base="${run_id_base}_appaug"; fi
if [ "${include_state}" = "True" ] || [ "${include_state}" = "true" ]; then run_id_base="${run_id_base}_state"; fi
if [ "${t_loss_weighting}" = "True" ] || [ "${t_loss_weighting}" = "true" ]; then run_id_base="${run_id_base}_tweight"; fi
case "${data_mix}" in
  robocasa_kitchen_all) ;;
  *) run_id_base="${run_id_base}_${data_mix#robocasa_kitchen_}" ;;
esac
# The half-data arm must never clobber the full-data row.
if [ "${max_episodes_per_task}" != "0" ]; then
  run_id_base="${run_id_base}_ep${max_episodes_per_task}"
  if [ "${episode_subsample_seed}" != "42" ]; then run_id_base="${run_id_base}s${episode_subsample_seed}"; fi
fi
if [ "${ema_enabled}" = "True" ] || [ "${ema_enabled}" = "true" ]; then run_id_base="${run_id_base}_ema"; fi
# E7.0 multi-tap goal conditioning: tag by tap count so a mtap run never clobbers the
# single-tap baseline (historical mtap4
# rows pinned RUN_ID by hand because this tag did not exist yet). No-op at one tap.
n_taps=$(echo "${extract_layers}" | tr -cd ',' | wc -c); n_taps=$((n_taps + 1))
if [ "${n_taps}" -gt 1 ]; then run_id_base="${run_id_base}_mtap${n_taps}"; fi
run_id="${RUN_ID:-${run_id_base}}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLANCEWAM_TORCHCODEC_CACHE_MAX=400

NUM_PROCESSES=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"

# Two-stage passthroughs — only appended when set, so the default recipe is untouched.
extra_args=()
[ -n "${pretrained_checkpoint}" ] && extra_args+=(--trainer.pretrained_checkpoint "${pretrained_checkpoint}")
[ -n "${reload_modules}" ] && extra_args+=(--trainer.reload_modules "${reload_modules}")
[ -n "${freeze_modules}" ] && extra_args+=(--trainer.freeze_modules "${freeze_modules}")

accelerate launch \
  --config_file glancewam/config/deepspeed/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port "${MAIN_PROCESS_PORT:-29522}" \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  glancewam/training/train.py \
  --config_yaml ${config_yaml} \
  --trainer.learning_rate.base 1e-4 \
  --trainer.scheduler_total_steps ${scheduler_total_steps} \
  --trainer.num_warmup_steps ${num_warmup_steps} \
  --framework.world_model.base_wm ${base_wm} \
  --framework.world_model.transformer_from "${transformer_from}" \
  --framework.world_model.extract_layers "${extract_layers}" \
  --framework.world_model.vae_input_size "${vae_input_size}" \
  --framework.world_model.camera_concat ${camera_concat} \
  --framework.world_model.skip_text_encoder ${skip_text_encoder} \
  --framework.world_model.skip_vae False \
  --framework.world_model.resident_text_table ${resident_text_table} \
  --datasets.vla_data.t5_cache.enabled ${t5_cache_enabled} \
  --datasets.vla_data.t5_cache.model_id ${t5_cache_model_id} \
  --datasets.vla_data.t5_cache.attach_embeds False \
  --framework.world_model.truncate_at_extract False \
  --framework.world_model.enable_gradient_checkpointing ${grad_ckpt} \
  --framework.world_model.compile_dit ${compile_dit} \
  --framework.world_model.compile_vae ${compile_vae} \
  --framework.world_model.num_history_frames ${num_history_frames} \
  --framework.video_cotrain.enabled True \
  --framework.video_cotrain.lambda ${cotrain_lambda} \
  --datasets.vla_data.future_frame_idx ${future_frame_idx} \
  "${goal_window_args[@]}" \
  --framework.video_cotrain.sigma_sampling ${sigma_sampling} \
  --framework.video_cotrain.clean_prefix_isolation ${clean_prefix_isolation} \
  --framework.goal_conditioning.enabled True \
  --framework.goal_conditioning.dropout_p ${goal_dropout_p} \
  --framework.goal_conditioning.goal_gen_steps ${goal_gen_steps} \
  --framework.goal_conditioning.h_embed_dim ${goal_h_embed_dim} \
  --framework.qwenvl.base_vlm ${base_wm} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.max_episodes_per_dataset ${max_episodes_per_task} \
  --datasets.vla_data.episode_subsample_seed ${episode_subsample_seed} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --datasets.vla_data.num_workers 16 \
  --datasets.vla_data.image_aug ${image_aug} \
  --datasets.vla_data.include_state ${include_state} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project GlanceWAM_Robocasa_kitchen \
  --trainer.max_train_steps ${max_train_steps} \
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
  --framework.action_model.t_loss_weighting ${t_loss_weighting} \
  "${extra_args[@]}" \
  --is_debug False
