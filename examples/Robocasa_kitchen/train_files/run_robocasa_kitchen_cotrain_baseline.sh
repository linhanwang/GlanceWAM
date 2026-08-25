#!/usr/bin/env bash
# SkyReelsV2-GR00T VIDEO + ACTION CO-TRAINING launcher — RoboCasa KITCHEN (cosmos-policy 24 tasks).
#
# and examples/LIBERO/train_files/run_libero_cotrain_baseline.sh. One forward pass jointly
# trains the GR00T action head (clean obs) AND the SkyReels DF video objective (noised future frames);
# both losses backprop into the unfrozen DiT θ. See
# config_cotrain_baseline_robocasa_kitchen.yaml and wam/CotrainBaseline.py.
#
# Key kitchen specifics:
#   - 3 cameras packed by camera_concat=side_by_side (default): full-res primary | secondary | wrist
#     -> one 672×224 frame, so VAE_INPUT_SIZE=[224,672]. (primary_inset = legacy 336×224 layout.)
#   - Data: 24 per-task lerobot v3 datasets under $DATA_ROOT/robocasa_cosmos_kitchen
#     (success column → reader drops failed demos for the policy loss).
#   - STATELESS by default; INCLUDE_STATE=True feeds the raw 9-dim cosmos proprio (eval must
#     then pass --include-state). IMAGE_AUG=True = appearance-only aug; T_LOSS_WEIGHTING=True =
#     flow-matching timestep loss weighting.
#   - RESIDENT_TEXT_TABLE defaults TRUE (GPU-resident UMT5 table): bit-exact, frees ~11 GB,
#     ~24% faster step. REQUIRES the UMT5 precompute over the kitchen datasets first
#     (bash examples/Robocasa_kitchen/train_files/run_precompute_umt5_skyreels.sh). Set
#     RESIDENT_TEXT_TABLE=False for the fully-live UMT5 path (no precompute needed).
#   - Held-out action-MSE monitor live-decodes (no kitchen VAE cache exists); EVAL_VAE_CACHE=False.
#
# This is the FROM-SCRATCH cotrain PRETRAINING entry: stock demos, 60k steps on the
# config's 100k cosine (1e-4 peak, 5k warmup), 4 GPUs x bsz 16 x grad-accum 2 = global
# batch 128. The offline-RL finetune legs are separate standalone launchers:
#   run_robocasa_kitchen_filteredbc_ft.sh   rung-1 filtered BC (stock + successful rollouts)
# Every knob below is env-overridable.
#
# 2026-07-17 (DEFAULT NOW): Hv=1.6 s / Hp=0.8 s dose-response point — video window
# future_frame_idx=32 (1.6 s) DECOUPLED from action_horizon=16 (0.8 s). Fills the
# missing cell in the horizon study (w32 confounds a 1.6 s video horizon
# with a doubled 1.6 s action chunk; this isolates the video horizon at the champion's Hp=16).
# Track B step 1 of §7. run_id tagged _w32_ah16 so it never collides with the aligned w32 (Hp=32)
# run. Everything else = the w16 recipe (blre4/fflast/sbs/appaug/state/tweight, EMA). Zero code
# change; the dataloader's Hc<Hv span-mismatch warning is EXPECTED (the 0715 run already tolerates
# it). Rollout-ft the 20k EMA afterward — §1.4 point 3: ft amplifies horizon differences.

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
config_yaml=./examples/Robocasa_kitchen/train_files/config_cotrain_baseline_robocasa_kitchen.yaml
data_root_dir="${DATA_ROOT:-results/Datasets}/robocasa_cosmos_kitchen"
run_root_dir="${RUN_ROOT_DIR:-./results/Checkpoints}"

# === Run knobs (override the config) ===
# Stock SkyReels-V2 DF-1.3B init (co-training learns kitchen dynamics jointly with the action head).
# "skyreels"/"SkyReels" must stay in the path so get_world_model's substring dispatch routes correctly.
base_wm="${BASE_WM:-Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers}"

data_mix="${DATA_MIX:-robocasa_kitchen_all}"
max_train_steps="${MAX_TRAIN_STEPS:-40000}"
# Held-out action-MSE monitor: holds out EVAL_HOLDOUT_EPISODES whole episodes PER TASK and scores
# them every EVAL_INTERVAL steps. Set EVAL_HOLDOUT_EPISODES=0 to disable.
eval_holdout_episodes="${EVAL_HOLDOUT_EPISODES:-0}"
eval_interval="${EVAL_INTERVAL:-500}"
eval_batch_size="${EVAL_BATCH_SIZE:-64}"
# No precomputed kitchen VAE cache → the held-out monitor live-decodes+encodes the obs frame.
# Action-head chunk length Hp, in NATIVE rows (RoboCasa is 20 Hz, so 16 = 0.8 s, 32 = 1.6 s).
# cosmos-policy predicts 32 and executes 16 in a receding horizon
#; eval needs no change since the kitchen
# sweep already defaults to --n-action-steps 16. Raising it grows the head's chunk and the
# zero-init action-cond embedder Linear(ah*action_dim, inner).
# DEFAULT: Hp=16 (0.8 s), DECOUPLED from the video horizon below — the missing dose-response
# point (Hv=1.6 s at Hp=16) of the horizon study Set ACTION_HORIZON
# equal to FUTURE_FRAME_IDX to recover an aligned run (e.g. both 32 = the old w32 recipe).
action_horizon="${ACTION_HORIZON:-16}"
extract_layers="${EXTRACT_LAYERS:-[19]}"
num_history_frames="${NUM_HISTORY_FRAMES:-1}"
# The ONE future frame the DiT predicts (video horizon Hv), at t+future_frame_idx NATIVE rows —
# the SAME units as action_horizon, and independent of it. Runs are tagged w<future_frame_idx>.
# DEFAULT: Hv=32 (1.6 s), longer than Hp=16 above so the video target stays non-trivial while the
# head stays short. The clean token
# spans only Hp (0.8 s) of this interval → under-posed Hc: the dataloader's span-mismatch warning
# is EXPECTED here (same semantics the 0715 run already tolerates, §1.4 point 2).
# A longer window is ~free: encode_frames is num_history_frames+tscale regardless, so tokens,
# memory and step-time do not move with it.
future_frame_idx="${FUTURE_FRAME_IDX:-32}"
# Proprio: feed the raw 9-d
# cosmos-native state [gripper_qpos(2), eef_pos(3), eef_quat(4,xyzw)] to the action head
# (state_dim: 9 is already wired in the YAML). Eval must then pass --include-state to the
# sweep orchestrator so the client sends the matching raw proprio — NEVER mix state-trained
# checkpoints with stateless eval or vice versa.
include_state="${INCLUDE_STATE:-True}"
# Image aug (gap doc, run A): APPEARANCE-ONLY (color jitter + JPEG q95; no crop/rotation).
# Jitter params are shared across the whole obs+future window (one transform call), so the
# video-loss target stays temporally consistent. Geometry-free => no eval-side change.
# The only knob with a real aggregate effect in the ablation (+0.018 @40k).
image_aug="${IMAGE_AUG:-True}"
# Flow-matching timestep loss weighting on the action head (gap doc, tweight row —
# neutral in the ablation; exposed for the cotrain sweep).
t_loss_weighting="${T_LOSS_WEIGHTING:-True}"
cotrain_lambda="${COTRAIN_LAMBDA:-1.0}"
# logitnormal (shift=5) matches the SkyReels video LoRA finetune timestep weighting.
sigma_sampling="${SIGMA_SAMPLING:-logitnormal}"
clean_prefix_isolation="${CLEAN_PREFIX_ISOLATION:-True}"
compile_dit="${COMPILE_DIT:-True}"
compile_vae="${COMPILE_VAE:-True}"
repeated_diffusion_steps="${RDS:-12}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-16}"
# Camera packing (vae_input_size is [H, W]):
#   side_by_side (default): all 3 cameras full-res (primary | secondary | wrist) -> 672×224.
#     Ablation-validated: 0.640/0.644/0.638 @ 20k/40k/60k ema vs primary_inset's
#     0.572/0.573/0.549 — the 112² inset downsample was the SR bottleneck (gains
#     concentrate in small-target tasks). Costs 2x latent tokens/frame; lower
#     PER_DEVICE_BATCH_SIZE if OOM.
#   primary_inset: primary 224² + secondary & wrist 112² insets -> 336×224.
camera_concat="${CAMERA_CONCAT:-side_by_side}"
if [ "${camera_concat}" = "side_by_side" ]; then
  vae_input_size="${VAE_INPUT_SIZE:-[224,672]}"
else
  vae_input_size="${VAE_INPUT_SIZE:-[224,336]}"
fi
# EMA: ON by default; every checkpoint writes a *_ema.* sibling for A/B at eval.
ema_enabled="${EMA:-True}"
ema_decay="${EMA_DECAY:-0.999}"
ema_device="${EMA_DEVICE:-cuda}"
save_interval="${SAVE_INTERVAL:-10000}"
# Text caching. TRUE (default) serves UMT5 embeds from a GPU-resident per-prompt table
# (bit-exact, frees ~11 GB, ~24% faster step) but REQUIRES the UMT5 precompute over the
# kitchen datasets first. FALSE = fully-live UMT5 (no precompute needed).
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
# When the action horizon (Hp) is decoupled from the video window (Hv = w<future_frame_idx>), tag
# it (_ah<Hp>) so the Hv=1.6 s/Hp=0.8 s dose-response run never collides with the aligned w32
# (Hp=32) run's name. Empty when aligned → backward-compatible with existing w16/w32 run_ids.
if [ "${action_horizon}" != "${future_frame_idx}" ]; then ah_tag="_ah${action_horizon}"; else ah_tag=""; fi
run_id_base="robocasa_kitchen_CotrainBaseline_${wm_tag}_w${future_frame_idx}${ah_tag}_blre4"
# Tag the full-res side-by-side camera packing (resolution ablation vs primary_inset).
if [ "${camera_concat}" = "side_by_side" ]; then
  run_id_base="${run_id_base}_sbs"
fi
# Ablation tags (mirroring run_robocasa_kitchen_gr00t_train.sh's naming).
if [ "${image_aug}" = "True" ] || [ "${image_aug}" = "true" ]; then
  run_id_base="${run_id_base}_appaug"
fi
if [ "${include_state}" = "True" ] || [ "${include_state}" = "true" ]; then
  run_id_base="${run_id_base}_state"
fi
if [ "${t_loss_weighting}" = "True" ] || [ "${t_loss_weighting}" = "true" ]; then
  run_id_base="${run_id_base}_tweight"
fi
# Tag non-default task subsets so they don't clobber the full-24-task run.
case "${data_mix}" in
  robocasa_kitchen_all) ;;
  *) run_id_base="${run_id_base}_${data_mix#robocasa_kitchen_}" ;;
esac
if [ "${ema_enabled}" = "True" ] || [ "${ema_enabled}" = "true" ]; then
  run_id_base="${run_id_base}_ema"
fi
run_id="${RUN_ID:-${run_id_base}}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLANCEWAM_TORCHCODEC_CACHE_MAX=400

NUM_PROCESSES=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"

# export WANDB_MODE=disabled

accelerate launch \
  --config_file glancewam/config/deepspeed/deepspeed_zero2.yaml \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port "${MAIN_PROCESS_PORT:-29521}" \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  glancewam/training/train.py \
  --config_yaml ${config_yaml} \
  --trainer.learning_rate.base 1e-4 \
  --framework.world_model.base_wm ${base_wm} \
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
  --framework.world_model.compile_dit ${compile_dit} \
  --framework.world_model.compile_vae ${compile_vae} \
  --framework.world_model.num_history_frames ${num_history_frames} \
  --framework.video_cotrain.enabled True \
  --framework.video_cotrain.lambda ${cotrain_lambda} \
  --datasets.vla_data.future_frame_idx ${future_frame_idx} \
  --framework.video_cotrain.sigma_sampling ${sigma_sampling} \
  --framework.video_cotrain.clean_prefix_isolation ${clean_prefix_isolation} \
  --framework.qwenvl.base_vlm ${base_wm} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
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
  --is_debug False
