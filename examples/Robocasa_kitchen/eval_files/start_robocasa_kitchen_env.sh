#!/bin/bash
# Fan out K RoboCasa-kitchen rollout-client shards for ONE task against ONE batched policy server.
#
# The server
# (started separately, with --max-batch >= total shards) coalesces the per-step inference requests
# from all concurrent clients into one batched forward. Each client runs a disjoint, strided slice
# of this task's episode indices via simulation_env.py's --args.num-shards / --args.shard-idx and
# writes its own JSON results. Per-episode horizon is task-driven (TASK_MAX_STEPS in simulation_env).
#
# Usage:
#   bash start_robocasa_kitchen_env.sh <ckpt_path> <task_name> <port> [num_shards]
#     task_name: bare RoboCasa task (e.g. TurnOffMicrowave) or robocasa/<Task>
#
# Env overrides:
#   ROBOCASA_python   python of the kitchen sim venv (py3.10)  [default ../robocasa-cosmos-policy/.venv]
#   host              server host (default 127.0.0.1)
#   n_episodes        rollouts per task (default 50)
#   n_action_steps    open-loop action chunk per policy query (default 16)
#   seed              base seed (default 195)
#   unnorm_key        dataset-statistics key for action un-normalization (required for multi-dataset ckpts)
#   save_video        1 to dump per-episode mp4s (default 0)
#   results_dir / log_dir / video_out_path   output locations (defaults derived from ckpt)
#   SPAWN_SLEEP       seconds between client spawns to stagger EGL/MuJoCo init (default 4)
set -u

ckpt_path=$1
task_name=$2
port=$3
num_shards=${4:-1}

###########################################################################################
# === Environment (override via env vars to match your cluster) ===
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export ROBOCASA_python=${ROBOCASA_python:-${REPO_ROOT}/../robocasa-cosmos-policy/.venv/bin/python}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# MuJoCo offscreen rendering. CUDA_VISIBLE_DEVICES does NOT steer the EGL renderer, and the EGL
# device order != nvidia-smi order, so to co-locate rendering on the SAME physical GPU as the
# server we translate the target nvidia-smi index -> EGL index via egl_device_map.py.
#
# The client does NO CUDA compute (the policy runs server-side), so we set MUJOCO_EGL_DEVICE_ID
# directly and UNSET CUDA_VISIBLE_DEVICES for the client — that avoids robosuite's import-time
# "MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES" assertion that a permuted EGL index would fail.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
# Comma-separated nvidia-smi indices whose graphics engine is bad; empty = all GPUs render fine
# (the default — verified on this node). Set e.g. BAD_EGL_RENDER_GPUS=0,3 on a host where some
# GPU's EGL context is wedged, and renders targeting it move to EGL_RENDER_FALLBACK_GPU.
BAD_EGL_RENDER_GPUS=${BAD_EGL_RENDER_GPUS:-}
EGL_RENDER_FALLBACK_GPU=${EGL_RENDER_FALLBACK_GPU:-2} # healthy nvidia-smi GPU to render on instead
target_gpu="${CUDA_VISIBLE_DEVICES%%,*}"   # first GPU in the caller's mask = the server's GPU
if [ -z "${MUJOCO_EGL_DEVICE_ID:-}" ] && [ -n "${target_gpu}" ]; then
  render_gpu="${target_gpu}"
  for bad in ${BAD_EGL_RENDER_GPUS//,/ }; do
    if [ "${target_gpu}" = "${bad}" ]; then
      render_gpu="${EGL_RENDER_FALLBACK_GPU}"
      echo "⚠️  nvidia-smi GPU ${target_gpu} has a bad EGL graphics engine; redirecting render -> GPU ${render_gpu}"
      break
    fi
  done
  # Probe runs unmasked (EGL enumeration + nvidia-smi, not CUDA) for true physical indices.
  egl_id=$(CUDA_VISIBLE_DEVICES= ${ROBOCASA_python} tools/egl_device_map.py "${render_gpu}" 2>/dev/null)
  export MUJOCO_EGL_DEVICE_ID="${egl_id:-0}"
fi
# Client does no CUDA compute; unmask so the EGL-index assertion universe is unrestricted.
unset CUDA_VISIBLE_DEVICES
echo "🖥️  MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-<unset>} (render GPU = nvidia-smi ${render_gpu:-$target_gpu})"

host=${host:-127.0.0.1}
n_episodes=${n_episodes:-50}
n_action_steps=${n_action_steps:-16}
seed=${seed:-195}
unnorm_key=${unnorm_key:-}
save_video=${save_video:-0}
# include_state=1: client sends the raw 9-d proprio (state-trained checkpoints ONLY —
# a stateless checkpoint fed state, or vice versa, silently mismatches training).
include_state=${include_state:-0}
SPAWN_SLEEP=${SPAWN_SLEEP:-4}
###########################################################################################

# Output layout (parallels the tabletop launcher).
model_root=$(echo "$ckpt_path" | awk -F'/checkpoints/' '{print $1}')
task_short=$(echo "$task_name" | sed -E 's#.*/##')

video_out_path=${video_out_path:-"${model_root}/videos/${task_short}"}
log_dir=${log_dir:-"${model_root}/logs/${task_short}"}
results_dir=${results_dir:-"${model_root}/results/${task_short}"}
mkdir -p "$video_out_path" "$log_dir" "$results_dir"

echo "=========================================="
echo " start_robocasa_kitchen_env: ${num_shards} client shard(s) → ${host}:${port}"
echo "   ckpt   : ${ckpt_path}"
echo "   task   : ${task_short}  (episodes=${n_episodes}, chunk=${n_action_steps}, unnorm_key=${unnorm_key:-<auto>})"
echo "   results: ${results_dir}"
echo "=========================================="

pids=()
for ((shard_idx=0; shard_idx<num_shards; shard_idx++)); do
    shard_log="${log_dir}/shard_${shard_idx}_of_${num_shards}.log"
    shard_json="${results_dir}/shard_${shard_idx}_of_${num_shards}.json"
    echo "▶️ shard ${shard_idx}/${num_shards} → ${shard_log}"

    extra=()
    [ -n "${unnorm_key}" ] && extra+=(--args.unnorm-key "${unnorm_key}")
    [ "${save_video}" = "1" ] && extra+=(--args.save-video)
    [ "${include_state}" = "1" ] && extra+=(--args.include-state)

    ${ROBOCASA_python} ./examples/Robocasa_kitchen/eval_files/simulation_env.py \
        --args.pretrained-path "${ckpt_path}" \
        --args.host "${host}" \
        --args.port "${port}" \
        --args.task-name "${task_name}" \
        --args.n-episodes "${n_episodes}" \
        --args.n-action-steps "${n_action_steps}" \
        --args.seed "${seed}" \
        --args.num-shards "${num_shards}" \
        --args.shard-idx "${shard_idx}" \
        --args.results-out "${shard_json}" \
        --args.video-out-path "${video_out_path}" \
        "${extra[@]}" \
        > "${shard_log}" 2>&1 &
    pids+=($!)

    sleep "${SPAWN_SLEEP}"
done

echo "--- ${num_shards} shard(s) launched for ${task_short}. Waiting... ---"
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done

echo "--- All shards finished for task ${task_short} (rc=${rc}) ---"
exit $rc
