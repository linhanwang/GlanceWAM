#!/bin/bash
# Fan out K LIBERO rollout-client shards against ONE batched policy server.
#
# the server (started separately, with --max-batch >= NUM_SHARDS) coalesces the
# per-step inference requests from all K clients into one batched forward pass, so
# throughput scales ~K× until the GPU saturates. Each client runs a disjoint,
# strided slice of the suite's (task_id, episode_idx) work via eval_libero.py's
# --args.num-shards / --args.shard-idx, and writes its own JSON results file.
#
# Usage:
#   bash start_libero_env.sh <ckpt_path> <task_suite_name> <port> [num_shards]
#
# Env overrides:
#   LIBERO_HOME            root of the LIBERO repo (for PYTHONPATH + configs)
#   LIBERO_python          python interpreter of the libero client venv (py3.10)
#   host                   server host (default 127.0.0.1)
#   num_trials_per_task    rollouts per task (default 50)
#   results_dir / log_dir / video_out_path   output locations (defaults derived from ckpt)
#   SPAWN_SLEEP            seconds between client spawns to stagger EGL/MuJoCo init (default 4)
set -u

ckpt_path=$1
task_suite_name=$2
port=$3
num_shards=${4:-8}

###########################################################################################
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# === Environment (override via env vars to match your cluster) ===
# LIBERO is installed as a SIBLING of this repo by install_libero.sh; override either var
# if you keep it elsewhere.
export LIBERO_HOME=${LIBERO_HOME:-${REPO_ROOT}/../LIBERO}
export LIBERO_python=${LIBERO_python:-${REPO_ROOT}/../LIBERO/.venv/bin/python}
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH="$(pwd):${LIBERO_HOME}:${PYTHONPATH:-}"

# MuJoCo offscreen rendering. CUDA_VISIBLE_DEVICES does NOT steer the EGL renderer, and the
# EGL device order != nvidia-smi order, so to co-locate rendering on the SAME physical GPU as
# the server we translate the target nvidia-smi index -> EGL index via egl_device_map.py.
#
# We export it as GLANCEWAM_EGL_DEVICE_ID, NOT MUJOCO_EGL_DEVICE_ID: robosuite's binding_utils
# asserts at import that MUJOCO_EGL_DEVICE_ID is a substring of CUDA_VISIBLE_DEVICES, which the
# permuted EGL index (e.g. 0 for GPU 3) would fail. eval_libero.py promotes GLANCEWAM_EGL_DEVICE_ID
# -> MUJOCO_EGL_DEVICE_ID after that import but before any render. Override by exporting either
# var yourself (the probe is skipped).
#
# Bad-graphics-engine deny-list: some GPUs have a wedged GRAPHICS engine (compute is fine) that
# MMU-faults under concurrent EGL contexts (Xid 31 -> Xid 109 ctx-switch-timeout storm), hanging
# every sim client. Confirmed on nvidia-smi GPU 3 (PCI 0000:64:00) on this box, 2026-05-30. The
# fault is GRAPHICS-only, so we keep the server's COMPUTE on the target GPU but redirect EGL render
# to a healthy GPU's graphics engine (EGL enumeration is independent of CUDA_VISIBLE_DEVICES, and
# LIBERO copies frames to CPU, so render-GPU != compute-GPU is free). Override the lists via env.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
# Default empty: render on the same GPU as the server. Set e.g. BAD_EGL_RENDER_GPUS=0,3 on a
# host where some GPU's EGL context is wedged; renders targeting it move to the fallback.
BAD_EGL_RENDER_GPUS=${BAD_EGL_RENDER_GPUS:-}
EGL_RENDER_FALLBACK_GPU=${EGL_RENDER_FALLBACK_GPU:-0}
target_gpu="${CUDA_VISIBLE_DEVICES:-}"; target_gpu="${target_gpu%%,*}"  # server GPU = first in the mask
if [ -z "${MUJOCO_EGL_DEVICE_ID:-}" ] && [ -z "${GLANCEWAM_EGL_DEVICE_ID:-}" ] && [ -n "${target_gpu}" ]; then
  render_gpu="${target_gpu}"
  for bad in ${BAD_EGL_RENDER_GPUS//,/ }; do
    if [ "${target_gpu}" = "${bad}" ]; then
      render_gpu="${EGL_RENDER_FALLBACK_GPU}"
      echo "⚠️  nvidia-smi GPU ${target_gpu} has a wedged EGL graphics engine; redirecting render -> GPU ${render_gpu} (compute stays on ${target_gpu})"
      break
    fi
  done
  # Probe runs unmasked (it uses EGL enumeration + nvidia-smi, not CUDA) for true physical indices.
  egl_id=$(CUDA_VISIBLE_DEVICES= ${LIBERO_python} tools/egl_device_map.py "${render_gpu}" 2>/dev/null)
  export GLANCEWAM_EGL_DEVICE_ID="${egl_id:-0}"
fi
echo "🖥️  GLANCEWAM_EGL_DEVICE_ID=${GLANCEWAM_EGL_DEVICE_ID:-<unset>} -> MUJOCO_EGL_DEVICE_ID (render GPU = nvidia-smi ${render_gpu:-$target_gpu}, server compute = nvidia-smi ${target_gpu:-?})"

host=${host:-127.0.0.1}
num_trials_per_task=${num_trials_per_task:-50}
SPAWN_SLEEP=${SPAWN_SLEEP:-4}
###########################################################################################

# Output layout (mirrors eval_libero_parall.sh: <model_root>/{videos,logs,results}/<suite>/<folder>)
model_root=$(echo "$ckpt_path" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "$ckpt_path" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

video_out_path=${video_out_path:-"${model_root}/videos/${task_suite_name}/${folder_name}"}
log_dir=${log_dir:-"${model_root}/logs/${task_suite_name}/${folder_name}"}
results_dir=${results_dir:-"${model_root}/results/${task_suite_name}/${folder_name}"}
mkdir -p "$video_out_path" "$log_dir" "$results_dir"

echo "=========================================="
echo " start_libero_env: ${num_shards} client shards → ${host}:${port}"
echo "   ckpt   : ${ckpt_path}"
echo "   suite  : ${task_suite_name}  (trials/task=${num_trials_per_task})"
echo "   results: ${results_dir}"
echo "=========================================="

pids=()
for ((shard_idx=0; shard_idx<num_shards; shard_idx++)); do
    shard_log="${log_dir}/shard_${shard_idx}_of_${num_shards}.log"
    shard_json="${results_dir}/shard_${shard_idx}_of_${num_shards}.json"
    echo "▶️ shard ${shard_idx}/${num_shards} → ${shard_log}"

    ${LIBERO_python} ./examples/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path "${ckpt_path}" \
        --args.host "${host}" \
        --args.port "${port}" \
        --args.task-suite-name "${task_suite_name}" \
        --args.num-trials-per-task "${num_trials_per_task}" \
        --args.num-shards "${num_shards}" \
        --args.shard-idx "${shard_idx}" \
        --args.video-out-path "${video_out_path}" \
        --args.results-out "${shard_json}" \
        > "${shard_log}" 2>&1 &
    pids+=($!)

    sleep "${SPAWN_SLEEP}"
done

echo "--- ${num_shards} shards launched. Waiting... ---"
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done

echo "--- All shards finished for suite ${task_suite_name} (rc=${rc}) ---"
exit $rc
