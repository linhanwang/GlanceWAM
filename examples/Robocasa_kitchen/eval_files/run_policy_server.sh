#!/bin/bash
# Policy server for RoboCasa-kitchen (cosmos-policy, PandaOmron) evaluation.
# Runs the glancewam model under the project's uv venv (./.venv at the repo root). The RoboCasa
# rollout client lives in its own py3.10 sim venv (../robocasa-cosmos-policy/.venv) and connects
# over the port below.
#
# Benchmark-agnostic server (deployment/model_server/server_policy.py + WebsocketPolicyServer) —
# Concurrent client
# shards coalesce into one batched forward when MAX_BATCH >= total shards.
#
# Usage:
#   bash run_policy_server.sh                            # use defaults below
#   bash run_policy_server.sh <ckpt_path>               # override ckpt
#   bash run_policy_server.sh <ckpt_path> <port>        # override ckpt + port
#   CUDA_VISIBLE_DEVICES=1 bash run_policy_server.sh ... # override GPU

############# Environment setup #############
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

export star_vla_python="${glancewam_python:-${REPO_ROOT}/.venv/bin/python}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Force torch's bundled cuBLAS/cuBLASLt to win over any system CUDA toolkit on LD_LIBRARY_PATH
# (a version mismatch surfaces as CUBLAS_STATUS_NOT_INITIALIZED on the first matmul).
export LD_LIBRARY_PATH="${REPO_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"

default_port=6699
default_gpu=0
default_ckpt=results/Checkpoints/robocasa_kitchen_GlanceWAM/checkpoints/steps_20000_pytorch_model_ema.pt

your_ckpt="${1:-${default_ckpt}}"
port="${2:-${default_port}}"
gpu_id="${CUDA_VISIBLE_DEVICES:-${default_gpu}}"

# Server-side request batching. The eval driver overrides these via env to
# MAX_BATCH=<total_shards> / BATCH_WAIT_MS=5 so one forward serves all rollout client shards.
# Set MAX_BATCH=1 to disable batching entirely (debugging only).
max_batch="${MAX_BATCH:-16}"
batch_wait_ms="${BATCH_WAIT_MS:-5}"

# Optional flow-matching Euler-step override (GR00T / PI heads). Unset => checkpoint default (4).
num_inference_timesteps="${NUM_INFERENCE_TIMESTEPS:-}"
extra_args=()
if [ -n "${num_inference_timesteps}" ]; then
    extra_args+=(--num-inference-timesteps "${num_inference_timesteps}")
fi
############# End environment setup #############

ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"

echo "▶️ policy server: ckpt=${your_ckpt} port=${port} gpu=${gpu_id} max_batch=${max_batch} batch_wait_ms=${batch_wait_ms} num_inference_timesteps=${num_inference_timesteps:-<default>} log=${log_file}"

#### run server #####
# Interactive shell: tee to console + log so the operator sees output live.
# Non-interactive (orchestrator subprocess): exec straight to log so SIGTERM propagates
# directly to the python process (no tee parent in between).
if [ -t 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpu_id} "${star_vla_python}" deployment/model_server/server_policy.py \
        --ckpt_path "${your_ckpt}" \
        --port "${port}" \
        --use_bf16 \
        --max-batch "${max_batch}" \
        --batch-wait-ms "${batch_wait_ms}" \
        "${extra_args[@]}" \
        2>&1 | tee "${log_file}"
else
    exec env CUDA_VISIBLE_DEVICES=${gpu_id} "${star_vla_python}" deployment/model_server/server_policy.py \
        --ckpt_path "${your_ckpt}" \
        --port "${port}" \
        --use_bf16 \
        --max-batch "${max_batch}" \
        --batch-wait-ms "${batch_wait_ms}" \
        "${extra_args[@]}" \
        >> "${log_file}" 2>&1
fi
