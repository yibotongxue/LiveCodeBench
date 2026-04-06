#!/bin/bash

# Auto-deploy vLLM servers and evaluate multiple local models using PIXIU.
# Usage: bash run_pixiu.sh <server_index>

set -euo pipefail

: "${OPENAI_API_KEY:=}"
export OPENAI_API_KEY

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR="${ROOT_DIR}"

# ============================================================================
# PIXIU Eval settings
# ============================================================================
# Set PYTHONPATH for PIXIU
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/src/financial-evaluation:${ROOT_DIR}/src/metrics/BARTScore"

# ============================================================================
# Tasks to evaluate
# ============================================================================
# Use comma-separated list for specific tasks:
#   "flare_en_finqa,flare_en_convfinqa" - FinQA and ConvFinQA (current)
#   "flare_en_*"                          - All English FLARE tasks
#   "flare_es_*"                          - All Spanish FLARE tasks
#   "flare_cra_*"                         - All Credit Risk Assessment tasks
#   "flare_sm_*"                          - All Stock Movement tasks
#   "flare_*"                             - All FLARE tasks (including test sets)
# TASKS="flare_finqa,flare_convfinqa"
TASKS="flare_cfa"

# ============================================================================
# PIXIU Eval parameters - aligned with repository defaults
# ============================================================================
# IMPORTANT: --model should be "gpt3" to use ChatLM class (messages format)
# The actual model being evaluated is specified via --base_url and --model_args
MODEL_TYPE="gpt3"  # Use "gpt3" to select ChatLM class (messages format), not the actual model

# Core evaluation parameters (affecting results)
# Based on repository defaults from run_evaluation.sh and chatlm.py
MODEL_PROMPT="no_prompt"  # No prefix needed for messages format
NUM_FEWSHOT=5  # Zero-shot (from run_evaluation.sh: --num_fewshot 0)
MAX_GEN_TOKS=8192  # Max tokens (from run_evaluation.sh: max_gen_toks=1024)
TEMPERATURE=1.0  # Temperature (0.0 = greedy decoding, 0.7 = sampling)
MAX_CONCURRENT=256  # Max concurrent API requests

# Async mode settings
USE_ASYNC=true          # Set to true to use async stream processing
ASYNC_CONCURRENCY=16    # Max concurrent problems (LLM + eval)
EVAL_CONCURRENCY=8      # Max concurrent eval subprocesses

# Output directory name (will be under outputs/)
OUTPUT_DIR="cfa-zero-shot"

# ============================================================================
# vLLM settings
# ============================================================================
TP_SIZE=1
GPU_MEMORY_UTIL=0.95
MAX_MODEL_LEN=16384
DTYPE="bfloat16"
HEALTH_TIMEOUT=600
BASE_PORT=8002

# 模型家族路径列表
MODEL_FAMILIES=(
    "/jizhicfs/linyibo/LlamaFactory/saves/Qwen2.5-7B"
    "/jizhicfs/linyibo/LlamaFactory/saves/Llama-3.1-8B"
    "/jizhicfs/linyibo/LlamaFactory/saves/Mistral-7B-v0.3"
    "/jizhicfs/linyibo/LlamaFactory/saves/Llama-3.2-3B"
)

# 与 MODEL_FAMILIES 一一对应的模板名称（用于构建 sft-${template} 路径）
TEMPLATES=(
    "qwen"
    "llama3"
    "mistral"
    "llama3"
)

# Ensure MODEL_FAMILIES and TEMPLATES have the same length
if [ "${#MODEL_FAMILIES[@]}" -ne "${#TEMPLATES[@]}" ]; then
    echo "Error: MODEL_FAMILIES and TEMPLATES must have the same length."
    exit 1
fi

SKIP_KEYWORDS=("oda" "stem" "lmsys_chat" "medical" "Medical" "science" "code")

# ========== 构建模型列表（同时记录对应模板） ==========
MODELS=()          # 模型完整路径
MODEL_TEMPLATES=() # 每个模型对应的模板名称
# shopt -s nullglob
for idx in "${!MODEL_FAMILIES[@]}"; do
    family="${MODEL_FAMILIES[$idx]}"
    template="${TEMPLATES[$idx]}"
    sft_dir="${family}/full/sft-${template}"
    if [ ! -d "$sft_dir" ]; then
        echo "Warning: missing directory $sft_dir, skipping."
        continue
    fi
    for subdir in "$sft_dir"/*; do
        if [ -d "$subdir" ]; then
            # 可选过滤规则
            subdir_name=$(basename "$subdir")
            skip=0
            for kw in "${SKIP_KEYWORDS[@]}"; do
                if [[ "$subdir_name" == *"$kw"* ]]; then
                    skip=1
                    break
                fi
            done
            [ $skip -eq 1 ] && continue
            # 检查模型文件存在（假设 safetensors 存在）
            if ! ls "$subdir"/model*.safetensors >/dev/null 2>&1; then
                echo "Warning: no .safetensors found in $subdir, skipping."
                continue
            fi
            MODELS+=("$subdir")
            MODEL_TEMPLATES+=("$template")
        fi
    done
done
# shopt -u nullglob

# ============================================================================
# Filter out models that already have output directories
# ============================================================================
echo "========================================"
echo "Checking existing output directories in: outputs/${OUTPUT_DIR}"
echo "========================================"
FILTERED_MODELS=()
for model_path in "${MODELS[@]}"; do
    # Calculate family and template from path
    # Path format: /xxx/LlamaFactory/saves/Llama-3.1-8B/full/sft-llama3/xxx
    family_name=$(basename "$(dirname "$(dirname "$(dirname "$model_path")")")")
    template_name=$(basename "$(dirname "$model_path")" | sed 's/sft-//')
    model_name=$(basename "$model_path")

    # Output dir for LiveCodeBench: output/<family>-<template>/<model_name>
    # Check if evaluation already completed
    lcb_eval_file="output/${OUTPUT_DIR}/${family_name}-${template_name}/${model_name}/codegeneration_1_0.2_eval.json"
    if [ -f "$lcb_eval_file" ]; then
        echo "Skipping $model_name (${family_name}-${template_name}): LiveCodeBench eval file already exists"
        continue
    fi
    FILTERED_MODELS+=("$model_path")
done
MODELS=("${FILTERED_MODELS[@]}")
echo ""

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "All models have already been evaluated. Exiting."
    exit 0
fi

echo "Remaining models to evaluate: ${#MODELS[@]}"
echo ""

# ============================================================================
# GPU pools by server index
# ============================================================================
GPU_POOLS=(
    # "0"
    "1 2 3 4 5 6 7"
    # "0 1 2 3 4 5 6 7"
    # "0 1 2 3 4 5 6 7"
    # "6"
    # "0 2 3 5 6"
    # "0 1 2 3 4 5"
    # "0 1 2 3 4 5 6 7"
)

usage() {
    echo "Usage: $0 <server_index>"
    echo "server_index must be a positive integer that selects a GPU pool."
}

if [ $# -ne 1 ]; then
    usage
    exit 1
fi

SERVER_IDX=$1
if ! [[ "$SERVER_IDX" =~ ^[0-9]+$ ]]; then
    echo "Error: server_index must be a positive integer."
    exit 1
fi

if [ "$SERVER_IDX" -ge "${#GPU_POOLS[@]}" ]; then
    echo "Error: server_index out of range."
    exit 1
fi

# Build a global GPU key list across all servers: "server_idx:gpu_id".
declare -a GLOBAL_GPU_KEYS
for ((s=0; s<${#GPU_POOLS[@]}; s++)); do
    read -r -a _gpus <<< "${GPU_POOLS[$s]}"
    for g in "${_gpus[@]}"; do
        GLOBAL_GPU_KEYS+=("${s}:${g}")
    done
done

if [ "${#GLOBAL_GPU_KEYS[@]}" -eq 0 ]; then
    echo "Error: no GPUs configured in GPU_POOLS."
    exit 1
fi

read -r -a LOCAL_GPUS <<< "${GPU_POOLS[$SERVER_IDX]}"
if [ "${#LOCAL_GPUS[@]}" -eq 0 ]; then
    echo "Error: no GPUs configured for server_index=$SERVER_IDX."
    exit 1
fi

# Assign models to all GPUs in round-robin order (across all servers).
declare -a ASSIGNMENTS
for ((i=0; i<${#GLOBAL_GPU_KEYS[@]}; i++)); do
    ASSIGNMENTS[$i]=""
done

for ((i=0; i<${#MODELS[@]}; i++)); do
    gpu_slot=$((i % ${#GLOBAL_GPU_KEYS[@]}))
    if [ -z "${ASSIGNMENTS[$gpu_slot]}" ]; then
        ASSIGNMENTS[$gpu_slot]="${MODELS[$i]}"
    else
        ASSIGNMENTS[$gpu_slot]="${ASSIGNMENTS[$gpu_slot]}|${MODELS[$i]}"
    fi
done

wait_for_health() {
    local health_url=$1
    local timeout_secs=$2
    local start_ts
    start_ts=$(date +%s)

    sleep 60
    while true; do
        if curl -s -o /dev/null -w "%{http_code}" "$health_url" | grep -q "^200$"; then
            return 0
        fi
        local now_ts
        now_ts=$(date +%s)
        if [ $((now_ts - start_ts)) -ge "$timeout_secs" ]; then
            return 1
        fi
        sleep 10
    done
}

run_models_on_gpu() {
    local gpu_id=$1
    local models_str=$2

    if [ -z "$models_str" ]; then
        return 0
    fi

    IFS='|' read -r -a models <<< "$models_str"

    for model_path in "${models[@]}"; do
        local model_name
        model_name=$(basename "$model_path")

        local port=$((BASE_PORT + gpu_id))
        local base_url="http://127.0.0.1:${port}/v1"

        # Extract family and template from path
        # Path: /xxx/LlamaFactory/saves/Llama-3.1-8B/full/sft-llama3/dataflow
        local family_name
        family_name=$(basename "$(dirname "$(dirname "$(dirname "$model_path")")")")
        local template_name
        template_name=$(basename "$(dirname "$model_path")" | sed 's/sft-//')

        # Output directory for LiveCodeBench
        local output_base_dir="output/${OUTPUT_DIR}/${family_name}-${template_name}/${model_name}"

        # Second check: if LiveCodeBench eval file already exists, skip this model
        if [ -f "${output_base_dir}/codegeneration_1_0.2_eval.json" ]; then
            echo "Skipping $model_name (${family_name}-${template_name}): LiveCodeBench eval file already exists"
            continue
        fi

        # Create output directory for each task
        mkdir -p "$output_base_dir"

        # Create placeholder to mark this model is being evaluated
        touch "${output_base_dir}/.evaluating"

        local log_dir="outputs/${OUTPUT_DIR}/vllm_logs"
        local log_file="${log_dir}/vllm_${family_name}_${model_name}_server${SERVER_IDX}_gpu${gpu_id}.log"

        mkdir -p "$log_dir"

        # Start vLLM server
        CUDA_VISIBLE_DEVICES="$gpu_id" python -m vllm.entrypoints.openai.api_server \
            --model "$model_path" \
            --served-model-name "$model_name" \
            --port "$port" \
            --tensor-parallel-size "$TP_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --dtype "$DTYPE" \
            > "$log_file" 2>&1 &
        local vllm_pid=$!

        if ! wait_for_health "http://127.0.0.1:${port}/health" "$HEALTH_TIMEOUT"; then
            echo "Error: vLLM health check failed for GPU $gpu_id, model $model_name."
            kill "$vllm_pid" 2>/dev/null || true
            wait "$vllm_pid" 2>/dev/null || true
            rm "${output_base_dir}/.evaluating" 2>/dev/null || true
            return 1
        fi

        # Run LiveCodeBench evaluation
        local lcb_scenario="codegeneration"
        local custom_save_name="${OUTPUT_DIR}/${family_name}-${template_name}/${model_name}"

        echo "Evaluating $model_name on LiveCodeBench scenario: $lcb_scenario..."

        # Run LiveCodeBench eval with tee logging
        local eval_log="${output_base_dir}/eval.log"
        mkdir -p "$output_base_dir"
        cd "$EVAL_DIR"

        # Build async args if enabled
        async_args=()
        if [ "$USE_ASYNC" = true ]; then
            async_args+=(
                --async_mode
                --async_concurrency "$ASYNC_CONCURRENCY"
                --eval_concurrency "$EVAL_CONCURRENCY"
            )
        fi

        if ! python -m lcb_runner.runner.main \
            --model "$model_name" \
            --scenario "$lcb_scenario" \
            --n 1 \
            --codegen_n 1 \
            --temperature 0.2 \
            --top_p 0.95 \
            --max_tokens 2000 \
            --multiprocess 0 \
            --release_version release_latest \
            --evaluate \
            --num_process_evaluate 12 \
            --timeout 6 \
            --openai_timeout 90 \
            --custom_output_save_name "$custom_save_name" \
            --api_base_url "$base_url" \
            --api_key "" \
            "${async_args[@]}" 2>&1 | tee "$eval_log"; then
            echo "Error: evaluation failed for GPU $gpu_id, model $model_name."
            cd - > /dev/null
            kill "$vllm_pid" 2>/dev/null || true
            wait "$vllm_pid" 2>/dev/null || true
            rm "${output_base_dir}/.evaluating" 2>/dev/null || true
            return 1
        fi
        cd - > /dev/null

        # Evaluation completed successfully
        kill "$vllm_pid" 2>/dev/null || true
        wait "$vllm_pid" 2>/dev/null || true
        rm "${output_base_dir}/.evaluating" 2>/dev/null || true

        echo "Successfully evaluated $model_name"
    done
}

read -p "Press Enter to start evaluations on server ${SERVER_IDX}..."

pids=()
for ((i=0; i<${#GLOBAL_GPU_KEYS[@]}; i++)); do
    key="${GLOBAL_GPU_KEYS[$i]}"
    server_id="${key%%:*}"
    gpu_id="${key##*:}"
    models_str="${ASSIGNMENTS[$i]}"

    if [ "$server_id" -ne "$SERVER_IDX" ]; then
        continue
    fi

    (
        run_models_on_gpu "$gpu_id" "$models_str"
    ) &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All evaluations completed."
