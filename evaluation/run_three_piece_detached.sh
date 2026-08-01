#!/usr/bin/env bash
set -u

evaluation_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner_path="${evaluation_root}/evaluation/run_question_evaluation.py"
calibration_runner="${evaluation_root}/evaluation/calibrate_judge.py"
judge_model_path="${SOPRANO_QA_JUDGE_MODEL_PATH:-/home/minhee/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
dataset_root="${SOPRANO_QA_RAG_DATASET_ROOT:-/home/minhee/soprano-qa-dataset}"
judge_max_tokens="${SOPRANO_QA_JUDGE_MAX_TOKENS:-2048}"
range_embedding_model_path="${SOPRANO_QA_RANGE_EMBEDDING_MODEL_PATH:-/home/minhee/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}"
range_nli_model_path="${SOPRANO_QA_RANGE_NLI_MODEL_PATH:-/home/minhee/.cache/huggingface/hub/models--chunwoolee0--klue_nli_roberta_base_model/snapshots/d025228dc83570f814626c2b32980d514c65875d}"
range_nli_threshold="${SOPRANO_QA_RANGE_NLI_THRESHOLD:-0.8}"
range_embedding_delta_threshold="${SOPRANO_QA_RANGE_EMBEDDING_DELTA_THRESHOLD:-0.05}"
calibration_path="${SOPRANO_QA_JUDGE_CALIBRATION:-${evaluation_root}/evaluation/judge_calibration_results.json}"
output_path="${SOPRANO_QA_EVALUATION_OUTPUT:-${evaluation_root}/evaluation/three_piece_results.json}"
status_path="${SOPRANO_QA_EVALUATION_STATUS:-${evaluation_root}/evaluation/three_piece_run_status.json}"
conda_path="/home/minhee/miniconda3/bin/conda"

write_status() {
  local phase="$1"
  local state="$2"
  local exit_code="$3"
  local temporary_status="${status_path}.tmp"
  printf \
    '{"phase":"%s","state":"%s","exit_code":%s,"updated_at":"%s"}\n' \
    "$phase" \
    "$state" \
    "$exit_code" \
    "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
    >"$temporary_status"
  mv "$temporary_status" "$status_path"
}

common_arguments=(
  --dataset-root "$dataset_root"
  --output "$output_path"
  --judge-backend transformers
  --judge-model-path "$judge_model_path"
  --judge-max-tokens "$judge_max_tokens"
  --range-embedding-model-path "$range_embedding_model_path"
  --range-nli-model-path "$range_nli_model_path"
  --range-nli-threshold "$range_nli_threshold"
  --range-embedding-delta-threshold "$range_embedding_delta_threshold"
  --judge-calibration "$calibration_path"
)

cd "$evaluation_root" || exit 1

write_status calibrate running 0
"$conda_path" run -n recsys-chall python "$calibration_runner" \
  --output "$calibration_path" \
  --dataset-root "$dataset_root" \
  --judge-model-path "$judge_model_path" \
  --judge-max-tokens "$judge_max_tokens" \
  --range-embedding-model-path "$range_embedding_model_path" \
  --range-nli-model-path "$range_nli_model_path" \
  --range-nli-threshold "$range_nli_threshold" \
  --range-embedding-delta-threshold "$range_embedding_delta_threshold"
run_status=$?
if ((run_status != 0)); then
  write_status calibrate failed "$run_status"
  exit "$run_status"
fi

write_status retrieval running 0
"$conda_path" run -n soprano-qa python "$runner_path" \
  --phase retrieval \
  "${common_arguments[@]}"
run_status=$?
if ((run_status != 0)); then
  write_status retrieval failed "$run_status"
  exit "$run_status"
fi

write_status generate running 0
"$conda_path" run -n soprano-qa python "$runner_path" \
  --phase generate \
  "${common_arguments[@]}"
run_status=$?
if ((run_status != 0)); then
  write_status generate failed "$run_status"
  exit "$run_status"
fi

write_status judge running 0
"$conda_path" run -n recsys-chall python "$runner_path" \
  --phase judge \
  "${common_arguments[@]}"
run_status=$?
if ((run_status != 0)); then
  write_status judge failed "$run_status"
  exit "$run_status"
fi

write_status complete complete 0
