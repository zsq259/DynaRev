#!/usr/bin/env bash
# Formal4 Batch0 baseline: 4 seeds × 3 variants × 3 policy models.
set -uo pipefail

cd "$(dirname "$0")/../.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dyna
set -a
source ./1.sh
set +a

export QWEN35_BASE_URL="${QWEN35_BASE_URL:-http://144.214.37.135:8080/v1}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
unset PLAYWRIGHT_BROWSERS_PATH
export PLAYWRIGHT_BROWSERS_PATH="$HOME/Library/Caches/ms-playwright"

DATASET="data/dyna-tasks/batches/batch0/tasks.jsonl"
DATE="${DATE:-20260915}"
TRIGGER_MODEL="${TRIGGER_MODEL:-deepseek-flash}"
EVALUATOR_MODEL="${EVALUATOR_MODEL:-deepseek-flash}"
MAX_STEPS="${MAX_STEPS:-30}"
overall_rc=0

ts() { date +'%Y-%m-%d %H:%M:%S'; }
trap 'echo "=== [$(ts)] INTERRUPTED ==="; exit 130' INT TERM

run_one() {
  local policy=$1
  local tag=$2
  local run="formal4_batch0_${tag}_${DATE}"

  echo "=== [$(ts)] EXEC policy=${policy} run=${run} ==="
  python -m src.dyna_agent \
    --dataset-type dyna_jsonl \
    --dataset-path "$DATASET" \
    --limit 0 \
    --policy-model "$policy" \
    --trigger-model "$TRIGGER_MODEL" \
    --max-steps "$MAX_STEPS" \
    --headless \
    --viewport-only \
    --no-task-isolation \
    --run-name "$run"
  local exec_rc=$?
  echo "=== [$(ts)] EXEC done rc=${exec_rc} ==="

  echo "=== [$(ts)] EVAL run=${run} ==="
  python -m src.evaluate_logs \
    --run-dir "logs/runs/$run" \
    --dataset-path "$DATASET" \
    --evaluator-model "$EVALUATOR_MODEL" \
    --force
  local eval_rc=$?
  echo "=== [$(ts)] EVAL done rc=${eval_rc} ==="

  if (( exec_rc != 0 || eval_rc != 0 )); then
    overall_rc=1
  fi
}

# Sequential: browser + API load stay manageable.
run_one "deepseek-flash" "deepseek-flash"
run_one "deepseek-v4-pro" "deepseek-v4-pro"
run_one "Qwen/Qwen3.5-27B" "qwen35"

echo "=== [$(ts)] ALL DONE (rc=${overall_rc}) ==="
exit "$overall_rc"
