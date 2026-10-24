#!/usr/bin/env bash
# 与 sh/run_seed_queue.sh 的用法一致：dataset key 后跟一个或多个手动指定的 seed。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
USE_CLASS_PROTOTYPE_QUERY="${USE_CLASS_PROTOTYPE_QUERY:-0}"
USE_CLASS_CONDITIONAL_SCALE_FUSION="${USE_CLASS_CONDITIONAL_SCALE_FUSION:-0}"
for value in "${USE_CLASS_PROTOTYPE_QUERY}" "${USE_CLASS_CONDITIONAL_SCALE_FUSION}"; do
  [[ "${value}" == 0 || "${value}" == 1 ]] || { echo "[ERROR] module switches must be 0 or 1" >&2; exit 2; }
done
case "${USE_CLASS_PROTOTYPE_QUERY}${USE_CLASS_CONDITIONAL_SCALE_FUSION}" in
  00) EXPERIMENT_NAME=baseline ;;
  10) EXPERIMENT_NAME=class_prototype_query ;;
  01) EXPERIMENT_NAME=class_conditional_scale_fusion ;;
  11) EXPERIMENT_NAME=prototype_query_and_scale_fusion ;;
esac
(( $# >= 2 )) || { echo "用法: bash research_emcad/scripts/train.sh <synapse|acdc|polyp|isic|busi|cell> <seed1> [seed2 ...]" >&2; exit 2; }
case "${1,,}" in
  synapse) TRAIN_ENTRY=train_synapse.py ;;
  acdc) TRAIN_ENTRY=train_acdc.py ;;
  polyp|cell) TRAIN_ENTRY=train_polyp.py ;;
  isic) TRAIN_ENTRY=train_isic.py ;;
  busi) TRAIN_ENTRY=train_busi.py ;;
  *) echo "[ERROR] invalid dataset family" >&2; exit 2 ;;
esac
test -f "${PROJECT_DIR}/sh/run_seed_queue.sh"
test -f "${PROJECT_DIR}/${TRAIN_ENTRY}" || { echo "[ERROR] training entry missing: ${TRAIN_ENTRY}" >&2; exit 1; }
export USE_CLASS_PROTOTYPE_QUERY USE_CLASS_CONDITIONAL_SCALE_FUSION
OUTPUT_ROOT="${OUTPUT_DIR:-${PROJECT_DIR}/research_emcad/runs}"
export OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
echo "[RESEARCH] experiment=${EXPERIMENT_NAME} prototype_query=${USE_CLASS_PROTOTYPE_QUERY} conditional_scale_fusion=${USE_CLASS_CONDITIONAL_SCALE_FUSION} output=${OUTPUT_DIR} dataset=$1 seeds=${*:2}"
exec bash "${PROJECT_DIR}/sh/run_seed_queue.sh" "$@"
