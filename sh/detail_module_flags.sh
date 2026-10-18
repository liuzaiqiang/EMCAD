#!/usr/bin/env bash
# 由各训练启动器 source；只定义显式配置，不修改训练超参数。
USE_FULL_RESOLUTION_DETAIL_BRANCH="${USE_FULL_RESOLUTION_DETAIL_BRANCH:-0}"
USE_SEMANTIC_GATED_RESIDUAL="${USE_SEMANTIC_GATED_RESIDUAL:-0}"
case "${USE_FULL_RESOLUTION_DETAIL_BRANCH}" in 0|1) ;; *) echo "[ERROR] USE_FULL_RESOLUTION_DETAIL_BRANCH must be 0 or 1" >&2; return 2 ;; esac
case "${USE_SEMANTIC_GATED_RESIDUAL}" in 0|1) ;; *) echo "[ERROR] USE_SEMANTIC_GATED_RESIDUAL must be 0 or 1" >&2; return 2 ;; esac
case "${USE_FULL_RESOLUTION_DETAIL_BRANCH}${USE_SEMANTIC_GATED_RESIDUAL}" in
  00) DETAIL_VARIANT="baseline" ;;
  10) DETAIL_VARIANT="full_resolution_detail" ;;
  01) DETAIL_VARIANT="semantic_gated_residual" ;;
  11) DETAIL_VARIANT="detail_and_semantic_gate" ;;
esac
test -f "${PROJECT_DIR}/lib/research_model.py" || { echo "[ERROR] research model factory missing" >&2; return 2; }
DETAIL_MODULE_ARGS=(--use_full_resolution_detail_branch "${USE_FULL_RESOLUTION_DETAIL_BRANCH}" --use_semantic_gated_residual "${USE_SEMANTIC_GATED_RESIDUAL}")
WAIT_FOR_TRAIN="${WAIT_FOR_TRAIN:-1}"
case "${WAIT_FOR_TRAIN}" in 0|1) ;; *) echo "[ERROR] WAIT_FOR_TRAIN must be 0 or 1" >&2; return 2 ;; esac
finish_detail_training() {
  if [[ "${WAIT_FOR_TRAIN}" == "1" ]]; then
    wait "$1"
  fi
}
