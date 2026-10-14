#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

source "${ROOT_DIR}/env_setup.sh"

WORK_DIR="${WORKD:-${SCRIPT_DIR}}"

CELL_NPZ="${CELL_NPZ:-${ROOT_DIR}/cases/active/data_et50_dt0p01/blocks/block16_halo2_s16_et50_dt0p01.npz}"
GRID_MODE="${GRID_MODE:-3d-block}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-3}"
LAMBDA_CONT="${LAMBDA_CONT:-200.0}"
LAMBDA_CONT_WARMUP_STEPS="${LAMBDA_CONT_WARMUP_STEPS:-100}"
BLOCK_DATA_LOSS="${BLOCK_DATA_LOSS:-masked}"
BLOCK_MASK_RATIO="${BLOCK_MASK_RATIO:-1.0}"
BLOCK_MASK_POLICY="${BLOCK_MASK_POLICY:-random}"
BLOCK_MASK_SIZE="${BLOCK_MASK_SIZE:-4}"
WIDTH="${WIDTH:-32}"
DEPTH="${DEPTH:-4}"
NORM="${NORM:-group}"
RESIDUAL="${RESIDUAL:-1}"
PADDING_MODE="${PADDING_MODE:-replicate}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOSS="${LOSS:-mse}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
DEVICE="${DEVICE:-auto}"
BEST_MODEL="${BEST_MODEL:-}"

EXTRA_ARGS=()
if [[ "${RESIDUAL}" == "1" || "${RESIDUAL}" == "true" || "${RESIDUAL}" == "yes" ]]; then
  EXTRA_ARGS+=(--residual)
fi
if [[ -n "${BEST_MODEL}" ]]; then
  EXTRA_ARGS+=(--save-best-model "${BEST_MODEL}")
fi

python3 "${WORK_DIR}/train_cnn.py" \
  --cell-npz "${CELL_NPZ}" \
  --grid-mode "${GRID_MODE}" \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --lambda-cont "${LAMBDA_CONT}" \
  --lambda-cont-warmup-steps "${LAMBDA_CONT_WARMUP_STEPS}" \
  --block-data-loss "${BLOCK_DATA_LOSS}" \
  --block-mask-ratio "${BLOCK_MASK_RATIO}" \
  --block-mask-policy "${BLOCK_MASK_POLICY}" \
  --block-mask-size "${BLOCK_MASK_SIZE}" \
  --width "${WIDTH}" \
  --depth "${DEPTH}" \
  --norm "${NORM}" \
  --padding-mode "${PADDING_MODE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --grad-clip "${GRAD_CLIP}" \
  --loss "${LOSS}" \
  --huber-delta "${HUBER_DELTA}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
