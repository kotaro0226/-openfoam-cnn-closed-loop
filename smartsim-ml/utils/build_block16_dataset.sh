#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

source "${ROOT_DIR}/env_setup.sh"

RAW_NPZ="${1:-${ROOT_DIR}/cases/active/data_et50_dt0p01/of-raw/et50_dt0p01.npz}"
CELL_MAPS="${2:-${ROOT_DIR}/cases/active/data_et50_dt0p01/maps/cell_maps_rank0_internal.npz}"
OUT_NPZ="${3:-${ROOT_DIR}/cases/active/data_et50_dt0p01/blocks/block16_halo2_s16_et50_dt0p01.npz}"

python "${ROOT_DIR}/smartsim-ml/utils/make_dense_block_dataset.py" \
  --raw-npz "${RAW_NPZ}" \
  --cell-maps "${CELL_MAPS}" \
  --out "${OUT_NPZ}" \
  --block-size 16 \
  --halo 2 \
  --stride 16
