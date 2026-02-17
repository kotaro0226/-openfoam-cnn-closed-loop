#!/bin/bash
#SBATCH -A gen053
#SBATCH -J OFRML_GRID
#SBATCH -p extended
##SBATCH -p batch
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH -S 0
#SBATCH -t 03:00:00
#SBATCH -o /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.err

# Setup
module load PrgEnv-gnu
module load cray-mpich
module load miniforge3/23.11.0-0
conda activate sickle_env

source /lustre/orion/scratch/shunya/gen053/src/OpenFOAM-v2312/etc/bashrc
export WM_MPLIB=CRAY-MPICH
export WM_CC=cc
export WM_CXX=CC

# Existing project paths (same as slurm.sh)
export HOMED="/lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3"
export DATAD="${HOMED}/data"
export WORKD="${HOMED}/smartsim-ml"
export SICKLE_ROOT="${HOMED}/sickle-of"
export CASED="TGStratifiedLES_ML"
export FOAM_RUN="${HOMED}/${CASED}"
export WM_PROJECT_USER_DIR="$FOAM_RUN"
export PORT=7000
export IFACE="lo"

# Optional: tweak grid by env without editing script
HIDDEN_GRID=${HIDDEN_GRID:-"32;16,16;8,8,8"}
LR_GRID=${LR_GRID:-"1e-3"}
DROPOUT_GRID=${DROPOUT_GRID:-"0.05"}
LAMBDA_CONT_GRID=${LAMBDA_CONT_GRID:-"0.0,1.0"}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-8192}
WHICH=${WHICH:-val}

# Reuse existing subsampling output path from Main_pype.py
CELL_NPZ=${CELL_NPZ:-"${DATAD}/sampled/subsampled_ofCell_random_random.npz"}

cd "$WORKD"

mkdir -p "${DATAD}/grid"

# Grid search (outputs under ${DATAD}/grid)
time python -u Main_pype_grid.py \
  --cell-npz "$CELL_NPZ" \
  --hidden-grid "$HIDDEN_GRID" \
  --lr-grid "$LR_GRID" \
  --dropout-grid "$DROPOUT_GRID" \
  --lambda-cont-grid "$LAMBDA_CONT_GRID" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --which "$WHICH" \
  --outdir "${DATAD}/grid" \
  --run-vis | tee log.grid.txt

