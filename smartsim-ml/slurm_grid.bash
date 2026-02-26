#!/bin/bash
#SBATCH -A gen053
#SBATCH -J OFRML_GRID
#SBATCH -p extended
##SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:45:00
#SBATCH -o /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.err

# Setup
export NODES=1
module load PrgEnv-gnu
module load cray-mpich/8.1.31
module load rocm/6.3.1
module load miniforge3/23.11.0-0
module unload libfabric
module load libfabric/1.20.1
module -t list
conda activate gpu_env

source /lustre/orion/scratch/shunya/gen053/src2/OpenFOAM-v2312/etc/bashrc
export WM_MPLIB=CRAY-MPICH
export WM_CC=cc
export WM_CXX=CC

#export SR_LOG_LEVEL=DEBUG
#export SR_LOG_FILE=$WORKD/smartredis_client.log

# Optimizations for inference
export MIOPEN_USER_DB_PATH="/tmp/${USER}/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

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
HIDDEN_GRID=${HIDDEN_GRID:-"16;32,32;32,32,32"}
LR_GRID=${LR_GRID:-"1e-3"}
DROPOUT_GRID=${DROPOUT_GRID:-"0.05"}
LAMBDA_CONT_GRID=${LAMBDA_CONT_GRID:-"0.0,1.0"}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-8192}
WHICH=${WHICH:-val}
DDP_BACKEND=${DDP_BACKEND:-"nccl"}

# Reuse existing subsampling output path from Main_pype.py
CELL_NPZ=${CELL_NPZ:-"${DATAD}/sampled/subsampled_ofCell_random_random.npz"}

cd "$WORKD"

mkdir -p "${DATAD}/grid"

# Grid search (outputs under ${DATAD}/grid)
# Unified grid search (DDP-ready):
# - default USE_TORCHRUN=0 keeps single-process behavior.
# - set USE_TORCHRUN=1 to use torchrun (single/multi-node).
USE_TORCHRUN=1
USE_TORCHRUN=${USE_TORCHRUN:-0}


if [ "$USE_TORCHRUN" = "1" ]; then
  NNODES=${SLURM_NNODES:-1}
  NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-1}
  export MASTER_ADDR=$(hostname -i)
  export NCCL_SOCKET_IFNAME=hsn0
  export MASTER_PORT=3442

  time srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=8 \
    torchrun \
      --nnodes="${NNODES}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --node_rank="${SLURM_NODEID}" \
      --rdzv_id="${SLURM_JOB_ID}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      train_grid.py \
      --ddp-backend "${DDP_BACKEND}" \
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
else
  time python -u train_grid.py \
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
fi


