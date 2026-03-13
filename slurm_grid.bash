#!/bin/bash
#SBATCH -A gen053
#SBATCH -J OFRML_GRID
#SBATCH -p extended
##SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00
#SBATCH -o /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.err

# Setup
export HOMED="/lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3"
source ${HOMED}/env_setup.sh
cd $WORKD

#------------------------

# Optional: tweak grid by env without editing script
HIDDEN_GRID=${HIDDEN_GRID:-"32,32;32,64;32,128;64,64;64,128;128,128"}
LR_GRID=${LR_GRID:-"1e-3"}
DROPOUT_GRID=${DROPOUT_GRID:-"0.05"}
LAMBDA_CONT_GRID=${LAMBDA_CONT_GRID:-"0.0,1.0"}
EPOCHS=${EPOCHS:-3000}
BATCH=${BATCH:-8000}
WHICH=${WHICH:-val}
DDP_BACKEND=${DDP_BACKEND:-"nccl"}

# Reuse existing subsampling output path from Main_pype.py
DATA="TGV_5-25_fixed"
#DATAD="subsampled_ofCell_random_random"
CVNPZ=0
CELL_NPZ=${CELL_NPZ:-"${DATAD}/of-raw_fixed/${DATA}.npz"}
#CELL_NPZ=${CELL_NPZ:-"${DATAD}/of-raw/${DATA}.npz"}
#CELL_NPZ=${CELL_NPZ:-"${DATAD}/sampled/${DATA}.npz"}

mkdir -p "${DATAD}/grid"

# Grid search (outputs under ${DATAD}/grid)
# Unified grid search (DDP-ready):
# - default USE_TORCHRUN=0 keeps single-process behavior.
# - set USE_TORCHRUN=1 to use torchrun (single/multi-node).
USE_TORCHRUN=1
USE_TORCHRUN=${USE_TORCHRUN:-0}


if [ "$CVNPZ" = "1" ];then
	echo "of-raw file fix & use"
	mkdir -p ${DATAD}/of-raw_fixed
	
	python smartsim-ml/utils/add_cell_continuity_meta.py \
	  --cell-npz ${CELL_NPZ}.npz \
	  --cell-maps ${DATAD}/maps/cell_maps_rank0_internal.npz \
	  --out ${DATAD}/of-raw_fixed/${DATA}_fixed.npz 

	CELL_NPZ=${CELL_NPZ:-"${DATAD}/of-raw_fixed/${DATA}_fixed.npz"}
	echo "Input NPZ file switched: ${CELL_NPZ}"
fi


if [ "$USE_TORCHRUN" = "1" ]; then
  NNODES=${SLURM_NNODES:-1}
  NTASKS=$((NNODES*8))
  export NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-1}
  echo "NODE : $NNODES"
  echo "TASK : $NTASKS"
  echo "PROC : $NPROC_PER_NODE"

  time srun -N "${NNODES}" -n "${NTASKS}" -c 7 --ntasks-per-node=8 \
	--gpus-per-task=1 --gpu-bind=closest \
    python -u train_grid.py \
      --ddp-backend "${DDP_BACKEND}" \
      --cell-npz "$CELL_NPZ" \
      --hidden-grid "$HIDDEN_GRID" \
      --lr-grid "$LR_GRID" \
      --dropout-grid "$DROPOUT_GRID" \
      --lambda-cont-grid "$LAMBDA_CONT_GRID" \
      --epochs "$EPOCHS" \
      --batch "$BATCH" \
      --which "$WHICH" \
      --num-workers "8" \
      --outdir "${DATAD}/grid" \
      --run-vis |& tee log.grid.txt
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


