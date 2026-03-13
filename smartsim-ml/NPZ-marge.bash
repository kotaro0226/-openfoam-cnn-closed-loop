#!/bin/bash
#SBATCH -A gen053
#SBATCH -J OFRML31
##SBATCH -p extended
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -S 0
#SBATCH -t 02:00:00
#SBATCH -o /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.err

#Setup#
module load miniforge3/23.11.0-0
#conda activate openfoam_smartsim_ML
conda activate sickle_env

export HOMED="/lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3"
export DATAD="${HOMED}/data"
export WORKD="${HOMED}/smartsim-ml"
export SICKLE_ROOT="${HOMED}/sickle-of"
export CASED="TGStratifiedLES_ML"
export FOAM_RUN="${HOMED}/${CASED}"

cd $WORKD

#------------------------

python utils/marge-npz.py \
  --inputs $DATAD/sampled/subsampled_a.npz data/sampled/subsampled_b.npz $DATAD/sampled/subsampled_b.npz \
  --output $DATAD/sampled/subsampled_merged.npz


