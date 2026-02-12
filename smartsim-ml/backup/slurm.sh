#!/bin/bash
#SBATCH -A gen053
#SBATCH -J OFRML1
##SBATCH -p extended
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -S 0
#SBATCH -t 01:00:00
#SBATCH -o /lustre/orion/scratch/$USER/gen053/work/of-r-ml_test1/smartsim/out/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/$USER/gen053/work/of-r-ml_test1/smartsim/out/%j/%x_%j.out

#Setup#
module load PrgEnv-gnu
module load miniforge3/23.11.0-0
conda activate openfoam_smartsim
source /lustre/orion/proj-shared/gen053/openfoam/OpenFOAM-v2312/etc/bashrc
#conda activate openfoam_smartsim_ML
#source /lustre/orion/scratch/$USER/gen053/src/OpenFOAM-v2312/etc/bashrc
export WORKD="/lustre/orion/scratch/$USER/gen053/work/of-r-ml_test1/smartsim"
export CASED="pitzDaily"
export FOAM_RUN="$WORKD/$CASED"
export WM_PROJECT_USER_DIR="$WORKD/$CASED"
export PORT=6780
export IFACE="lo"   # localhost

cd $WORKD

#------------------------

time python Main_pype.py

#------------------------

