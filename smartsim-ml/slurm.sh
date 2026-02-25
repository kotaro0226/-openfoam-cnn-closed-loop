#!/bin/bash
#SBATCH -A gen053
#SBATCH -J OFRML31
##SBATCH -p extended
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -S 0
#SBATCH -t 01:00:00
#SBATCH -o /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.err

#Setup#
export NODES=1
module load PrgEnv-gnu
module load cray-mpich
module load miniforge3/23.11.0-0
#conda activate openfoam_smartsim_ML
conda activate sickle_env

source /lustre/orion/scratch/shunya/gen053/src/OpenFOAM-v2312/etc/bashrc
export WM_MPLIB=CRAY-MPICH
export WM_CC=cc
export WM_CXX=CC

export HOMED="/lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3"
export DATAD="${HOMED}/data"
export WORKD="${HOMED}/smartsim-ml"
export SICKLE_ROOT="${HOMED}/sickle-of"
export CASED="TGStratifiedLES_ML"
export FOAM_RUN="${HOMED}/${CASED}"
export WM_PROJECT_USER_DIR="$FOAM_RUN"
export PORT=7000
export IFACE="lo"   # localhost


cd $WORKD

#------------------------

time python Main_pype.py | tee log.exe.txt

#------------------------





