#!/bin/bash
#SBATCH -A gen053
#SBATCH -J OFRML31
##SBATCH -p extended
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
##SBATCH --ntasks-per-node=8
##SBATCH --gpus-per-node=8
##SBATCH --cpus-per-task=7
#SBATCH -t 00:20:00
#SBATCH -o /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3/out/%j/%x_%j.err

#Setup#
export HOMED="/lustre/orion/scratch/shunya/gen053/work/tmp/of-r-ml_3"
source ${HOMED}/env_setup.sh

cd $WORKD

#------------------------

time python Main_pype.py | tee log.exe.txt


