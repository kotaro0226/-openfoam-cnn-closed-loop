#!/bin/bash
#SBATCH -A gen053
#SBATCH -J OFTGML
##SBATCH -p extended
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -S 0
#SBATCH -t 00:30:00
#SBATCH -o /lustre/orion/scratch/shunya/gen053/work/TGStratifiedLES_ML/out/%j/%x_%j.out
#SBATCH -e /lustre/orion/scratch/shunya/gen053/work/TGStratifiedLES_ML/out/%j/%x_%j.out

module load PrgEnv-gnu
source /lustre/orion/proj-shared/gen053/openfoam/OpenFOAM-v2312/etc/bashrc
#source /lustre/orion/scratch/shunya/gen053/src/OpenFOAM-v2312/etc/bashrc
export FOAM_RUN="/lustre/orion/scratch/$USER/work/TGStratifiedLES_ML"
export WM_PROJECT_USER_DIR="/lustre/orion/scratch/$USER/work/TGStratifiedLES_ML"
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions        # Tutorial run functions

module load miniforge3/23.11.0-0
conda activate openfoam_smartsim_ML

cd $MEMBERWORK/gen053/work/TGStratifiedLES_ML

###----------------------------------------------------

#restore0Dir

if [ ! -d constant/polyMesh ]
then

    echo "meshing"
    blockMesh
    runApplication renumberMesh -overwrite -constant
    runApplication checkMesh -allTopology -allGeometry -constant

fi

postProcess -func writeCellCentres -time 0

echo "decomposing"
runApplication decomposePar -force

echo "initializing"
python initialize_T.py
python initialize_U.py

echo "running"
time srun -n 64 buoyantBoussinesqPimpleFoam -parallel
#time srun buoyantBoussinesqPimpleFoam

echo "reconstructPar"
reconstructPar -latestTime

echo "postProcess"
postProcess -func "fieldMinMax(U)" -time latestTime

echo "foamToVTK"
foamToVTK -latestTime

#


