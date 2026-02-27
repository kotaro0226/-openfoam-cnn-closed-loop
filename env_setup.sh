#Setup#
module load PrgEnv-gnu
module load cray-mpich/8.1.31
module load rocm/6.3.1
module load miniforge3/23.11.0-0
module unload libfabric
module load libfabric/1.20.1
module -t list
conda activate gpu_env
conda list

export NODES=$SLURM_JOB_NUM_NODES

#PATH
export DATAD="${HOMED}/data"
export WORKD="${HOMED}/smartsim-ml"
export SICKLE_ROOT="${HOMED}/sickle-of"
export CASED="TGStratifiedLES_ML"
export FOAM_RUN="${HOMED}/${CASED}"
export WM_PROJECT_USER_DIR="$FOAM_RUN"

#OpenFOAM
source /lustre/orion/scratch/shunya/gen053/src/OpenFOAM-v2312/etc2/bashrc
export WM_MPLIB=CRAY-MPICH
export WM_CC=cc
export WM_CXX=CC

#SmartSIM
#export PORT=7000
#export IFACE="lo"   # localhost
export MIOPEN_USER_DB_PATH="/tmp/${USER}/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}
#export SR_LOG_LEVEL=DEBUG
#export SR_LOG_FILE=$WORKD/smartredis_client.log


