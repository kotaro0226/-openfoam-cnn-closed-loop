---------------------------------------------------------------------------
## Build & Archival Notes (OpenFOAM v2312 on Frontier)

These notes document how OpenFOAM v2312 was built on Frontier to ensure
the environment can be reconstructed for reproducibility and long-term archiving.

### 1. System & Environment

System:  OLCF Frontier (HPE Cray EX)
Compiler: GCC (PrgEnv-gnu)
MPI: Cray MPICH (default)
Build type: Double precision, DPInt32

Active modules:
    module load gcc
    module load cray-mpich

### 2. Environment Variables

Set the OpenFOAM MPI type for Cray MPICH:

    export WM_MPLIB=CRAY-MPICH

### 3. Source Retrieval

Clone OpenFOAM from the official development repository:

    git clone https://develop.openfoam.com/Development/openfoam.git OpenFOAM-v2312
    cd OpenFOAM-v2312
    git checkout OpenFOAM-v2312

For reference, see the official build guide:
    https://develop.openfoam.com/Development/openfoam/-/blob/master/doc/Build.md

### 4. Build

    ./Allwmake -j 16 | tee log.build

To verify the installation:

    foamInstallationTest

### 5. Environment Activation

Each time you log in, re-enable OpenFOAM:

    module load gcc
    module load cray-mpich
    source ~/OpenFOAM/OpenFOAM-v2312/etc/bashrc

Optionally define the run directory on Frontier scratch:

    export FOAM_RUN=/lustre/orion/scratch/$USER/cfd135/run
    mkdir -p $FOAM_RUN

---------------------------------------------------------------------------

