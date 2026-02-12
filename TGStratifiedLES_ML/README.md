# TGStratifiedLES

Large-Eddy Simulation (LES) of stratified Taylor–Green vortices using
`buoyantBoussinesqPimpleFoam` in OpenFOAM v2312.
Configured to run on Frontier (OLCF) using the Boussinesq approximation
with the LES Smagorinsky turbulence model.

---------------------------------------------------------------------------
## Environment Setup (Frontier)

1. Load modules and source OpenFOAM

    module load PrgEnv-gnu

    source $PROJWORK/gen150/openfoam/OpenFOAM-v2312/etc/bashrc

2. Set environment variables to run dir

    export FOAM_RUN=/lustre/orion/scratch/$USER/cfd135/TGStratifiedLES

    export WM_PROJECT_USER_DIR=/lustre/orion/scratch/$USER/cfd135/TGStratifiedLES

---------------------------------------------------------------------------
## Case Directory Structure

```
TGStratifiedLES/
├── 0/                  ← initial fields (U, T, p_rgh, nut, alphat)
├── constant/           ← mesh, physical properties, turbulence setup
│   ├── g
│   ├── transportProperties
│   ├── turbulenceProperties
│   └── polyMesh/
│       └── blockMeshDict
└── system/             ← solver and runtime settings
    ├── controlDict
    ├── fvSchemes
    ├── fvSolution
    └── (optional) decomposeParDict
```

## Running the Simulation

    cd $FOAM_RUN/TGStratifiedLES

## Generate the mesh

    blockMesh

## Desompose into parallel sub-parts

    decomposePar -force

## Generate initialization files T and U

    python initialize_T.py

    python initialize_U.py

## Run the LES solver

    srun -n 64 buoyantBoussinesqPimpleFoam -parallel > log.TG 2>&1

## Post-Processing

To reconstruct full solution from parallel parts:

    reconstructPar -latestTime

Then can run something like:

    postProcess -func "fieldMinMax(U)" -time latestTime

Also, can convert results for ParaView visualization:

    foamToVTK

## Notes

- Solver: buoyantBoussinesqPimpleFoam
- orig: Model: LES, Smagorinsky, incompressible Boussinesq
- now: Model: DNS, incompressible Boussinesq
- Typical grid: 512 × 512 × 256 (gravity along y-axis)
- Gravity vector defined in constant/g
- Time control in system/controlDict

---------------------------------------------------------------------------
## Next Steps

- Integrate ViT-based ML model for pressure initialization (replacing
  part of the pressure–Poisson iteration).
- Validate convergence reduction with ML-predicted pressure fields.

---------------------------------------------------------------------------
