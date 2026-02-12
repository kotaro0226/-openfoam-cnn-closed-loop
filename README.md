# OpenFOAM × SmartSim × Redis Inner Loop Verification

## Overview

This code was created by modifying OpenFOAM v2312's buoyantBoussinesqPimpleFoam to verify the operation of sending temporary variables within the inner loop to Redis.

By executing this code, you can:

- Retrieve temporary variables generated inside the inner loop of the OpenFOAM solver
- Send those variables to Redis
- Verify the size of the transmitted arrays

The solver places temporary variables into the objectRegistry and sends them to Redis via functionObject.

---

## Environment Setup

This code has been verified on Frontier with the following environment:

- miniforge3/23.11.0-0  
- Python 3.10  
- SmartSim 6.2.0  
- RedisAI (SmartSim 6.2.0)  
- PyTorch Backend (SmartSim 6.2.0)  
- SICKLE  

Related repository:
https://github.com/OFDataCommittee/openfoam-smartsim/tree/main

---

### Environment Validation Cases

Before running this code, verify your environment using:

- TGStratifiedLES  
- openfoam-smartsim/tutorials/functionObject  
- osmi-bench/smartsim/small_lstm case  

---

## Links

https://code.ornl.gov/autosm/tgstratifiedles  
https://github.com/OFDataCommittee/openfoam-smartsim/tree/main  
https://github.com/laszewsk/osmi-bench-new/tree/main  
https://github.com/at-aaims/sickle  

---

## Solver Installation

The required solver is included.

Place it under:

```
OpenFOAM-v2312/applications/solvers/heatTransfer/
```

Then build it.

---

# Project Structure

```
$PROJECT_ROOT/
├── buoyantBoussinesqPimpleFoam_ML/
│   # Custom OpenFOAM solver with pEqnFlux_ml hook
│
├── TGStratifiedLES_ML/           # OpenFOAM case
│   ├── system/
│   │   └── controlDict           # function changed
│   ├── constant/
│   │   └── polyMesh/
│   │       ├── owner
│   │       ├── neighbour
│   │       └── ...
│   ├── 0/
│   ├── initialize_T.py
│   └── initialize_U.py
│
├── sickle-of/                  # SICKLE framework (subsampling & analysis)
│   ├── subsample.py              # Output changed
│   ├── args.py                   # Extensions such as dtype modified
│   ├── constants.py
│   ├── dataloaders/
│   │   ├── __init__.py
│   │   ├── of-face-npz.py        # Custom DataLoader (face-wise NPZ)
│   │   ├── of-cell-npz.py        # Custom DataLoader (cell-wise NPZ)
│   │   └── ...
│   ├── subsampling/
│   │   ├── random.py
│   │   ├── maxent.py
│   │   └── ...
│   ├── hypercubes/
│   ├── viz/
│   ├── snapshots/                # Output of SICKLE subsampling
│   │   ├── subsampled_ofFace_x_random_random.npz
│   │   ├── subsampled_ofFace_y_random_random.npz
│   │   ├── subsampled_ofFace_z_random_random.npz
│   │   └── subsampled_ofCell_random_random.npz
│   └── plots/
│
└── smartsim-ml/                     # $WORKD (main working directory)
    ├── wsl.bash
    ├── slrum.sh
    ├── srun.bash
    ├── clean.bash
    ├── Main_pype.py
    ├── exp_openfoam.py           # SmartSim + OpenFOAM execution
    ├── of_common.py              # Redis / OpenFOAM utilities
    ├── make_cfd_dataset.py       # Redis → face-wise NPZ
    ├── make_face_maps_from_C.py  # Cell-centers → structured face maps
    ├── make_cell_maps.py         # Face maps & V → structured cell maps
    ├── train.py                  # Training script
    ├── benchmark_cfd.py
    │
    ├── archs/
    │   └── mlp.py
    │
    │
    ├── cfd_xy_2.4-2.8.npz        # Raw face-wise dataset
    └── face_maps_rank0_internal.npz
        # map_x, map_y, map_z, dims_cells, counts
```

---

# Execution

You can execute by:

- Submitting a job using `slurm.sh`
- Running interactively via `srun.bash`

Please configure environment settings to match your system before execution.

Execution is controlled by:

```
Main_pype.py
```

---

## Workflow Stages

1. Computation  
2. Dataset Generation  
3. Subsample  
4. Train  
5. Inference  

If CFD and ML are not integrated:

- You may skip #1 or #3  
- When using CFD-generated data, #1 must be executed  

---

# Subprocess Descriptions

---

## #1 Computation

Script: `exp_openfoam.py`

- Execute when using CFD-generated data  
- Calculation control is performed via `controlDict`  
- When changing the case, port the function:

```
pEqnInnerTest
```

Outer loop is not supported.  
Set:

```
nOuterCorrectors = 1
```

in `fvSolution`.

---

## #2 Dataset Generation

### `make_cfd_dataset.py`

- Saves CFD data into `.npz` format  

### `make_face_maps_from_C.py`

- Creates face maps from dataset  

### `make_cell_maps.py`

- Creates cell maps  
- Adds volume information from OpenFOAM `V` file  

Not mandatory, but required if:

- Performing cell-wise sampling  
- Using physics-informed penalty on discontinuous predictions  

---

## #3 Subsample

### Face-wise Sampling

```
run_sickle_subsample.py
```

- Sampling performed separately for each direction  
- Results are combined afterward  

### Cell-wise Sampling

```
sickle-main/subsample.py
```

- Requires cell map  
- Enables gradient calculation  

---

## #4 Train

Script: `train.py`

- Face-wise sampling:
  - Three input files (x, y, z)  

- Cell-wise sampling:
  - Continuity loss available  
  - Controlled via:

```
--lambda-cont
```

Set to `0` to disable.

---

## #5 Inference

Under construction

---

# Notes

- Designed for verifying inner-loop variable transfer to Redis  
- Integrates CFD + ML pipeline  
- Modular execution via `Main_pype.py`  
- Supports both face-wise and cell-wise sampling workflows  

