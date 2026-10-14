# OpenFOAM CNN Closed-Loop Solver

This repository contains a modified OpenFOAM `buoyantBoussinesqPimpleFoam`
solver that can call a TorchScript 3D CNN during the pressure-correction loop
and use the predicted pressure-equation flux (`pEqnFlux`) in the solver.

The review focus is the solver-side CNN integration:

- build the OpenFOAM solver with the CNN hooks;
- load a TorchScript CNN through a LibTorch bridge;
- construct the 18-channel block input from OpenFOAM fields;
- map the CNN block output back to OpenFOAM faces;
- blend, limit, project, or diagnose the CNN-predicted pressure flux.

Large training datasets and generated OpenFOAM run directories are not intended
to be committed.  The repository keeps the preprocessing code, model metadata,
solver integration code, and diagnostic scripts needed to review the workflow.

## Workflow Overview

The project has four separate stages:

```text
1. Teacher-data generation
   OpenFOAM + SmartSim/Redis utilities
   -> raw face/cell NPZ data

2. CNN dataset construction
   raw NPZ + OpenFOAM face/cell maps
   -> dense block16 + halo2 CNN dataset

3. CNN training and export
   dense block dataset
   -> PyTorch checkpoint
   -> TorchScript model

4. Closed-loop solver inference
   OpenFOAM solver + LibTorch bridge + TorchScript model
   -> pEqnFlux_cnn used inside pEqn.H
```

SmartSim/Redis is kept for stage 1, where OpenFOAM fields are collected into
teacher-data files for future retraining.  The closed-loop solver in stage 4
does not call SmartSim or Redis; it loads the TorchScript CNN through LibTorch.

## Main Components

```text
buoyantBoussinesqPimpleFoam_ML/
  OpenFOAM solver with CNN input/output hooks and pEqnFlux usage.

smartsim-ml/archs/cnn.py
smartsim-ml/train_cnn.py
smartsim-ml/eval_cnn_block.py
smartsim-ml/export_cnn_torchscript.py
  CNN architecture, training, evaluation, and TorchScript export.

smartsim-ml/cpp/
  LibTorch bridge loaded by the OpenFOAM solver at runtime.

smartsim-ml/utils/make_dense_block_dataset.py
smartsim-ml/utils/build_block16_dataset.sh
  Dataset conversion code for dense 3D CNN block samples.

smartsim-ml/exp_openfoam.py
smartsim-ml/of_common.py
smartsim-ml/utils/make_cfd_dataset.py
  SmartSim/Redis-based utilities used for generating or regenerating raw
  OpenFOAM teacher-data NPZ files.

smartsim-ml/run_closed_loop_alpha_sweep.sh
smartsim-ml/extend_closed_loop_alpha_sweep.sh
smartsim-ml/prepare_native_t1_restart.sh
  Closed-loop run helpers.

smartsim-ml/utils/evaluate_closed_loop_timeseries.py
smartsim-ml/utils/diagnose_pEqnFlux_closed_loop.py
smartsim-ml/utils/compare_flux_divergence.py
smartsim-ml/utils/compare_face_correspondence.py
  Post-run diagnostics and evaluation tools.
```

## CNN Interface

The exported CNN used by the solver has the following interface:

```text
input  : (B, 18, 20, 20, 20)
output : (B,  6, 16, 16, 16)
```

The input is physical, unnormalized OpenFOAM data.  The 18 channels are arranged
as six face directions times three per-face features.  The output is physical
`pEqnFlux` on the center-cropped 16x16x16 block core.

The exported model metadata is stored beside the TorchScript artifact, for
example:

```text
runs/cnn_w64_d5_lc200_b16_mse_mask1p0_steps1200_20260908/best_inference.json
```

The large source dataset is excluded from the review package because of size.
Only the schema, preprocessing code, and model metadata are kept in the source
tree.

## Stage 1: Teacher-Data Generation

Teacher-data generation is the stage where SmartSim/Redis can be used.  The
goal is to collect OpenFOAM face fields and write raw NPZ files that can later
be converted into CNN block datasets.

Relevant files:

```text
smartsim-ml/exp_openfoam.py
smartsim-ml/of_common.py
smartsim-ml/utils/make_cfd_dataset.py
smartsim-ml/utils/make_cfd_dataset_from_functionObject.py
smartsim-ml/utils/make_face_maps_from_C.py
smartsim-ml/utils/make_cell_maps.py
```

This stage is not part of the runtime CNN closed-loop path, but it is retained
so the model can be retrained from regenerated OpenFOAM data.

## Stage 2: CNN Dataset Construction

The raw OpenFOAM NPZ data is converted into dense local blocks used by the 3D
CNN.  The current model family uses block16 samples with halo2 context.

Relevant files:

```text
smartsim-ml/utils/make_dense_block_dataset.py
smartsim-ml/utils/build_block16_dataset.sh
```

The block dataset has the same logical interface as the exported model:

```text
input core context : 18 channels on 20x20x20 blocks
supervised output : 6 pEqnFlux face directions on 16x16x16 cores
```

## Stage 3: CNN Training And Export

Training reads the dense block dataset directly with PyTorch.  SmartSim is not
used during this stage.

Relevant files:

```text
smartsim-ml/archs/cnn.py
smartsim-ml/train_cnn.py
smartsim-ml/eval_cnn_block.py
smartsim-ml/export_cnn_torchscript.py
smartsim-ml/run_cnn_3d_block_compare.sh
smartsim-ml/run_cnn_probe_inference.py
```

The export step wraps normalization and output inverse-normalization into the
TorchScript module, so the OpenFOAM solver passes physical, unnormalized input.

## Solver-Side Flow

At each relevant solver timestep:

1. `buoyantBoussinesqPimpleFoam_ML.C` computes the native pressure-equation
   fields and includes the CNN hooks before `pEqn.H`.
2. `writeCnnInput.H` can optionally write a CSV probe of the 18-channel input.
3. `writeCnnFlux.H` loads the LibTorch bridge, runs block inference, maps the
   CNN output back to OpenFOAM faces, and writes `pEqnFlux_cnn` plus a mask.
4. `pEqn.H` can use the CNN flux as a replacement, blend, residual correction,
   or initial pressure guess depending on environment variables.

Important runtime switches include:

```text
ML_CNN_MODEL
ML_CNN_BRIDGE
ML_CNN_DEVICE
ML_USE_CNN_FLUX
ML_CNN_FLUX_ALPHA
ML_CNN_INFERENCE_INTERVAL
ML_CNN_FLUX_LIMIT_REL
ML_CNN_FLUX_LIMIT_ABS
ML_CNN_FLUX_LIMIT_FLOOR
ML_CNN_FLUX_FALLBACK_NONFINITE
ML_CNN_FLUX_DIAG
ML_CNN_FLUX_PROJECT_CONSERVATIVE
ML_CNN_FLUX_RESIDUAL_CORRECTION
ML_CNN_FLUX_RESIDUAL_BLEND
ML_CNN_INITIAL_GUESS
ML_CNN_INITIAL_GUESS_BLEND
ML_CNN_INITIAL_GUESS_SCALE
ML_WRITE_CNN_INPUT
ML_WRITE_CNN_FLUX_FIELDS
ML_WRITE_CNN_INITIAL_GUESS_FIELDS
ML_WRITE_NATIVE_FLUX
ML_WRITE_USED_FLUX
ML_WRITE_TIMING
```

## Stage 4: Closed-Loop Solver Inference

The closed-loop path uses the OpenFOAM solver, a LibTorch bridge, and the
TorchScript CNN model.  It does not require a SmartSim database.

Relevant files:

```text
buoyantBoussinesqPimpleFoam_ML/writeCnnInput.H
buoyantBoussinesqPimpleFoam_ML/writeCnnFlux.H
buoyantBoussinesqPimpleFoam_ML/pEqn.H
smartsim-ml/cpp/cnn_libtorch_bridge.cpp
smartsim-ml/run_closed_loop_alpha_sweep.sh
smartsim-ml/extend_closed_loop_alpha_sweep.sh
smartsim-ml/prepare_native_t1_restart.sh
```

## Build Notes

OpenFOAM itself is an external dependency and is not vendored in this
repository.  Use an existing OpenFOAM installation on the target machine, source
its environment first, then build only the modified solver stored here.

For example:

```bash
source /path/to/OpenFOAM-v2312/etc/bashrc
export HOMED=/path/to/CNN_test
```

Build the OpenFOAM solver with `wmake` from the solver directory:

```bash
cd buoyantBoussinesqPimpleFoam_ML
wmake
```

`wmake` is provided by OpenFOAM.  The current `Make/files` writes the solver
binary to `${HOMED}/bin/buoyantBoussinesqPimpleFoam_ML`, so `HOMED` must point
to the repository root before building.  If a site prefers the standard
OpenFOAM user binary directory, change `Make/files` to use
`$(FOAM_USER_APPBIN)/buoyantBoussinesqPimpleFoam_ML` instead.

Build the LibTorch bridge with CMake:

```bash
cd smartsim-ml/cpp
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build
```

The solver loads the bridge specified by `ML_CNN_BRIDGE`.  If the variable is
not set, it falls back to `libcnn_libtorch_bridge.so` on the runtime library
path.

## Example Closed-Loop Run

The run helper expects an existing restart case and a TorchScript model:

```bash
export ML_CNN_MODEL=/path/to/best_inference.ts
export ML_CNN_BRIDGE=/path/to/libcnn_libtorch_bridge.so
export BASE_CASE=/path/to/native_restart_case
export START_TIME=1
export END_TIME=10
export ALPHAS="0 0.25 0.5 1.0"

bash smartsim-ml/run_closed_loop_alpha_sweep.sh
```

For a single solver run, the key variables are:

```bash
export ML_CNN_MODEL=/path/to/best_inference.ts
export ML_CNN_BRIDGE=/path/to/libcnn_libtorch_bridge.so
export ML_USE_CNN_FLUX=1
export ML_CNN_FLUX_ALPHA=0.25

buoyantBoussinesqPimpleFoam_ML -case /path/to/case
```

## Evaluation

Closed-loop runs can be compared against a native baseline with:

```bash
python smartsim-ml/utils/evaluate_closed_loop_timeseries.py \
  --case baseline=/path/to/baseline \
  --case cnn=/path/to/cnn_case \
  --out-dir outputs/closed_loop_timeseries
```

Flux-level diagnostics can be generated with:

```bash
python smartsim-ml/utils/diagnose_pEqnFlux_closed_loop.py \
  --case /path/to/cnn_case \
  --out-dir outputs/pEqnFlux_diagnostics
```

Additional checks for face correspondence and flux divergence are available in
`smartsim-ml/utils/`.

## Repository Hygiene

The following are source or review artifacts and should generally be kept:

```text
buoyantBoussinesqPimpleFoam_ML/
smartsim-ml/archs/cnn.py
smartsim-ml/train_cnn.py
smartsim-ml/eval_cnn_block.py
smartsim-ml/export_cnn_torchscript.py
smartsim-ml/cpp/
smartsim-ml/exp_openfoam.py
smartsim-ml/of_common.py
smartsim-ml/utils/make_cfd_dataset.py
smartsim-ml/utils/make_cfd_dataset_from_functionObject.py
smartsim-ml/run_closed_loop_alpha_sweep.sh
smartsim-ml/extend_closed_loop_alpha_sweep.sh
smartsim-ml/utils/make_dense_block_dataset.py
smartsim-ml/utils/evaluate_closed_loop_timeseries.py
smartsim-ml/utils/diagnose_pEqnFlux_closed_loop.py
```

The following are generated or large local artifacts and are normally excluded
from GitHub:

```text
cases/
tmp_runs/
backups/
notes/
smartsim-ml/.smartsim/
smartsim-ml/orchestrator*/
smartsim-ml/cpp/build/
buoyantBoussinesqPimpleFoam_ML/Make/linux64GccDPInt32Opt/
```
