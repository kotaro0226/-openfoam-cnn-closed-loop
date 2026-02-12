# EXAMPLES

This document provides detailed examples for using SICKLE in various scenarios, including local testing, Frontier runs, parallel execution, and training.

## Table of Contents

- [Subsampling Examples](#subsampling-examples)
  - [Local Testing with Random and Maxent Methods](#local-testing-with-random-and-maxent-methods)
  - [Frontier Testing](#frontier-testing)
- [Training Examples](#training-examples)
- [Parallel Execution Examples](#parallel-execution-examples)
- [Flow Over Cylinder Example](#flow-over-cylinder-example)
- [Comparing Methods](#comparing-methods)

## Subsampling Examples

### Local Testing with Random and Maxent Methods

- **Random Subsampling on a Laptop:**

  ```bash
  python subsample.py config/laptop/default.yaml -m random --target drag -ns 540
  ```

- **Maxent Subsampling on a Laptop:**

  ```bash
  python subsample.py config/laptop/default.yaml -m maxent --target drag -ns 540 -cv p
  ```

### Frontier Testing

- **Maxent Subsampling on Frontier:**

  Use the YAML configuration file to provide defaults, with optional command-line switches:
  
  ```bash
  python subsample.py config/OF/default.yaml --plot
  ```

- **Full Subsampling on Frontier:**

  ```bash
  python subsample.py config/OF/default.yaml -m full --plot -ns 100
  ```

## Training Examples

- **Basic Training Using a YAML File:**

  ```bash
  python -u train.py config/OF/default.yaml
  ```

- **Overriding Specific Parameters (e.g., Epochs):**

  ```bash
  python -u train.py config/OF/default.yaml --epochs 3000
  ```

## Parallel Execution Examples

### Using MPI for Parallel Subsampling

- **OpenFOAM Dataset with Random and Maxent Methods:**

  ```bash
  srun -n 4 python -u subsample-mpi.py config/OF/default.yaml -m random --target drag -ns 540
  srun -n 4 python -u subsample-mpi.py config/OF/default.yaml -m maxent --target drag -ns 540
  ```

- **Taylor Green Dataset Example (with Specific Timesteps):**

  ```bash
  OPENBLAS_NUM_THREADS=4 srun -n 4 python -u subsample-mpi.py config/OF/default.yaml -m maxent --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64
  ```

## Flow Over Cylinder Example

- **Subsampling with the Flow Over Cylinder Case:**

  ```bash
  python subsample.py config/OF/default.yaml -m maxent --target drag -ns 1080 -nc 20 -cv wz
  python subsample.py config/OF/default.yaml -m uips --target drag -ns 1080 -nc 20 --plot
  ```

## Comparing Methods

- **Comparing Subsampling Distributions:**

  Generate a histogram to compare different subsampling methods:
  
  ```bash
  python compare_methods.py config/OF/default.yaml --target drag -ns 1080 -nc 20 -cv wz
  ```

- **Additional Comparison Example:**

  ```bash
  python compare_methods.py config/OF/default.yaml --plot -ns 100
  ```

---

*Note:* Replace configuration file paths (e.g., `config/OF/default.yaml` or `config/laptop/default.yaml`) with the actual paths used in your environment.
