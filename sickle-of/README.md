# SICKLE

**SICKLE** (Sparse Intelligent Curation frameworK for Learning Efficiency) is a tool designed to extract data with the highest probabilistic information content, thereby reducing the cost of training large models.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Advanced Topics](#advanced-topics)
- [SLURM Scripts](#slurm-scripts)
- [License](#license)
- [Additional Resources](#additional-resources)

## Overview

SICKLE helps "separate the wheat from the chaff" by using various subsampling methods (e.g., maxent, random) to extract the most informative data segments. It supports both training and testing modes and can be run on different systems (e.g., local laptop, Frontier).

## Installation

1. **Download/Clone Repository:**  
   Clone the repository to your local machine.

2. **Local Environment Setup:**  
   - Activate the required Python virtual environment:
     ```bash
     source /path/to/venv/bin/activate
     ```
   - Load required modules:
     ```bash
     module load cray-python/3.10.10 rocm
     ```

3. **Frontier Environment Setup:**  
   On Frontier, a simple environment setup is provided. Instead of running the above two steps separately, simply execute:
   ```bash
   . contrib/env-frontier
   ```
   This single command sets up both the virtual environment and loads the required modules.


## Usage

SICKLE is run from the command line. It supports both direct command-line specification of parameters and YAML configuration files. When a YAML configuration file is provided, its settings are used as defaults; any additional command-line switches will override the YAML values.

### Subsampling

Instead of specifying every parameter on the command line, you can use a YAML configuration file. For example:

- **Using the YAML file only:**
  ```bash
  python subsample.py config/OF/default.yaml
  ```
- **Overriding specific parameters:**
  ```bash
  python subsample.py config/OF/default.yaml --plot
  ```

### Training

Similarly, for training you can:

- **Using the YAML file only:**
  ```bash
  python -u train.py config/OF/default.yaml
  ```
- **Overriding specific parameters (e.g., epochs):**
  ```bash
  python -u train.py config/OF/default.yaml --epochs 3000
  ```

For a complete list of options, see the [`args.py`](./args.py) file.

## Configuration

SICKLE uses YAML configuration files to set parameters. All configurations are flattened, which means they don't need to be nested under a hierarchy. Below is an example configuration snippet:

```yaml
shared:
  dims: 3
  dtype: sst-binary
  noseed: true
  input_vars: [u, v, w, r]
  output_vars: [p, pv]
  cluster_var: [p, pv]
  nx: 514
  ny: 512
  nz: 256
  gravity: z
  fileprefix: "SST-P1-H{hypercubes}-cubes{num_hypercubes}-X{method}-ns{num_samples}-window{window}"

subsample:
  hypercubes: maxent
  num_hypercubes: 32
  method: maxent  # or random
  path: /path/to/data/
  num_samples: 3277
  num_clusters: 20
  nxsl: 32
  nysl: 32
  nzsl: 32

train:
  epochs: 1000
  batch: 16
  target: p_full
  window: 1
  arch: MLP_transformer
  sequence: true
```

*Note:* Adjust the YAML details as needed for your use cases.

## Examples

Detailed examples (including commands for testing on laptops, Frontier, parallel runs, and flow over cylinder cases) are provided in a separate file: [EXAMPLES.md](./EXAMPLES.md).

## Advanced Topics

- **Parallel Processing:**  
  SICKLE supports parallel execution (e.g., using `srun` for MPI-based tests).
- **Mixed Precision and Scalability:**  
  Options such as mixed precision (`amp`) and network architectures like `MLP_transformer` are available.
- **Integration with PyTorch:**  
  See the [PyTorch Frontier Documentation](https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html) for further details on the training environment.

## SLURM Scripts

SLURM Scripts and YAML config files are in sickle-contrib repo. Run:

```
git submodule update --init --recursive
```

For users running on clusters with SLURM, sample scripts are available in the `scripts` directory. These scripts are set up to perform both a subsampling and a training event in a single SLURM session. **Before using them, make sure to modify the scripts to specify your account information.** For example, you can submit a job with:

```bash
sbatch contrib/slurm-scripts/slurm.sh
```

## License

This project is licensed under the **MIT License**.

For more details, see the [LICENSE](./LICENSE) file.

## Additional Resources

- [PyTorch Frontier Documentation](https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html)
- For further details on configuration and command-line options, refer to the inline comments in [`args.py`](./args.py).
