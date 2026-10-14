#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

load_module_if_available() {
    local module_name="$1"
    if command -v module >/dev/null 2>&1; then
        module load "$module_name"
    fi
}

# Optional site modules. Set ENV_SETUP_MODULES from the job script when needed,
# for example: "miniforge3/23.11.0-0 mpi/openmpi-x86_64 cuda/12.6".
DEFAULT_MODULES="${DEFAULT_MODULES:-}"
ENV_SETUP_MODULES="${ENV_SETUP_MODULES:-$DEFAULT_MODULES}"

# Optional module setup:
#   export ENV_SETUP_MODULES="miniforge3/23.11.0-0 mpi/openmpi-x86_64 cuda/12.6"
if [[ -n "${ENV_SETUP_MODULES:-}" ]]; then
    for mod in ${ENV_SETUP_MODULES}; do
        load_module_if_available "$mod"
    done
fi

if [[ -n "${ENV_SETUP_CONDA_MODULE:-}" ]]; then
    load_module_if_available "${ENV_SETUP_CONDA_MODULE}"
fi

if [[ -z "${ENV_SETUP_CONDA_ENV:-}" ]]; then
    DEFAULT_CONDA_ENV_NAME="${DEFAULT_CONDA_ENV_NAME:-TGV_test}"
    if [[ -d "${HOME}/.conda/envs/${DEFAULT_CONDA_ENV_NAME}" ]]; then
        ENV_SETUP_CONDA_ENV="${DEFAULT_CONDA_ENV_NAME}"
    fi
fi

if [[ -n "${ENV_SETUP_CONDA_ENV:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    conda activate "${ENV_SETUP_CONDA_ENV}"
fi

export HOMED="${HOMED:-$SCRIPT_DIR}"
export DATAD="${DATAD:-${HOMED}/cases/active/data}"
export WORKD="${WORKD:-${HOMED}/smartsim-ml}"
export SICKLE_ROOT="${SICKLE_ROOT:-${HOMED}/sickle-of}"
export CASED="${CASED:-${HOMED}/cases/active/TGStratifiedLES_ML}"
INITIAL_FOAM_RUN="${FOAM_RUN:-${CASED}}"

mkdir -p "${DATAD}" "${HOMED}/out"

# OpenFOAM environment:
#   export OPENFOAM_BASHRC=/path/to/OpenFOAM-v2312/etc/bashrc
if [[ -n "${OPENFOAM_BASHRC:-}" ]]; then
    # OpenFOAM's bashrc is not strict-shell clean and may emit non-fatal MPI warnings.
    set +e +u
    # shellcheck disable=SC1090
    source "${OPENFOAM_BASHRC}"
    openfoam_rc=$?
    set -e -u
    if [[ ${openfoam_rc} -ne 0 ]]; then
        echo "[WARN] Sourcing OPENFOAM_BASHRC returned ${openfoam_rc}; continuing with the environment that was set up." >&2
    fi
elif [[ -z "${WM_PROJECT_DIR:-}" ]]; then
    echo "[WARN] OPENFOAM_BASHRC is not set; assuming OpenFOAM is already loaded." >&2
fi

export FOAM_RUN="${INITIAL_FOAM_RUN}"

export WM_PROJECT_USER_DIR="${WM_PROJECT_USER_DIR:-$FOAM_RUN}"
export WM_MPLIB="${WM_MPLIB:-${FOAM_MPI_FLAVOR:-}}"
export WM_CC="${WM_CC:-cc}"
export WM_CXX="${WM_CXX:-c++}"

SMARTREDIS_CPP_LIB_DEFAULT="${SMARTREDIS_CPP_LIB_DEFAULT:-${HOMED}/smartredis/install/lib64}"
if [[ -z "${SMARTREDIS_CPP_LIB:-}" && -d "${SMARTREDIS_CPP_LIB_DEFAULT}" ]]; then
    export SMARTREDIS_CPP_LIB="${SMARTREDIS_CPP_LIB_DEFAULT}"
fi

append_ld_library_path() {
    local path="$1"
    if [[ -n "${path}" ]] && [[ -d "${path}" ]]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${path}:"*) ;;
            *) export LD_LIBRARY_PATH="${path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
        esac
    fi
}

append_ld_library_path "${FOAM_LIBBIN:-}"
append_ld_library_path "${FOAM_USER_LIBBIN:-}"
append_ld_library_path "${SMARTREDIS_CPP_LIB:-}"

export NODES="${NODES:-${SLURM_JOB_NUM_NODES:-1}}"
export SMARTSIM_LAUNCHER="${SMARTSIM_LAUNCHER:-slurm}"
export SMARTSIM_DB_NODES="${SMARTSIM_DB_NODES:-1}"
export SMARTSIM_INTERFACES="${SMARTSIM_INTERFACES:-lo}"
export OF_LAUNCHER="${OF_LAUNCHER:-srun}"
export OF_NODES="${OF_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
export OF_NTASKS="${OF_NTASKS:-${SLURM_NTASKS:-1}}"
export OF_TASKS_PER_NODE="${OF_TASKS_PER_NODE:-${SLURM_NTASKS_PER_NODE:-${OF_NTASKS}}}"

if [[ -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)}"
else
    export MASTER_ADDR="${MASTER_ADDR:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
    export MASTER_ADDR="127.0.0.1"
fi

export MASTER_PORT="${MASTER_PORT:-3442}"

if [[ -n "${NCCL_SOCKET_IFNAME_DEFAULT:-}" ]] && [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_DEFAULT}"
fi

if [[ -n "${ROCM_PATH:-}" || -d /opt/rocm ]]; then
    export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-/tmp/${USER}/miopen-cache}"
    export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-${MIOPEN_USER_DB_PATH}}"
    rm -rf "${MIOPEN_USER_DB_PATH}"
    mkdir -p "${MIOPEN_USER_DB_PATH}"
fi
