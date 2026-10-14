#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# shellcheck source=../env_setup.sh
source "${ROOT_DIR}/env_setup.sh"

export PATH="${HOMED}/bin:${PATH}"

BASE_CASE="${BASE_CASE:-${HOMED}/tmp_runs/native_restart_t1_dt0p01}"
START_TIME="${START_TIME:-1}"
END_TIME="${END_TIME:-10}"
TRAIN_TIME_MIN="${TRAIN_TIME_MIN:-1}"
TRAIN_TIME_MAX="${TRAIN_TIME_MAX:-50}"
RUN_ROOT="${RUN_ROOT:-${HOMED}/tmp_runs/alpha_sweep_t${START_TIME//./p}_to_${END_TIME//./p}_$(date +%Y%m%d_%H%M%S)}"
MODEL="${ML_CNN_MODEL:-}"
BRIDGE="${ML_CNN_BRIDGE:-${HOMED}/smartsim-ml/cpp/build/libcnn_libtorch_bridge.so}"
SOLVER="${OF_SOLVER:-${HOMED}/bin/buoyantBoussinesqPimpleFoam_ML}"
WRITE_INTERVAL="${WRITE_INTERVAL:-100}"  # controlDict writeControl=timeStep; 100 * deltaT=1.0 for dt=0.01
ALPHAS="${ALPHAS:-0 0.25 0.5 1.0}"

numeric_ge() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'
}

numeric_le() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a <= b) }'
}

numeric_gt() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'
}

case_has_time_dir() {
    local case_dir="$1"
    local target="$2"
    find "${case_dir}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
        | awk -v target="${target}" '
            $0 ~ /^([0-9]+)(\.[0-9]+)?$/ {
                if (($0 + 0) == (target + 0)) found = 1
            }
            END { exit !found }
        '
}

has_cnn_alpha() {
    local alpha
    for alpha in ${ALPHAS}; do
        if [[ "${alpha}" != "0" ]]; then
            return 0
        fi
    done
    return 1
}

preflight_runtime_files() {
    if [[ ! -x "${SOLVER}" ]]; then
        echo "[ERROR] Solver executable does not exist or is not executable: ${SOLVER}" >&2
        echo "Build it first with: cd buoyantBoussinesqPimpleFoam_ML && wmake" >&2
        return 1
    fi

    if has_cnn_alpha; then
        if [[ -z "${MODEL}" ]]; then
            echo "[ERROR] ML_CNN_MODEL must be set for nonzero CNN alpha runs." >&2
            return 1
        fi
        if [[ ! -f "${MODEL}" ]]; then
            echo "[ERROR] ML_CNN_MODEL does not exist: ${MODEL}" >&2
            return 1
        fi
        if [[ ! -f "${BRIDGE}" ]]; then
            echo "[ERROR] ML_CNN_BRIDGE does not exist: ${BRIDGE}" >&2
            echo "Build it first with: cd smartsim-ml/cpp && cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch && cmake --build build" >&2
            return 1
        fi
    fi
}

preflight_time_window() {
    if [[ ! -d "${BASE_CASE}" ]]; then
        cat >&2 <<EOF
[ERROR] BASE_CASE does not exist yet: ${BASE_CASE}

Create the native t=${START_TIME} restart case first, for example:
  bash smartsim-ml/prepare_native_t1_restart.sh

Then rerun this alpha sweep script.
EOF
        return 1
    fi

    if ! numeric_ge "${START_TIME}" "${TRAIN_TIME_MIN}" || ! numeric_le "${END_TIME}" "${TRAIN_TIME_MAX}"; then
        cat >&2 <<EOF
[ERROR] Planned validation window is outside the CNN training sampling window.
  training sampling window : t=${TRAIN_TIME_MIN} to t=${TRAIN_TIME_MAX}
  planned validation window: t=${START_TIME} to t=${END_TIME}

This would make the CFD-coupled validation a temporal extrapolation test, not a
matched validation run. Stop here and discuss the intended interpretation before
running. Do not proceed by simply answering yes/no.
EOF
        return 1
    fi

    if ! numeric_gt "${END_TIME}" "${START_TIME}"; then
        echo "[ERROR] END_TIME must be greater than START_TIME. START_TIME=${START_TIME} END_TIME=${END_TIME}" >&2
        return 1
    fi

    if ! case_has_time_dir "${BASE_CASE}" "${START_TIME}"; then
        cat >&2 <<EOF
[ERROR] BASE_CASE does not contain restart fields for START_TIME=${START_TIME}.
  base_case: ${BASE_CASE}

Prepare or select a case directory with fields at the requested start time before
launching the sweep. For example, use a restart case whose time directories
cover the validation start time within the training sampling window.
EOF
        return 1
    fi
}

set_control() {
    local case_dir="$1"
    foamDictionary -entry startFrom -set startTime "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry startTime -set "${START_TIME}" "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry stopAt -set endTime "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry endTime -set "${END_TIME}" "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry writeControl -set timeStep "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry writeInterval -set "${WRITE_INTERVAL}" "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry purgeWrite -set 0 "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry functions.pEqnInnerTest.enabled -set false "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry functions.pUPhiTest.enabled -set false "${case_dir}/system/controlDict" >/dev/null
}

prepare_case() {
    local case_dir="$1"
    cp -a "${BASE_CASE}" "${case_dir}"
    set_control "${case_dir}"
    rm -rf "${case_dir}"/processor*

    if [[ "${START_TIME}" == "0" || "${START_TIME}" == "0.0" ]]; then
        rm -rf "${case_dir}/constant/polyMesh"
        blockMesh -case "${case_dir}" > "${case_dir}/log.blockMesh" 2>&1
        renumberMesh -case "${case_dir}" -overwrite > "${case_dir}/log.renumberMesh" 2>&1
        postProcess -func writeCellCentres -case "${case_dir}" > "${case_dir}/log.writeCellCentres" 2>&1
        postProcess -func writeCellVolumes -case "${case_dir}" > "${case_dir}/log.writeCellVolumes" 2>&1
        (cd "${case_dir}" && python initialize_T2.py) > "${case_dir}/log.initialize_T2" 2>&1
        (cd "${case_dir}" && python initialize_U2.py) > "${case_dir}/log.initialize_U2" 2>&1
    else
        postProcess -func writeCellCentres -case "${case_dir}" > "${case_dir}/log.writeCellCentres" 2>&1
        postProcess -func writeCellVolumes -case "${case_dir}" > "${case_dir}/log.writeCellVolumes" 2>&1
    fi
}

alpha_label() {
    local alpha="$1"
    if [[ "${alpha}" == "0" ]]; then
        printf "baseline"
    else
        printf "alpha%s" "${alpha//./p}"
    fi
}

run_case() {
    local alpha="$1"
    local label
    label=$(alpha_label "${alpha}")
    local case_dir="${RUN_ROOT}/${label}"

    echo "[INFO] Preparing ${label}: ${case_dir}"
    prepare_case "${case_dir}"

    echo "[INFO] Running ${label} alpha=${alpha}"
    if [[ "${alpha}" == "0" ]]; then
        (
            unset ML_CNN_MODEL ML_CNN_BRIDGE ML_USE_CNN_FLUX ML_CNN_FLUX_ALPHA
            "${SOLVER}" -case "${case_dir}"
        ) > "${case_dir}/log.solver" 2>&1
    else
        (
            export ML_CNN_MODEL="${MODEL}"
            export ML_CNN_BRIDGE="${BRIDGE}"
            export ML_USE_CNN_FLUX=1
            export ML_CNN_FLUX_ALPHA="${alpha}"
            "${SOLVER}" -case "${case_dir}"
        ) > "${case_dir}/log.solver" 2>&1
    fi
    echo "[INFO] Finished ${label}"
}

preflight_time_window
preflight_runtime_files

mkdir -p "${RUN_ROOT}"

{
    echo "run_root=${RUN_ROOT}"
    echo "base_case=${BASE_CASE}"
    echo "training_time_min=${TRAIN_TIME_MIN}"
    echo "training_time_max=${TRAIN_TIME_MAX}"
    echo "start_time=${START_TIME}"
    echo "model=${MODEL}"
    echo "bridge=${BRIDGE}"
    echo "solver=${SOLVER}"
    echo "end_time=${END_TIME}"
    echo "write_interval=${WRITE_INTERVAL}"
    echo "alphas=${ALPHAS}"
} > "${RUN_ROOT}/run_config.txt"

for alpha in ${ALPHAS}; do
    run_case "${alpha}"
done

echo "[INFO] Alpha sweep complete: ${RUN_ROOT}"
