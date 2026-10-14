#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# shellcheck source=../env_setup.sh
source "${ROOT_DIR}/env_setup.sh"

export PATH="${HOMED}/bin:${PATH}"

RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to an existing alpha sweep directory}"
MODEL="${ML_CNN_MODEL:-}"
BRIDGE="${ML_CNN_BRIDGE:-${HOMED}/smartsim-ml/cpp/build/libcnn_libtorch_bridge.so}"
SOLVER="${OF_SOLVER:-${HOMED}/bin/buoyantBoussinesqPimpleFoam_ML}"
END_TIME="${END_TIME:-0.5}"
TRAIN_TIME_MIN="${TRAIN_TIME_MIN:-1}"
TRAIN_TIME_MAX="${TRAIN_TIME_MAX:-50}"
WRITE_INTERVAL="${WRITE_INTERVAL:-10}"
ALPHAS="${ALPHAS:-0 0.0125 0.025 0.05 0.1}"

numeric_ge() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'
}

numeric_le() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a <= b) }'
}

numeric_gt() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'
}

latest_numeric_time() {
    local case_dir="$1"
    find "${case_dir}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
        | awk '
            $0 ~ /^([0-9]+)(\.[0-9]+)?$/ {
                if (!seen || ($0 + 0) > max_time) {
                    max_time = $0 + 0
                    seen = 1
                }
            }
            END {
                if (!seen) exit 1
                print max_time
            }
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

preflight_case_time_window() {
    local case_dir="$1"
    local start_time
    if ! start_time=$(latest_numeric_time "${case_dir}"); then
        echo "[ERROR] Could not determine latest numeric time in ${case_dir}" >&2
        return 1
    fi

    if ! numeric_ge "${start_time}" "${TRAIN_TIME_MIN}" || ! numeric_le "${END_TIME}" "${TRAIN_TIME_MAX}"; then
        cat >&2 <<EOF
[ERROR] Planned extension window is outside the CNN training sampling window.
  training sampling window : t=${TRAIN_TIME_MIN} to t=${TRAIN_TIME_MAX}
  planned extension window : t=${start_time} to t=${END_TIME}
  case directory           : ${case_dir}

This would make the CFD-coupled validation a temporal extrapolation test, not a
matched validation run. Stop here and discuss the intended interpretation before
running. Do not proceed by simply answering yes/no.
EOF
        return 1
    fi

    if ! numeric_gt "${END_TIME}" "${start_time}"; then
        echo "[ERROR] END_TIME must be greater than latestTime. latestTime=${start_time} END_TIME=${END_TIME} case=${case_dir}" >&2
        return 1
    fi
}

set_control() {
    local case_dir="$1"
    foamDictionary -entry startFrom -set latestTime "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry stopAt -set endTime "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry endTime -set "${END_TIME}" "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry writeControl -set timeStep "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry writeInterval -set "${WRITE_INTERVAL}" "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry purgeWrite -set 0 "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry functions.pEqnInnerTest.enabled -set false "${case_dir}/system/controlDict" >/dev/null
    foamDictionary -entry functions.pUPhiTest.enabled -set false "${case_dir}/system/controlDict" >/dev/null
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
    if [[ ! -d "${case_dir}" ]]; then
        echo "[ERROR] Missing case directory: ${case_dir}" >&2
        return 1
    fi

    preflight_case_time_window "${case_dir}"
    set_control "${case_dir}"
    echo "[INFO] Extending ${label} alpha=${alpha} to endTime=${END_TIME}"
    if [[ "${alpha}" == "0" ]]; then
        (
            unset ML_CNN_MODEL ML_CNN_BRIDGE ML_USE_CNN_FLUX ML_CNN_FLUX_ALPHA
            "${SOLVER}" -case "${case_dir}"
        ) > "${case_dir}/log.extend_${END_TIME//./p}" 2>&1
    else
        (
            export ML_CNN_MODEL="${MODEL}"
            export ML_CNN_BRIDGE="${BRIDGE}"
            export ML_USE_CNN_FLUX=1
            export ML_CNN_FLUX_ALPHA="${alpha}"
            "${SOLVER}" -case "${case_dir}"
        ) > "${case_dir}/log.extend_${END_TIME//./p}" 2>&1
    fi
    echo "[INFO] Finished ${label}"
}

preflight_runtime_files

{
    echo "run_root=${RUN_ROOT}"
    echo "training_time_min=${TRAIN_TIME_MIN}"
    echo "training_time_max=${TRAIN_TIME_MAX}"
    echo "model=${MODEL}"
    echo "bridge=${BRIDGE}"
    echo "solver=${SOLVER}"
    echo "extended_end_time=${END_TIME}"
    echo "write_interval=${WRITE_INTERVAL}"
    echo "alphas=${ALPHAS}"
} >> "${RUN_ROOT}/run_config.txt"

for alpha in ${ALPHAS}; do
    run_case "${alpha}"
done

echo "[INFO] Extension complete: ${RUN_ROOT}"
