#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# shellcheck source=../env_setup.sh
source "${ROOT_DIR}/env_setup.sh"

export PATH="${HOMED}/bin:${PATH}"

BASE_CASE="${BASE_CASE:-${HOMED}/cases/active/TGStratifiedLES_ML_dt0p01_et50}"
RESTART_CASE="${RESTART_CASE:-${HOMED}/tmp_runs/native_restart_t1_dt0p01}"
SOLVER="${OF_SOLVER:-${HOMED}/bin/buoyantBoussinesqPimpleFoam_ML}"
RESTART_TIME="${RESTART_TIME:-1}"
WRITE_INTERVAL="${WRITE_INTERVAL:-100}"  # dt=0.01 -> write t=1 after 100 steps
OVERWRITE="${OVERWRITE:-0}"

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

if [[ ! -d "${BASE_CASE}" ]]; then
    echo "[ERROR] BASE_CASE does not exist: ${BASE_CASE}" >&2
    exit 1
fi

if [[ -e "${RESTART_CASE}" ]]; then
    if [[ "${OVERWRITE}" != "1" ]]; then
        cat >&2 <<EOF
[ERROR] RESTART_CASE already exists: ${RESTART_CASE}
Set OVERWRITE=1 only if you intentionally want to replace this generated
restart case.
EOF
        exit 1
    fi
    rm -rf "${RESTART_CASE}"
fi

if ! numeric_gt "${RESTART_TIME}" "0"; then
    echo "[ERROR] RESTART_TIME must be greater than 0. Got ${RESTART_TIME}" >&2
    exit 1
fi

echo "[INFO] Creating native restart case"
echo "[INFO] base_case=${BASE_CASE}"
echo "[INFO] restart_case=${RESTART_CASE}"
echo "[INFO] restart_time=${RESTART_TIME}"

cp -a "${BASE_CASE}" "${RESTART_CASE}"

# Keep only initial-condition directories before generating the native restart.
find "${RESTART_CASE}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' \
    | awk '$0 ~ /^([0-9]+)(\.[0-9]+)?$/ && ($0 + 0) != 0 { print }' \
    | while IFS= read -r time_dir; do
        rm -rf "${RESTART_CASE}/${time_dir}"
    done

rm -rf "${RESTART_CASE}"/processor*

foamDictionary -entry startFrom -set startTime "${RESTART_CASE}/system/controlDict" >/dev/null
foamDictionary -entry startTime -set 0 "${RESTART_CASE}/system/controlDict" >/dev/null
foamDictionary -entry stopAt -set endTime "${RESTART_CASE}/system/controlDict" >/dev/null
foamDictionary -entry endTime -set "${RESTART_TIME}" "${RESTART_CASE}/system/controlDict" >/dev/null
foamDictionary -entry writeControl -set timeStep "${RESTART_CASE}/system/controlDict" >/dev/null
foamDictionary -entry writeInterval -set "${WRITE_INTERVAL}" "${RESTART_CASE}/system/controlDict" >/dev/null
foamDictionary -entry purgeWrite -set 0 "${RESTART_CASE}/system/controlDict" >/dev/null
foamDictionary -entry functions.pEqnInnerTest.enabled -set false "${RESTART_CASE}/system/controlDict" >/dev/null
foamDictionary -entry functions.pUPhiTest.enabled -set false "${RESTART_CASE}/system/controlDict" >/dev/null

(
    unset ML_CNN_MODEL ML_CNN_BRIDGE ML_USE_CNN_FLUX ML_CNN_FLUX_ALPHA
    "${SOLVER}" -case "${RESTART_CASE}"
) > "${RESTART_CASE}/log.native_to_t${RESTART_TIME//./p}" 2>&1

if ! case_has_time_dir "${RESTART_CASE}" "${RESTART_TIME}"; then
    cat >&2 <<EOF
[ERROR] Native restart run finished but did not write t=${RESTART_TIME}.
Check ${RESTART_CASE}/log.native_to_t${RESTART_TIME//./p}
EOF
    exit 1
fi

echo "[INFO] Native restart is ready: ${RESTART_CASE}"
