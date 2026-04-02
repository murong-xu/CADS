#!/usr/bin/env bash
set -euo pipefail

# ===============================================================
#  Spine-R sequential batch runner
#  Each dataset listed in --list will be processed sequentially
#  with docker run, keeping logs and OK/FAIL markers.
# ===============================================================

IMAGE="spine-r-docker"
HOSTNAME_FIXED="SPINER"

CONFIG_DIR="./config"
CONFIG_FILE="config.yaml"
PROCESSING_ROOT="/mnt/hddb/murong/spine-r-processing"
HOST_MNT_ROOT="/mnt/hddb"

LIST_FILE=""
GPU_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)        LIST_FILE="$2"; shift 2 ;;
    --gpu-id)      GPU_ID="$2"; shift 2 ;;
    --config-dir)  CONFIG_DIR="$2"; shift 2 ;;
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    *)
      echo "[ERR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${LIST_FILE}" || ! -f "${LIST_FILE}" ]]; then
  echo "[ERR] Must provide --list <file> and file must exist." >&2
  exit 1
fi

CONFIG_DIR="$(readlink -f "${CONFIG_DIR}")"
PROCESSING_ROOT="$(readlink -f "${PROCESSING_ROOT}")"
HOST_MNT_ROOT="$(readlink -f "${HOST_MNT_ROOT}")"

LOG_DIR="${PROCESSING_ROOT}/_logs"
MARK_DIR="${PROCESSING_ROOT}/_marks"
mkdir -p "$LOG_DIR" "$MARK_DIR"

DOCKER_ARGS=( --rm --hostname "${HOSTNAME_FIXED}"
  -v "${CONFIG_DIR}:/root/.bonescreen/spine-r"
)

if [[ -n "${GPU_ID}" ]]; then
  DOCKER_ARGS+=( --gpus "device=${GPU_ID}" )
fi

DOCKER_ARGS+=( -v "${HOST_MNT_ROOT}:${HOST_MNT_ROOT}:ro" )

CONFIG_ON_HOST="${CONFIG_DIR}/${CONFIG_FILE}"
CONFIG_IN_CONT="/root/.bonescreen/spine-r/${CONFIG_FILE}"
USE_CFG_ARG=()
if [[ -f "${CONFIG_ON_HOST}" ]]; then
  USE_CFG_ARG=( -c "${CONFIG_IN_CONT}" )
else
  echo "[WARN] Config file not found: ${CONFIG_ON_HOST} — running without -c"
fi

process_one() {
  local ds_name="$1"
  local ds_path="${PROCESSING_ROOT}/${ds_name}"

  if [[ ! -d "$ds_path" ]]; then
    echo "[WARN] Skip ${ds_name}: not found at ${ds_path}"
    return 0
  fi

  local mark_ok="${MARK_DIR}/${ds_name}.OK"
  local mark_fail="${MARK_DIR}/${ds_name}.FAIL"
  local log_file="${LOG_DIR}/${ds_name}.log"

  if [[ -f "$mark_ok" ]]; then
    echo "[SKIP] ${ds_name} already OK"
    return 0
  fi

  echo "==== [START] ${ds_name} $(date) ====" | tee -a "$log_file"

  set +e
  docker run "${DOCKER_ARGS[@]}" \
    -v "${ds_path}:/data" \
    "${IMAGE}" spine-r process -p /data "${USE_CFG_ARG[@]}" \
    2>&1 | tee -a "$log_file"
  status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -eq 0 ]]; then
    touch "$mark_ok"
    rm -f "$mark_fail" 2>/dev/null || true
    echo "==== [OK] ${ds_name} $(date) ====" | tee -a "$log_file"
  else
    touch "$mark_fail"
    echo "==== [FAIL(${status})] ${ds_name} $(date) ====" | tee -a "$log_file"
  fi
}

echo "=== Starting sequential processing ==="
echo "List file      : ${LIST_FILE}"
echo "Processing root: ${PROCESSING_ROOT}"
echo "Config dir     : ${CONFIG_DIR}"
echo "Config file    : ${CONFIG_FILE} $( [[ -f "${CONFIG_ON_HOST}" ]] && echo '[FOUND]' || echo '[MISSING]' )"
echo "GPU binding    : $( [[ -n "${GPU_ID}" ]] && echo "${GPU_ID}" || echo 'NONE' )"
echo

while IFS= read -r ds_name; do
  [[ -z "$ds_name" ]] && continue
  process_one "$ds_name"
done < "${LIST_FILE}"

echo
echo "=== DONE $(date) ==="
echo "OK count  : $(ls ${MARK_DIR}/*.OK 2>/dev/null | wc -l || true)"
echo "FAIL count: $(ls ${MARK_DIR}/*.FAIL 2>/dev/null | wc -l || true)"