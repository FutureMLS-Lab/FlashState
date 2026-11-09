#!/bin/bash
# One measurement without Docker: the same steps as run_point.sh, with the engine, the Mooncake master and the client
# installed on this machine (for example from source, see docs/install.md).
#
#   benchmarks/run_local.sh CONFIG ENGINE SESSIONS OUT_DIR
#
# Environment: MODEL_DIR (required), CUDA_VISIBLE_DEVICES (the server's two GPUs), PORT (default 8000), VLLM (the
# command that runs vLLM's CLI for this ENGINE, default "vllm"), MOONCAKE_MASTER (default mooncake_master), AIPERF
# (the replay client, default aiperf), HF_HOME (the dataset cache, see tools/fetch_dataset.sh), RUN_DURATION and
# RUN_WINDOW (shorter checks).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
config=$1 engine=$2 sessions=$3 out=$4
: "${MODEL_DIR:?set MODEL_DIR to the model snapshot}"
PORT=${PORT:-8000}
VLLM=${VLLM:-vllm}
MOONCAKE_MASTER=${MOONCAKE_MASTER:-mooncake_master}
AIPERF=${AIPERF:-aiperf}
mkdir -p "$out"
out=$(cd "$out" && pwd)
RUN_DIR=$out/server
mkdir -p "$RUN_DIR"
# shellcheck source=configs/common.sh
source "$here/configs/common.sh"
# shellcheck disable=SC1090
source "$here/configs/$config.sh"
# Shorter checks: RUN_DURATION and RUN_WINDOW ("LO HI") replace the configuration's run length and window.
DURATION=${RUN_DURATION:-$DURATION}
read -r -a WINDOW <<< "${RUN_WINDOW:-${WINDOW[*]}}"

case $engine in
  official) serve=("${SERVE_BASE[@]}" "${SERVE_COMMON[@]}" "${SERVE_OFFICIAL[@]}")
            env=("${ENV_COMMON[@]}"); master=("${MASTER_COMMON[@]}" "${MASTER_OFFICIAL[@]}") ;;
  flashstate) serve=("${SERVE_BASE[@]}" "${SERVE_COMMON[@]}" "${SERVE_FLASHSTATE[@]}")
              env=("${ENV_COMMON[@]}" "${ENV_FLASHSTATE[@]}"); master=("${MASTER_COMMON[@]}" "${MASTER_FLASHSTATE[@]}") ;;
  *) echo "ENGINE is official or flashstate" >&2; exit 2 ;;
esac
if [[ " ${serve[*]} " == *" $RUN_DIR/official_mtp.yaml "* ]]; then
  python3 "$here/tools/official_mtp_config.py" "$MODEL_DIR" "$RUN_DIR/official_mtp.yaml"
fi

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2> /dev/null || true; done; wait 2> /dev/null || true; }
trap cleanup EXIT
if [[ $STORE == mooncake ]]; then
  printf '{"mode": "embedded", "metadata_server": "P2PHANDSHAKE", "master_server_address": "127.0.0.1:%s", "protocol": "tcp", "device_name": "", "global_segment_size": %s, "local_buffer_size": "4GB", "enable_offload": false}\n' \
    "$MASTER_PORT" "$MOONCAKE_SEGMENT_BYTES" > "$RUN_DIR/mooncake.json"
  env+=(MOONCAKE_CONFIG_PATH="$RUN_DIR/mooncake.json")
  $MOONCAKE_MASTER "${master[@]}" > "$RUN_DIR/master.log" 2>&1 &
  pids+=($!)
  sleep 5
fi
printf '%s\n' "${env[@]}" > "$RUN_DIR/env.txt"
printf '%q ' $VLLM serve "$MODEL_DIR" "${serve[@]}" --host 127.0.0.1 --port "$PORT" > "$RUN_DIR/command.txt"
# shellcheck disable=SC2086
env "${env[@]}" $VLLM serve "$MODEL_DIR" "${serve[@]}" --host 127.0.0.1 --port "$PORT" > "$RUN_DIR/server.log" 2>&1 &
pids+=($!)
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  kill -0 "${pids[-1]}" 2> /dev/null || { echo "server exited (see $RUN_DIR/server.log)" >&2; exit 1; }
  sleep 10
done
sleep 30

# The dataset loads offline from its cache (tools/fetch_dataset.sh). HF_HUB_OFFLINE stays unset: with it, the client
# looks the tokenizer up as a Hub repository instead of loading the local directory.
env HF_DATASETS_OFFLINE=1 "${CLIENT_ENV[@]}" $AIPERF profile "${CLIENT_BASE[@]}" "${CLIENT[@]}" \
  --url "http://127.0.0.1:$PORT" --model "$SERVED" --tokenizer "$MODEL_DIR" --concurrency "$sessions" \
  --benchmark-duration "$DURATION" --output-artifact-dir "$out/client" > "$out/client.log" 2>&1
python3 "$here/../analysis/readings.py" "$out/client" --gpus 2 --window "${WINDOW[@]}" | tee "$out/readings.json"
