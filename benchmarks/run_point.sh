#!/bin/bash
# One measurement: a fresh server of one engine, then one closed-loop replay of the coding-agent traces at one session
# count, then the readings. Docker on one machine with the server's GPUs.
#
#   benchmarks/run_point.sh CONFIG ENGINE SESSIONS OUT_DIR
#
#   CONFIG    a configuration in benchmarks/configs (for example glm53flash-nvfp4-gpu)
#   ENGINE    official (the official vLLM image, as shipped) or flashstate (the FlashStates image)
#   SESSIONS  concurrent root sessions (sub-agents run inside them)
#
# Environment: MODEL_DIR (required: the model snapshot at the configuration's revision), HF_HOME (the dataset cache,
# see benchmarks/tools/fetch_dataset.sh; default ~/.cache/huggingface), GPUS (default 0,1), PORT (default 8000),
# FS_IMAGE and OFFICIAL_IMAGE (defaults below).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
config=$1 engine=$2 sessions=$3 out=$4
: "${MODEL_DIR:?set MODEL_DIR to the model snapshot}"
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
GPUS=${GPUS:-0,1}
PORT=${PORT:-8000}
FS_IMAGE=${FS_IMAGE:-ghcr.io/futuremls-lab/flashstates-b200:v0.33.0-fs3cfd432-mc59ca03f-b200@sha256:605a03ca0f80ab021063744ea16c40613dbe6b82d066022022811e47dc9aaef3}
OFFICIAL_IMAGE=${OFFICIAL_IMAGE:-vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e}
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
  official) image=$OFFICIAL_IMAGE; serve=("${SERVE_BASE[@]}" "${SERVE_COMMON[@]}" "${SERVE_OFFICIAL[@]}")
            env=("${ENV_COMMON[@]}"); master=("${MASTER_COMMON[@]}" "${MASTER_OFFICIAL[@]}") ;;
  flashstate) image=$FS_IMAGE; serve=("${SERVE_BASE[@]}" "${SERVE_COMMON[@]}" "${SERVE_FLASHSTATE[@]}")
              env=("${ENV_COMMON[@]}" "${ENV_FLASHSTATE[@]}"); master=("${MASTER_COMMON[@]}" "${MASTER_FLASHSTATE[@]}") ;;
  *) echo "ENGINE is official or flashstate" >&2; exit 2 ;;
esac
if [[ " ${serve[*]} " == *" $RUN_DIR/official_mtp.yaml "* ]]; then
  python3 "$here/tools/official_mtp_config.py" "$MODEL_DIR" "$RUN_DIR/official_mtp.yaml"
fi

server_cmd=$(printf '%q ' vllm serve /model "${serve[@]}" --host 127.0.0.1 --port "$PORT")
if [[ $STORE == mooncake ]]; then
  printf '{"mode": "embedded", "metadata_server": "P2PHANDSHAKE", "master_server_address": "127.0.0.1:%s", "protocol": "tcp", "device_name": "", "global_segment_size": %s, "local_buffer_size": "4GB", "enable_offload": false}\n' \
    "$MASTER_PORT" "$MOONCAKE_SEGMENT_BYTES" > "$RUN_DIR/mooncake.json"
  env+=(MOONCAKE_CONFIG_PATH="$RUN_DIR/mooncake.json")
  server_cmd="$(printf '%q ' mooncake_master "${master[@]}") > $RUN_DIR/master.log 2>&1 & sleep 5; exec $server_cmd"
fi
printf '%s\n' "$server_cmd" > "$RUN_DIR/command.txt"
printf '%s\n' "${env[@]}" > "$RUN_DIR/env.txt"

name=flashstate-bench-$PORT
docker rm -f "$name" > /dev/null 2>&1 || true
docker run -d --name "$name" --gpus "\"device=$GPUS\"" --ipc=host --network host --shm-size 32g \
  -v "$MODEL_DIR":/model:ro -v "$RUN_DIR":"$RUN_DIR" --env-file "$RUN_DIR/env.txt" --entrypoint bash \
  "$image" -c "$server_cmd" > /dev/null
trap 'docker logs "$name" > "$RUN_DIR/server.log" 2>&1 || true; docker rm -f "$name" > /dev/null 2>&1 || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  docker inspect -f '{{.State.Running}}' "$name" | grep -q true || { echo "server exited" >&2; exit 1; }
  sleep 10
done
sleep 30

# The client: the replay client of the FlashStates image, with the dataset cache pinned to the configuration's revision.
# The dataset loads offline from that cache. HF_HUB_OFFLINE stays unset: with it, the client looks the tokenizer up as
# a Hub repository instead of loading the model directory.
client_env=()
for kv in "${CLIENT_ENV[@]}"; do client_env+=(-e "$kv"); done
docker run --rm --network host -v "$MODEL_DIR":/model:ro -v "$HF_HOME":/hf -v "$out":/out \
  -e HF_HOME=/hf -e HF_DATASETS_OFFLINE=1 "${client_env[@]}" --entrypoint aiperf "$FS_IMAGE" \
  profile "${CLIENT_BASE[@]}" "${CLIENT[@]}" --url "http://127.0.0.1:$PORT" --model "$SERVED" --tokenizer /model \
  --concurrency "$sessions" --benchmark-duration "$DURATION" --output-artifact-dir /out/client > "$out/client.log" 2>&1
python3 "$here/../analysis/readings.py" "$out/client" --gpus 2 --window "${WINDOW[@]}" | tee "$out/readings.json"
