# Nemotron 3 Super FP8, GPU only: TP2 on two B200 GPUs per server, a 22.46 GiB KV pool per GPU.
MODEL_REPO=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
MODEL_REVISION=744b1880a37996c5d56bf454ae164dfd74d77c4e
SERVED=default
DURATION=4500
WINDOW=(900 4500)
SESSIONS=(28 36 40 44 48)
ENV_COMMON+=(VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
SERVE_COMMON=(--served-model-name "$SERVED" --max-model-len 1048576 --kv-cache-memory-bytes 24111672637
              --compilation-config "$CAPTURE" --long-prefill-token-threshold 8320)
SERVE_OFFICIAL=()
SERVE_FLASHSTATE=("${FLASHSTATE_SWITCHES[@]}" --prefill-cost-coefficients 1.322e-5 4.19e-6)
CLIENT=()
