# GLM-5.3-Flash NVFP4, GPU only, MTP off: TP2 on two B200 GPUs per server.
MODEL_REPO=nvidia/GLM-5.3-Flash-NVFP4
MODEL_REVISION=09b04e5e74bca08ca8549fc736d4cdd8624bfde3
SERVED=glm-5.3-flash
DURATION=2700
WINDOW=(900 2700)
SESSIONS=(24 36 40 44 48)
SERVE_COMMON=(--served-model-name "$SERVED" --max-model-len 1048576 --kv-cache-memory-bytes 67053718689
              --compilation-config "$CAPTURE" --long-prefill-token-threshold 8704)
SERVE_OFFICIAL=()
SERVE_FLASHSTATE=("${FLASHSTATE_SWITCHES[@]}" --prefill-cost-coefficients 1.862e-5 3.33e-6)
CLIENT=()
