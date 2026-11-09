# Qwen3.8-Flash-Next FP8 with a 512K context (YaRN factor 2.0 over 262,144), GPU only: TP2 on two B200 GPUs per server.
MODEL_REPO=Qwen/Qwen3.8-Flash-Next-FP8
MODEL_REVISION=236dfdf285828023ca3bcd3f37366c58a3469b13
SERVED=qwen38-flash
DURATION=4500
WINDOW=(900 4500)
SESSIONS=(32 40 48 56)
YARN='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}'
ENV_COMMON+=(VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
SERVE_COMMON=(--served-model-name "$SERVED" --max-model-len 524288 --kv-cache-memory-bytes 72224538624
              --kv-cache-dtype fp8 --hf-overrides "$YARN" --compilation-config "$CAPTURE"
              --long-prefill-token-threshold 6272)
SERVE_OFFICIAL=()
SERVE_FLASHSTATE=("${FLASHSTATE_SWITCHES[@]}" --prefill-cost-coefficients 2.230e-5 4.34e-6)
# Traces whose recorded peak exceeds the context are left out (323 of 393 remain).
CLIENT=(--max-context-length 524288)
