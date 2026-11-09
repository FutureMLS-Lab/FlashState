# Settings shared by every configuration. Sourced by benchmarks/run_point.sh.
# The CUDA graph batch sizes the GPU-only servers capture (91 sizes up to 4,608; vLLM's default is 83 up to 1,024).
CAPTURE='{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,528,544,560,576,592,608,624,640,656,672,688,704,720,736,752,768,784,800,816,832,848,864,880,896,912,928,944,960,976,992,1008,1024,1280,1536,2048,2560,3072,3584,4096,4608]}'
# Every server, both engines.
SERVE_BASE=(--tensor-parallel-size 2 --enable-prefix-caching --max-num-batched-tokens 16384
            --gpu-memory-utilization 0.85 --seed 0 --prefix-cache-retention-interval 0
            --enable-prompt-tokens-details --cudagraph-metrics)
ENV_COMMON=(VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900)
# FlashStates only: detokenizer priming (an option of the FlashStates fork; the official image runs as shipped).
ENV_FLASHSTATE=(VLLM_DETOKENIZE_PRIME_SUFFIX=1)
# The four FlashStates switches; each configuration adds its model's prefill cost coefficients (seconds per token d,
# and e per token per 100,000 tokens of context; only their ratio matters).
FLASHSTATE_SWITCHES=(--mamba-frontier-checkpoint --mamba-prefill-export --prefix-cache-eviction gds
                     --prefix-cache-leaf-last --prefix-cache-leaf-idle-seconds 60)
# The workload: the public coding-agent traces at a pinned revision, replayed live by token ID.
DATASET=semianalysis_cc_traces_weka_062126
DATASET_REPO=semianalysisai/cc-traces-weka-062126
DATASET_REVISION=23f152f6f0f9399a85901b89a6458def0ef16729
CLIENT_BASE=(--scenario agentx --unsafe-override --endpoint /v1/completions --endpoint-type completions
             --weka-token-ids --weka-live-assistant --tokenizer-trust-remote-code
             --public-dataset "$DATASET" --ui-type none --export-level raw)
CLIENT_ENV=()
STORE=none
