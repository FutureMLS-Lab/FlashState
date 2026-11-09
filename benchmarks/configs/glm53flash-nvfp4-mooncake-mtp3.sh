# GLM-5.3-Flash NVFP4 with MTP 3 and a 512 GiB Mooncake store per server (each of the two TP ranks contributes
# 256 GiB): TP2 on two B200 GPUs, about 640 GiB of host memory per server.
MODEL_REPO=nvidia/GLM-5.3-Flash-NVFP4
MODEL_REVISION=09b04e5e74bca08ca8549fc736d4cdd8624bfde3
SERVED=glm-5.3-flash
DURATION=2700
WINDOW=(900 2700)
SESSIONS=(32 40 48 56 64 72 80)
STORE=mooncake
MOONCAKE_SEGMENT_BYTES=274877906944
# The master's ports follow the server's PORT (8000: 50051 and 9003), and so does the connector's lookup channel (an
# IPC socket named by host name and lookup_rpc_port), so two store servers can share a machine.
MASTER_PORT=$((PORT + 42051))
MASTER_METRICS_PORT=$((PORT + 1003))
MASTER_COMMON=(--rpc_port="$MASTER_PORT" --metrics_port="$MASTER_METRICS_PORT" --allocation_strategy=local_first
               --enable_metric_reporting=true --default_kv_lease_ttl=10000 --eviction_high_watermark_ratio=0.98
               --eviction_ratio=0.05)
MASTER_OFFICIAL=()
MASTER_FLASHSTATE=(--eviction_policy=unit_gds --unit_leaf_last=true --unit_idle_demote_s=300 --unit_orphan_grace_s=120
                   --unit_anchor_tier=true --unit_dead_headroom_ratio=0.05 --unit_debug=true)
OFFICIAL_KV='{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"load_async":true,"lookup_rpc_port":'"$PORT"'}}'
FS_KV='{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"load_async":true,"lookup_newest_state":true,"frontier_write_through":true,"frontier_supersede_remove":true,"frontier_lookup_memo_s":30,"unit_policy_url":"http://127.0.0.1:'"$MASTER_METRICS_PORT"'","frontier_load_local_pages":true,"frontier_write_wait_s":3,"frontier_supersede_lag":2,"frontier_locator_cache":true,"staging_buffer_mib":4096,"lookup_rpc_port":'"$PORT"'}}'
SERVE_COMMON=(--served-model-name "$SERVED" --max-model-len 1048576)
# Official vLLM cannot load the checkpoint's BF16 MTP layer as quantized: official_mtp.yaml (benchmarks/tools/
# official_mtp_config.py) lists it as unquantized, with no file of the checkpoint changed.
SERVE_OFFICIAL=(--kv-cache-memory-bytes 55000000000 --config "$RUN_DIR/official_mtp.yaml"
                --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
                --long-prefill-token-threshold 8960 --prefix-match-unit 1120 --kv-transfer-config "$OFFICIAL_KV")
# Both engines size attention blocks to the KDA state's page under MTP 3 (4,480 tokens in v0.33.0) and match prefixes
# in quarter blocks.
SERVE_FLASHSTATE=(--kv-cache-memory-bytes 54000000000 "${FLASHSTATE_SWITCHES[@]}" --prefill-cost-coefficients 1.862e-5 3.33e-6
                  --prefix-match-unit 1120 --long-prefill-token-threshold 8960 --mamba-spec-state-layout compact
                  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"truncate_acceptance_at_stop":true,"disable_eagle_block_drop":true}'
                  --kv-transfer-config "$FS_KV")
ENV_FLASHSTATE+=(MC_GPU_COPY=pinned)
CLIENT=()
CLIENT_ENV=(AIPERF_AGENTIC_WARMUP=off)
