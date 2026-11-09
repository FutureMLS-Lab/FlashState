# What FlashStates changes

Every change is an option that is off by default. In vLLM (`third_party/vllm`):

| Option | Effect |
|---|---|
| `--mamba-frontier-checkpoint` | keep a finished request's recurrent state at its exact end, so the next turn resumes there instead of at the last block boundary of its prompt |
| `--mamba-prefill-export` | let a prefill chunk run past a position where a state must be cached and have the backend write the state there |
| `--prefix-cache-eviction gds`, `--prefill-cost-coefficients D E` | evict whole resumable states by GreedyDual-Size over the prefill seconds a state saves per block it frees |
| `--prefix-cache-leaf-last`, `--prefix-cache-leaf-idle-seconds S` | evict every inner state before any leaf; a leaf unused for `S` seconds ranks as inner |
| `--mamba-spec-state-layout compact` | under speculative decoding, keep one state block per running request and fold accepted updates in after sampling |
| `VLLM_DETOKENIZE_PRIME_SUFFIX=1` | prime the incremental detokenizer with the prompt's last tokens |
| Mooncake store connector keys (`frontier_write_through`, `lookup_newest_state`, `unit_policy_url`, ...) | write each finished request's state through to the store, look up the newest state first and register recovery units with the master |

In Mooncake (`third_party/mooncake`): `--eviction_policy=unit_gds` on the master evicts whole recovery units (a state
and the pages only it needs) by the same priority, with `--unit_leaf_last`, `--unit_idle_demote_s`,
`--unit_anchor_tier` and `--unit_dead_headroom_ratio` as above; `MC_GPU_COPY=pinned` on the client copies between GPU
and host memory on a non-blocking stream through pinned buffers.

With every option off, the engines behave as vLLM v0.33.0 and Mooncake 0.3.13.post1, except for three vLLM fixes that
are always on: a Mamba block whose state column a chunk did not write is not cached; GLM-5.3-Flash's MTP layer is built
unquantized when the checkpoint stores it in BF16; and upstream fix
[#55112](https://github.com/vllm-project/vllm/pull/55112) (copy every overlaid KV view when copying KV cache blocks).
[`benchmarks/configs/`](../benchmarks/configs) shows which options each configuration sets.
