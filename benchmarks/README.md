# Benchmarks

Every result compares FlashStates with official vLLM on the same model, GPUs and workload: closed-loop coding-agent
sessions from the public traces
[`semianalysisai/cc-traces-weka-062126`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
(revision `23f152f6f0f9399a85901b89a6458def0ef16729`, 393 sessions with sub-agents), replayed live by token ID: each
turn's prompt begins with the previous turn's prompt and the server's own output, token for token.

- **FlashStates** runs the FlashStates image (`docker/Dockerfile.b200`).
- **Official vLLM** runs the official `vllm/vllm-openai:v0.33.0` image as shipped, with its own Mooncake client and
  master when a store is used.
- The client is `agentx_harness/` (AIPerf with live continuation), run from the FlashStates image for both engines.

## One point

```bash
# once: the traces at the pinned revision, with the client's dataset cache (or, with the client installed,
# HF_HOME=/data/hf benchmarks/tools/fetch_dataset.sh)
docker run --rm -v /data/hf:/hf -e HF_HOME=/hf --entrypoint bash \
    ghcr.io/futuremls-lab/flashstates-b200:v0.33.0-fs3cfd432-mc59ca03f-b200 \
    -c 'PYTHON=/opt/aiperf/bin/python /opt/flashstate/benchmarks/tools/fetch_dataset.sh'
hf download nvidia/GLM-5.3-Flash-NVFP4 --revision 09b04e5e74bca08ca8549fc736d4cdd8624bfde3 --local-dir /models/glm53flash
MODEL_DIR=/models/glm53flash HF_HOME=/data/hf GPUS=0,1 PORT=8000 \
    benchmarks/run_point.sh glm53flash-nvfp4-gpu flashstate 48 results/glm53flash-nvfp4-gpu/flashstate/48
MODEL_DIR=/models/glm53flash HF_HOME=/data/hf GPUS=2,3 PORT=8001 \
    benchmarks/run_point.sh glm53flash-nvfp4-gpu official 48 results/glm53flash-nvfp4-gpu/official/48
```

Each run starts a fresh server (and, with a store, a fresh Mooncake master), waits for it to be healthy, replays for
the configuration's duration, stops the server and writes `readings.json` (`analysis/readings.py`): output tokens per
minute, prompt tokens per minute, TTFT p50 and p90, p50 decode speed and the prefix-cache hit rate of the requests in
the measurement window. The readings skip the first 15 minutes, while the cache fills, so a point takes about 50
minutes on GLM-5.3-Flash and about 80 on Qwen3.8-Flash-Next and Nemotron 3 Super. Two servers can run side by side on
one 4-GPU machine with different `GPUS` and `PORT` (a store server's master ports and lookup channel follow `PORT`);
the two clients then start together.

## Configurations

`configs/common.sh` holds what every server and client shares; each file below adds one setup.

| Configuration | Model | GPUs and memory per server | Engine settings | Run, window | Session counts |
| --- | --- | --- | --- | --- | --- |
| `glm53flash-nvfp4-gpu` | GLM-5.3-Flash NVFP4, 1M context | 2 B200, 67,053,718,689-byte KV pool per GPU | GPU only, MTP off | 2,700 s, [900, 2,700) s | 24, 36, 40, 44, 48 |
| `glm53flash-nvfp4-mooncake-mtp3` | GLM-5.3-Flash NVFP4, 1M context | 2 B200, about 640 GiB host memory, 512 GiB Mooncake store | MTP 3, Mooncake store | 2,700 s, [900, 2,700) s | 32, 40, 48, 56, 64, 72, 80 |
| `qwen38flashnext-fp8-512k-gpu` | Qwen3.8-Flash-Next FP8, 512K context (YaRN 2.0) | 2 B200, 72,224,538,624-byte KV pool per GPU | GPU only | 4,500 s, [900, 4,500) s | 32, 40, 48, 56 |
| `nemotron3super-fp8-gpu` | Nemotron 3 Super FP8, 1M context | 2 B200, 24,111,672,637-byte KV pool per GPU | GPU only | 4,500 s, [900, 4,500) s | 28, 36, 40, 44, 48 |

Both engines get the same GPU memory. FlashStates adds its four switches (frontier checkpoints, in-chunk state export,
eviction by GreedyDual-Size over the prefill a state saves, inner states before each session's newest one) with the
model's prefill cost coefficients, and detokenizer priming (`VLLM_DETOKENIZE_PRIME_SUFFIX=1`). With the store,
FlashStates also writes each finished request's state through to Mooncake and its master evicts whole recovery units
(`unit_gds`). Official vLLM cannot load GLM-5.3-Flash's BF16 MTP layer as quantized, so `tools/official_mtp_config.py`
writes a config file that lists it as unquantized, with no file of the checkpoint changed.

Each configuration pins its model's revision (`MODEL_REVISION`):

| Model | Revision |
| --- | --- |
| [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` |
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | `236dfdf285828023ca3bcd3f37366c58a3469b13` |
| [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) | `744b1880a37996c5d56bf454ae164dfd74d77c4e` |

## Hardware

- Two B200 GPUs (SM100, about 180 GB each) per server. Other GPUs need their own KV pool sizes
  ([other GPUs](../docs/install.md#other-gpus-and-models)).
- Host memory: the client needs about 70 GiB while it builds the sessions; store runs need about 640 GiB per server.
- Disk: the model weights (about 205 GB for GLM-5.3-Flash NVFP4) and up to about 10 GB per run, most of it the
  client's per-request export.

## The README's figures

The README shows the paper's measurements, as on [flashstates.ai](https://flashstates.ai). Their rows are in
`docs/figures/`, in the format of `analysis/collect.py --json`.

- **GPU only** ([`gpu-only.json`](../docs/figures/gpu-only.json)): the values as read from the paper's figure. 393
  sessions with sub-agents (the 323 that fit Qwen3.8-Flash-Next's 512K context), two B200 GPUs per server, vLLM and
  FlashStates with the same cache budget. Decode TPM per GPU counts output tokens per minute per GPU. Prefill TPM per GPU
  counts the prompt tokens (read from the cache or computed) of the requests whose first token arrives in the window.
  Uncached tokens p50 is the median of the prompt tokens a request computes rather than reads from the cache.
- **Mooncake store** ([`mooncake-sweep.json`](../docs/figures/mooncake-sweep.json)): GLM-5.3-Flash NVFP4 with MTP 3,
  two B200 GPUs and 512 GiB of host memory for cache per engine: official vLLM and FlashStates with a 512 GiB Mooncake
  store; SGLang v0.5.23 with HiCache's host tier only (511.5 GiB, write-through, no Mooncake) and EAGLE with 3 steps,
  run later with the same client and GPU settings. One run per engine and point at 32 to 80 sessions in steps of 8,
  read over minutes 15 to 45. Input tokens include those read from the cache, and bubble area shows p90 TTFT.
  FlashStates' peak is 26% above the best of vLLM or SGLang (SGLang at 64 sessions) and 24% above vLLM's best measured
  run (60 sessions, not shown).
  Measured before the port to vLLM v0.33.0: official vLLM at commit `ec8a3725`, FlashStates on the same base.

- **Kimi-K3.1-Pro on GB300** ([`k31-sweep.json`](../docs/figures/k31-sweep.json)): one engine on two GB300 trays of
  one NVL72 domain (TP8, decode context parallelism 8, the specialized decode path), DSpark speculative decoding with 5
  draft tokens, and a 640 GiB Mooncake store per engine (80 GB per GPU). vLLM is the official
  `vllm/vllm-openai:kimi-k31-cu134` image with its own Mooncake; FlashStates ran the code of the GB300 image over the
  same base (its 52 changed vLLM files are byte-identical, with the same Mooncake build). Both pin the GPU pool to
  1,050 blocks of 2,048 tokens. One run per engine and point at 24 to 52 sessions in steps of 4, each a fresh engine with a
  5-minute warm-up at 16 sessions and then an hour of replay, read over minutes 15 to 45. The y axis counts output tokens, which move less than input tokens with
  the prompt lengths of the requests in the window, and bubble area shows p90 TTFT. These runs used a two-node Slurm
  harness, not `run_point.sh`.

```bash
python3 analysis/plot.py serving docs/figures/gpu-only.json \
    glm53flash-nvfp4-gpu,nemotron3super-fp8-gpu,qwen38flashnext-fp8-512k-gpu docs/figures/gpu-only-serving.png
python3 analysis/plot.py tradeoff docs/figures/mooncake-sweep.json glm53flash-nvfp4-mooncake-mtp3 docs/figures/mooncake-tradeoff.png
python3 analysis/plot.py tradeoff docs/figures/k31-sweep.json kimi-k31-pro-gb300-mooncake-dspark docs/figures/k31-tradeoff.png
```

## Results

Each run writes `readings.json` into its output directory. With the layout above
(`results/<configuration>/<engine>/<sessions>`):

```bash
python3 analysis/collect.py results --json results/rows.json                   # one row per run
python3 analysis/table.py results/rows.json summary glm53flash-nvfp4-gpu       # FlashStates' change per session count
python3 analysis/table.py results/rows.json values glm53flash-nvfp4-gpu        # both engines' readings, as below
python3 analysis/plot.py serving results/rows.json \
    glm53flash-nvfp4-gpu,nemotron3super-fp8-gpu,qwen38flashnext-fp8-512k-gpu serving.png    # the README's GPU-only figure
python3 analysis/plot.py tradeoff results/rows.json glm53flash-nvfp4-mooncake-mtp3 tradeoff.png  # its Mooncake figure
```

The tables below come from the release validation runs: one fresh server per point on 2 B200 GPUs, vLLM being the
official `vllm/vllm-openai:v0.33.0` image as shipped. FlashStates' rows give its change against vLLM in parentheses.

### GLM-5.3-Flash NVFP4, GPU only, 1M context (`glm53flash-nvfp4-gpu`)

| Sessions | Engine | Decode TPM per GPU (K) | Prefill TPM per GPU (M) | Uncached tokens p50 | TTFT p90 (s) | Cache hit rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 24 | vLLM | 26.2 | 3.82 | 4,108 | 0.61 | 0.952 |
| 24 | FlashStates | 26.3 (1.01×) | 3.88 (1.02×) | 769 (−81%) | 0.75 (+22%) | 0.959 |
| 36 | vLLM | 29.0 | 4.04 | 4,178 | 1.24 | 0.886 |
| 36 | FlashStates | 31.4 (1.08×) | 4.48 (1.11×) | 961 (−77%) | 1.25 (+1%) | 0.917 |
| 40 | vLLM | 29.1 | 3.77 | 4,305 | 2.54 | 0.847 |
| 40 | FlashStates | 34.9 (1.20×) | 4.71 (1.25×) | 961 (−78%) | 1.28 (−49%) | 0.917 |
| 44 | vLLM | 19.7 | 2.60 | 4,416 | 8.24 | 0.679 |
| 44 | FlashStates | 33.8 (1.72×) | 4.92 (1.89×) | 1,025 (−77%) | 1.57 (−81%) | 0.903 |
| 48 | vLLM | 10.1 | 1.43 | 33,632 | 23.79 | 0.247 |
| 48 | FlashStates | 30.7 (3.05×) | 4.58 (3.20×) | 1,025 (−97%) | 2.28 (−90%) | 0.864 |

### Qwen3.8-Flash-Next FP8, GPU only, 512K context (`qwen38flashnext-fp8-512k-gpu`)

| Sessions | Engine | Decode TPM per GPU (K) | Prefill TPM per GPU (M) | Uncached tokens p50 | TTFT p90 (s) | Cache hit rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | vLLM | 30.7 | 3.41 | 3,368 | 0.75 | 0.912 |
| 32 | FlashStates | 30.1 (0.98×) | 3.36 (0.98×) | 1,089 (−68%) | 1.15 (+54%) | 0.898 |
| 40 | vLLM | 34.4 | 3.46 | 3,432 | 1.36 | 0.849 |
| 40 | FlashStates | 37.5 (1.09×) | 3.69 (1.07×) | 1,217 (−65%) | 1.19 (−13%) | 0.882 |
| 48 | vLLM | 24.3 | 2.36 | 4,282 | 8.75 | 0.642 |
| 48 | FlashStates | 38.9 (1.60×) | 3.77 (1.60×) | 1,345 (−69%) | 1.53 (−82%) | 0.844 |
| 56 | vLLM | 13.7 | 1.52 | 43,859 | 15.86 | 0.314 |
| 56 | FlashStates | 37.8 (2.75×) | 3.46 (2.28×) | 1,601 (−96%) | 2.75 (−83%) | 0.791 |

### Nemotron 3 Super FP8, GPU only, 1M context (`nemotron3super-fp8-gpu`)

| Sessions | Engine | Decode TPM per GPU (K) | Prefill TPM per GPU (M) | Uncached tokens p50 | TTFT p90 (s) | Cache hit rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 28 | vLLM | 27.3 | 4.78 | 6,490 | 1.07 | 0.909 |
| 28 | FlashStates | 27.6 (1.01×) | 4.94 (1.04×) | 833 (−87%) | 0.71 (−34%) | 0.924 |
| 36 | vLLM | 23.9 | 3.55 | 6,945 | 8.38 | 0.762 |
| 36 | FlashStates | 33.0 (1.38×) | 5.26 (1.48×) | 961 (−86%) | 0.82 (−90%) | 0.899 |
| 40 | vLLM | 22.1 | 2.96 | 7,375 | 10.61 | 0.650 |
| 40 | FlashStates | 35.2 (1.60×) | 5.31 (1.79×) | 961 (−87%) | 1.03 (−90%) | 0.887 |
| 44 | vLLM | 19.2 | 2.56 | 7,792 | 16.37 | 0.567 |
| 44 | FlashStates | 35.1 (1.83×) | 5.19 (2.03×) | 1,153 (−85%) | 1.51 (−91%) | 0.859 |
| 48 | vLLM | 13.1 | 1.90 | 59,508 | 23.54 | 0.255 |
| 48 | FlashStates | 29.5 (2.25×) | 4.24 (2.23×) | 1,281 (−98%) | 3.99 (−83%) | 0.800 |

### GLM-5.3-Flash NVFP4, MTP 3, 512 GiB Mooncake store (`glm53flash-nvfp4-mooncake-mtp3`)

| Sessions | Engine | Decode TPM per GPU (K) | Prefill TPM per GPU (M) | Uncached tokens p50 | TTFT p90 (s) | Cache hit rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | vLLM | 39.1 | 5.87 | 3,441 | 0.82 | 0.964 |
| 32 | FlashStates | 40.0 (1.02×) | 6.22 (1.06×) | 641 (−81%) | 0.57 (−30%) | 0.979 |
| 40 | vLLM | 47.7 | 7.09 | 3,437 | 0.95 | 0.964 |
| 40 | FlashStates | 52.0 (1.09×) | 7.61 (1.07×) | 705 (−79%) | 0.58 (−39%) | 0.981 |
| 48 | vLLM | 53.8 | 8.25 | 3,396 | 1.30 | 0.964 |
| 48 | FlashStates | 57.6 (1.07×) | 8.84 (1.07×) | 833 (−75%) | 0.78 (−40%) | 0.978 |
| 56 | vLLM | 62.5 | 8.74 | 3,486 | 2.10 | 0.958 |
| 56 | FlashStates | 69.2 (1.11×) | 9.62 (1.10×) | 960 (−72%) | 1.10 (−48%) | 0.976 |
| 64 | vLLM | 63.6 | 8.75 | 3,486 | 7.95 | 0.954 |
| 64 | FlashStates | 76.1 (1.20×) | 10.19 (1.17×) | 960 (−72%) | 1.66 (−79%) | 0.972 |
| 72 | vLLM | 61.4 | 8.27 | 3,499 | 16.30 | 0.945 |
| 72 | FlashStates | 84.2 (1.37×) | 11.19 (1.35×) | 1,024 (−71%) | 3.16 (−81%) | 0.969 |
| 80 | vLLM | 61.0 | 7.98 | 3,476 | 22.47 | 0.939 |
| 80 | FlashStates | 85.7 (1.40×) | 11.10 (1.39×) | 1,032 (−70%) | 6.82 (−70%) | 0.963 |

### Kimi-K3.1-Pro on GB300, DSpark, 640 GiB Mooncake store (`kimi-k31-pro-gb300-mooncake-dspark`)

The rows of the README's Kimi-K3.1-Pro figure (setup above), over minutes 15 to 45 of each hour-long run: eight GB300
GPUs per engine, so per GPU means per eighth of the engine.

| Sessions | Engine | Decode TPM per GPU (K) | Prefill TPM per GPU (M) | Uncached tokens p50 | TTFT p90 (s) | Cache hit rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 24 | vLLM | 5.6 | 0.77 | 2,505 | 0.91 | 0.963 |
| 24 | FlashStates | 6.0 (1.06×) | 0.82 (1.07×) | 577 (−77%) | 0.97 (+7%) | 0.979 |
| 28 | vLLM | 5.7 | 0.79 | 2,446 | 1.02 | 0.957 |
| 28 | FlashStates | 6.1 (1.06×) | 0.87 (1.11×) | 512 (−79%) | 0.93 (−9%) | 0.978 |
| 32 | vLLM | 6.0 | 0.83 | 2,623 | 1.17 | 0.956 |
| 32 | FlashStates | 6.5 (1.09×) | 0.91 (1.10×) | 544 (−79%) | 1.14 (−3%) | 0.978 |
| 36 | vLLM | 5.7 | 0.81 | 2,668 | 1.77 | 0.941 |
| 36 | FlashStates | 6.8 (1.20×) | 0.95 (1.18×) | 641 (−76%) | 1.24 (−30%) | 0.974 |
| 40 | vLLM | 6.2 | 0.81 | 2,888 | 2.32 | 0.927 |
| 40 | FlashStates | 7.0 (1.13×) | 0.90 (1.10×) | 832 (−71%) | 1.39 (−40%) | 0.970 |
| 44 | vLLM | 6.2 | 0.84 | 2,838 | 2.91 | 0.931 |
| 44 | FlashStates | 7.7 (1.23×) | 0.99 (1.19×) | 832 (−71%) | 1.33 (−54%) | 0.970 |
| 48 | vLLM | 6.3 | 0.85 | 2,740 | 4.99 | 0.919 |
| 48 | FlashStates | 7.9 (1.25×) | 1.08 (1.28×) | 769 (−72%) | 1.84 (−63%) | 0.969 |
| 52 | vLLM | 6.4 | 0.85 | 2,677 | 9.13 | 0.908 |
| 52 | FlashStates | 8.7 (1.36×) | 1.07 (1.25×) | 896 (−67%) | 1.95 (−79%) | 0.968 |
