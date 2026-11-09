# Install from source

The B200 image (`docker/Dockerfile.b200`) is the reference install. This page does the same steps on your own machine,
for example for other GPUs. You need Linux x86_64, CUDA 13 drivers, Python 3.12 and [uv](https://docs.astral.sh/uv/).

| Part | Source | Needed for |
| --- | --- | --- |
| vLLM fork | `third_party/vllm`: vLLM v0.33.0 plus FlashStates' changes | every configuration |
| Mooncake fork | `third_party/mooncake`: Mooncake 0.3.13.post1 plus FlashStates' store changes | the Mooncake store configuration |
| replay client | `benchmarks/agentx_harness` | the benchmarks |

Every FlashStates change is an option that is off by default; with the options off, each fork behaves as its upstream
version.

## 1. vLLM

The fork changes only Python files of vLLM v0.33.0 and adds one optional kernel, so it runs on v0.33.0's compiled
kernels. Install v0.33.0, then copy the changed files over it (`docker/vllm_overlay.txt` lists them, as `git diff
--name-status` against v0.33.0; the image does the same):

```bash
git submodule update --init third_party/vllm
uv venv --python 3.12 ~/flashstate && source ~/flashstate/bin/activate
uv pip install vllm==0.33.0
site=$(python -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')
while read -r status path; do
  case $status in
    D) rm -f "$site/${path#vllm/}" ;;
    *) install -D -m 644 "third_party/vllm/$path" "$site/${path#vllm/}" ;;
  esac
done < docker/vllm_overlay.txt
```

**KDA export kernel (GLM-5.3-Flash on B200 only).** GLM-5.3-Flash's linear-attention layers save recurrent checkpoints
inside a prefill chunk (`--mamba-prefill-export`) with an SM100 kernel built from FlashKDA. Build it with the CUDA 13
toolkit (`nvcc`), from the commits the fork's CMake pins:

```bash
mkdir -p ~/src && cd ~/src
curl -fsSL https://github.com/vllm-project/FlashKDA/archive/17a037d98da546deb4591e967cf961a43c034d8b.tar.gz | tar -xz
curl -fsSL https://github.com/NVIDIA/cutlass/archive/5c149f52a436782210263fb2f19b354443a61c6a.tar.gz | tar -xz
cd -
uv pip install ninja
python docker/build_flashkda_export.py third_party/vllm ~/src/FlashKDA-17a037d98da546deb4591e967cf961a43c034d8b \
    ~/src/cutlass-5c149f52a436782210263fb2f19b354443a61c6a /tmp/kda
cp /tmp/kda/_flashkda_export_C.so "$site/"
```

Without this build, as on other GPUs, `--mamba-prefill-export` takes FlashKDA's own checkpoint path. Qwen3.8-Flash-Next
and Nemotron 3 Super do not use the kernel: Qwen3.8-Flash-Next exports through FlashInfer (a vendored kernel on B200,
FlashInfer's stride checkpoints elsewhere), and Nemotron 3 Super through the Triton SSD on every GPU.

## 2. Mooncake (store configuration only)

FlashStates' store changes are in the store client module and the master. Install the Mooncake 0.3.13.post1 wheel,
then replace its store module with the fork's and use the fork's master. The build dependencies are the apt packages
in the `mooncake-build` stage of `docker/Dockerfile.b200`.

```bash
git submodule update --init --recursive third_party/mooncake
uv pip install mooncake-transfer-engine-cuda13==0.3.13.post1 cmake==3.31.6
cmake -S third_party/mooncake -B build/mooncake -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_DEBUG_SYMBOLS=OFF \
    -DUSE_CUDA=ON -DUSE_TCP=ON -DUSE_ETCD=OFF -DSTORE_USE_ETCD=OFF -DUSE_HTTP=ON -DWITH_TE=ON -DWITH_STORE=ON \
    -DWITH_P2P_STORE=OFF -DWITH_STORE_RUST=OFF -DWITH_RUST_EXAMPLE=OFF -DWITH_EP=OFF -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DPython_EXECUTABLE="$(command -v python)"
cmake --build build/mooncake --target store mooncake_master
pkg=$(python -c 'import os, mooncake; print(os.path.dirname(mooncake.__file__))')
cp "$(find build/mooncake/mooncake-integration -maxdepth 1 -name 'store*.so')" "$(find "$pkg" -maxdepth 1 -name 'store*.so')"
export MOONCAKE_MASTER=$(find "$PWD/build/mooncake" -name mooncake_master -type f -executable | head -n 1)
```

## 3. The replay client and the traces

The client gets its own environment:

```bash
uv venv --python 3.12 ~/aiperf
VIRTUAL_ENV=~/aiperf uv pip install benchmarks/agentx_harness
HF_HOME=/data/hf PYTHON=~/aiperf/bin/python benchmarks/tools/fetch_dataset.sh
```

## 4. Run

`benchmarks/run_local.sh` takes the same arguments as `benchmarks/run_point.sh` and finds the programs through
environment variables:

```bash
MODEL_DIR=/models/glm53flash CUDA_VISIBLE_DEVICES=0,1 PORT=8000 HF_HOME=/data/hf \
VLLM=~/flashstate/bin/vllm AIPERF=~/aiperf/bin/aiperf \
    benchmarks/run_local.sh glm53flash-nvfp4-gpu flashstate 48 results/glm53flash-nvfp4-gpu/flashstate/48
```

For the store configuration, also set `MOONCAKE_MASTER` (above). For the official baseline, install `vllm==0.33.0` and
`mooncake-transfer-engine-cuda13==0.3.13.post1` (the version the official image ships) into a separate environment
and point `VLLM` and `MOONCAKE_MASTER` at its programs, with the engine argument `official`.

## Other GPUs and models

The configurations size each engine's KV cache in bytes per GPU for B200 (about 180 GB per GPU). On other GPUs, set
`--kv-cache-memory-bytes` in a copy of the configuration to what fits, the same for both engines. For another hybrid
model, its prefill cost coefficients (`--prefill-cost-coefficients`) weight GreedyDual-Size eviction by the prefill a
cached state saves; only their ratio matters.
