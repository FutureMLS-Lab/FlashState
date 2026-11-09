# How the image is built

[`docker/Dockerfile.b200`](../docker/Dockerfile.b200) builds the B200 image from the repository root, with the
submodules checked out:

```bash
git submodule update --init --recursive
docker/write_source.sh "$(git -C third_party/vllm rev-parse HEAD)" "$(git -C third_party/mooncake rev-parse HEAD)" \
    "$(git rev-parse HEAD)"
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.b200 -t flashstates-b200 .
```

The image starts from `vllm/vllm-openai:v0.33.0`, pinned by digest, and adds three parts:

- **FlashStates' vLLM.** The files FlashStates changes under `vllm/` are copied over the installed package
  ([`docker/vllm_overlay.txt`](../docker/vllm_overlay.txt) lists them), so the official compiled kernels stay. One
  compiled kernel is added: the SM100 KDA state export (`vllm._flashkda_export_C`), built from FlashKDA and CUTLASS at
  pinned commits.
- **FlashStates' Mooncake.** The `mooncake-transfer-engine-cuda13==0.3.13.post1` wheel that the official image ships,
  with the `store` module and `mooncake_master` built from `third_party/mooncake`.
- **The replay client and the benchmark scripts.** The client has its own virtual environment at `/opt/aiperf`; the
  scripts are in `/opt/flashstate`.

`docker/write_source.sh` records the commits and inputs of each build in `/opt/flashstate/SOURCE.json`. The build has no
GPU, so it checks only that the engine imports and that no compiled addition misses a library. The image has no
entrypoint: run `vllm serve`, `mooncake_master` or `aiperf` as its command. The workflow
[`.github/workflows/image-b200.yml`](../.github/workflows/image-b200.yml) builds and pushes the image on manual
dispatch.

The published B200 image is `ghcr.io/futuremls-lab/flashstates-b200:v0.33.0-fs3cfd432-mc59ca03f-b200` (also tagged
`v0.1.0-b200`, x86-64; on Docker Hub `weili0234/flashstates:v0.33.0-fs3cfd432-mc59ca03f-b200` and
`weili0234/flashstates-vllm:v0.1.0-b200`): FlashStates' vLLM at commit `3cfd432` and FlashStates' Mooncake at commit
`59ca03f`, the `flashstate` branches of [FutureMLS-Lab/vllm](https://github.com/FutureMLS-Lab/vllm/tree/flashstate) and
[FutureMLS-Lab/Mooncake](https://github.com/FutureMLS-Lab/Mooncake/tree/flashstate).

[Install from source](install.md) installs the same parts without Docker.

## The GB300 image for Kimi-K3.1-Pro

`ghcr.io/futuremls-lab/flashstates-gb300:kimi-k31-cu134-266e88c-fs88bad5a-mcf74d68c-gb300` (also tagged
`v0.1.0-kimi-k31-gb300`, aarch64; on Docker Hub `weili0234/flashstates:kimi-k31-cu134-266e88c-fs88bad5a-mcf74d68c-gb300`
and `weili0234/flashstates-vllm:v0.1.0-kimi-k31-gb300`) is one layer over the official `vllm/vllm-openai:kimi-k31-cu134`
image, whose vLLM is commit `266e88c05`:

- **FlashStates' vLLM.** The 52 files FlashStates changes under `vllm/`, from the `flashstate-k31pro-gb300` branch of
  [FutureMLS-Lab/vllm](https://github.com/FutureMLS-Lab/vllm/tree/flashstate-k31pro-gb300) (commit `88bad5a`).
- **FlashStates' Mooncake.** The store client and `mooncake_master` built from the `flashstate-k31pro-gb300` branch of
  [FutureMLS-Lab/Mooncake](https://github.com/FutureMLS-Lab/Mooncake/tree/flashstate-k31pro-gb300) (commit `f74d68c`), in
  place of the image's own Mooncake.
- **`WITH_NVIDIA_PEERMEM=0`.** GB300 trays have no GPU peer-memory module, so the store registers GPU memory through
  DMA-BUF.

The image has no replay client and no benchmark scripts; the serve command is in the branch's
[README](https://github.com/FutureMLS-Lab/vllm/blob/flashstate-k31pro-gb300/.github/README.md).
