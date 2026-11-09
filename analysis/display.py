"""Display names and metrics shared by the tables and figures."""

TITLES = {
    "glm53flash-nvfp4-gpu": "GLM-5.3-Flash NVFP4, GPU only, 1M context",
    "qwen38flashnext-fp8-512k-gpu": "Qwen3.8-Flash-Next FP8, GPU only, 512K context",
    "nemotron3super-fp8-gpu": "Nemotron 3 Super FP8, GPU only, 1M context",
    "glm53flash-nvfp4-mooncake-mtp3": "GLM-5.3-Flash NVFP4, MTP 3, Mooncake store",
    "kimi-k31-pro-gb300-mooncake-dspark": "Kimi-K3.1-Pro on GB300, DSpark, Mooncake store",
}
SHORT = {
    "glm53flash-nvfp4-gpu": "GLM-5.3-Flash NVFP4, 1M",
    "qwen38flashnext-fp8-512k-gpu": "Qwen3.8-Flash-Next, 512K",
    "nemotron3super-fp8-gpu": "Nemotron 3 Super, 1M",
}
# (key, label, unit, scale, format, higher is better): the paper's serving metrics.
METRICS = (
    ("output_tpm_per_gpu", "Decode TPM per GPU", "K", 1e3, ",.1f", True),
    ("prompt_tpm_per_gpu", "Prefill TPM per GPU", "M", 1e6, ",.2f", True),
    ("uncached_p50", "Uncached tokens p50", "", 1, ",.0f", False),
    ("ttft_p90_s", "TTFT p90", "s", 1, ",.2f", False),
)


def title(config: str) -> str:
    return TITLES.get(config, config)


def change(vllm: float, flashstate: float, higher_is_better: bool) -> str:
    """FlashStates' change against vLLM: a ratio where higher is better, else a percentage."""
    if not vllm:
        return "-"
    if higher_is_better:
        return f"{flashstate / vllm:.2f}×"
    return f"{(flashstate / vllm - 1) * 100:+.0f}%".replace("-", "−")
