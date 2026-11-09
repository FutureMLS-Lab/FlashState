# AgentX replay client: provenance

This directory is AIPerf 0.12.0 as shipped in the AgentX harness
([SemiAnalysisAI/agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness), Apache License 2.0) at commit
`104c7c808666385fbf3962911b55f68bf08a04a`, with our changes for live continuation. Only the package itself is copied:
`src/`, `pyproject.toml`, `LICENSE`, the `ATTRIBUTIONS*.md` files, `README.md` and `docs/tutorials/weka-trace.md`.
Upstream's tests, documentation site and tooling are not included.

## Why

The upstream AgentX scenario replays each turn's recorded prompt, so turn *k*+1 never contains what the server
generated in turn *k*. A cache that resumes at the exact end of the previous turn cannot be measured that way. These
changes make every session continue from the model's actual output.

## Changes over upstream

| Change | Option | Used by the results |
| --- | --- | --- |
| Replace each recorded assistant message with the server's actual output | `--weka-live-assistant` | yes |
| Send prompts as token IDs to `/v1/completions` and splice in the exact output token IDs, so each prompt begins with the previous prompt and output, token for token | `--weka-token-ids` | yes |
| Turn off AIPerf's synthesized agentic warm-up phase | `AIPERF_AGENTIC_WARMUP=off` (environment) | the Mooncake store runs |
| Replay user turns and clear the reasoning of earlier turns | `--weka-user-turns`, `--weka-clear-thinking` | no |
| A hint-free mix of sessions with and without cleared reasoning | `--weka-clear-thinking-mix` | no |
| Reshape the idle gaps between turns | `--weka-gap-shape` | no |

Every option is off by default; without them the client behaves as upstream. The option changes are these commits on
top of upstream, in order:

- `8e592d22` feat(weka): add --weka-live-assistant replay of live model outputs
- `81faffc3` feat(weka): add --weka-token-ids token-ID replay for exact live joins
- `560e1f60` feat(weka): add --weka-user-turns and --weka-clear-thinking replay
- `403d049c` feat(weka): add --weka-clear-thinking-mix hint-free mixed clear-thinking replay
- `9a6529fb` feat(weka): add --weka-gap-shape idle-gap reshaping replay

Two edits were made when copying: the warm-up switch in `src/aiperf/timing/config.py` (the same change the store
runs applied to the installed client) and the wording of one docstring in `src/aiperf/dataset/live_assistant.py`.
