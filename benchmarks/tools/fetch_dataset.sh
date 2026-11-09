#!/bin/bash
# Download the coding-agent traces at the pinned revision into a Hugging Face cache, build the dataset cache the client
# reads with the network off (the client loads the dataset without a revision, which offline works from that cache
# only), point the dataset's main reference at the revision, and check that the client's own loader reads it offline.
#
#   HF_HOME=<cache> [PYTHON=<the client's python>] benchmarks/tools/fetch_dataset.sh
#
# PYTHON defaults to the interpreter of the environment that holds `aiperf` on PATH. With the FlashStates image:
#   docker run --rm -v "$HF_HOME":/hf -e HF_HOME=/hf -v "$PWD":/w -w /w --entrypoint bash IMAGE \
#       -c 'PYTHON=/opt/aiperf/bin/python benchmarks/tools/fetch_dataset.sh'
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=../configs/common.sh
source "$here/configs/common.sh"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
PYTHON=${PYTHON:-$(dirname "$(readlink -f "$(command -v aiperf)")")/python}
env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE "$PYTHON" - "$DATASET_REPO" "$DATASET_REVISION" <<'PY'
import sys

from datasets import load_dataset
from huggingface_hub import snapshot_download

repo, revision = sys.argv[1:]
snapshot_download(repo, repo_type="dataset", revision=revision)
print("rows", len(load_dataset(repo, split="train", revision=revision)))
PY
refs="$HF_HOME/hub/datasets--${DATASET_REPO//\//--}/refs"
mkdir -p "$refs"
printf '%s' "$DATASET_REVISION" > "$refs/main"
HF_DATASETS_OFFLINE=1 "$PYTHON" - "$DATASET_REPO" <<'PY'
import sys

from aiperf.dataset.loader.semianalysis_cc_traces_weka import SemiAnalysisCCTracesWekaLoader

loader = SemiAnalysisCCTracesWekaLoader.__new__(SemiAnalysisCCTracesWekaLoader)
loader.hf_dataset_name, loader.hf_split, loader.hf_subset, loader.streaming = sys.argv[1], "train", None, False
print("offline rows", len(loader._load_hf_dataset()))
PY
echo "$DATASET_REPO at $DATASET_REVISION in $HF_HOME"
