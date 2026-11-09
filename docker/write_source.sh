#!/bin/bash
# Write docker/SOURCE.json: the commits and inputs the B200 image is built from (copied into the image).
#
#   docker/write_source.sh VLLM_COMMIT MOONCAKE_COMMIT [REPO_COMMIT]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
vllm_commit, mooncake_commit = sys.argv[1:3]
repo_commit = sys.argv[3] if len(sys.argv) > 3 else None
dockerfile = Path("docker/Dockerfile.b200").read_text()
arg = lambda name: re.search(rf"^ARG {name}=(\S+)$", dockerfile, re.M).group(1)
overlay = Path("docker/vllm_overlay.txt").read_bytes()
record = {
    "base_image": arg("BASE"),
    "vllm_fork_commit": vllm_commit,
    "mooncake_fork_commit": mooncake_commit,
    "mooncake_wheel": arg("MOONCAKE_WHEEL"),
    "flashkda": {"url": arg("FLASHKDA_URL"), "sha256": arg("FLASHKDA_SHA256")},
    "cutlass": {"url": arg("CUTLASS_URL"), "sha256": arg("CUTLASS_SHA256")},
    "vllm_overlay_sha256": hashlib.sha256(overlay).hexdigest(),
    "vllm_overlay_files": len(overlay.splitlines()),
    "release_repo_commit": repo_commit,
}
Path("docker/SOURCE.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps(record, indent=1))
PY
