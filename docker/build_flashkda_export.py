"""Build vllm/_flashkda_export_C outside vLLM's CMake, from the same sources, include order and flags as the fork's
cmake/external_projects/flashkda.cmake target `_flashkda_export_C` (sm_100f), for an image without git (CMake's
FetchContent needs it). Writes OUT_DIR/_flashkda_export_C.so and prints its sha256.

    python3 build_flashkda_export.py VLLM_SRC FLASHKDA_SRC CUTLASS_SRC OUT_DIR

VLLM_SRC     the fork's tree (csrc/flashkda_export/, csrc/flashkda_export_registration.cpp, csrc/core/registration.h)
FLASHKDA_SRC FlashKDA at 17a037d98da546deb4591e967cf961a43c034d8b (the commit the fork's CMake fetches)
CUTLASS_SRC  CUTLASS at 5c149f52a436782210263fb2f19b354443a61c6a (FlashKDA's submodule at that commit)
"""

import hashlib
import os
import sys

from torch.utils.cpp_extension import load

vllm_src, flashkda, cutlass, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
os.environ.setdefault("MAX_JOBS", "16")
# NDEBUG comes from CMake's build type (setup.py's RelWithDebInfo); without it CuTe's device asserts stay in and the
# kernels run 12-24 % slower.
defines = ["-DNDEBUG", "-UPy_LIMITED_API", "-DUSE_CUDA", "-DTORCH_TARGET_VERSION=0x020B000000000000ULL"]
module = load(
    name="_flashkda_export_C",
    sources=[f"{vllm_src}/csrc/flashkda_export_registration.cpp", f"{flashkda}/csrc/flash_kda.cpp",
             f"{vllm_src}/csrc/flashkda_export/fwd_launch.cu"],
    extra_include_paths=[f"{vllm_src}/csrc/flashkda_export", f"{flashkda}/csrc/smxx", f"{flashkda}/csrc",
                         f"{cutlass}/include", f"{cutlass}/examples/common", f"{cutlass}/tools/util/include",
                         f"{vllm_src}/csrc"],
    extra_cflags=["-O3", "-std=c++20", "-fvisibility=hidden", *defines],
    extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-std=c++20",
                       "-Xcompiler=-fvisibility=hidden", "-gencode=arch=compute_100f,code=sm_100f", *defines],
    build_directory=out,
    verbose=False,
)
digest = hashlib.sha256(open(module.__file__, "rb").read()).hexdigest()
print("BUILT", module.__file__, digest)
