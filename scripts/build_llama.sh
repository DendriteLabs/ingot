#!/usr/bin/env bash
# Build the pinned Dendrite llama.cpp fork (llama-prism) with CUDA: the only runtime for
# q2 (PQ2_0) models; stock llama.cpp rejects or garbles them. Binaries land in its build/bin/,
# and the eval's llama-states (src/eval_service/llama_states) in build/llama_states/.
#   bash scripts/build_llama.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in cmake c++ nvcc; do
  command -v "$tool" >/dev/null || { echo "missing $tool"; exit 1; }
done
git submodule update --init llama-prism
want=$(uv run --no-sync python -c "from eval_service.config import LLAMA_CPP_COMMIT; print(LLAMA_CPP_COMMIT)")
got=$(git -C llama-prism rev-parse HEAD)
[ "$got" = "$want" ] || { echo "llama-prism is at $got, eval_service.config pins $want"; exit 1; }
cmake -S llama-prism -B llama-prism/build \
  -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build llama-prism/build -j "$(nproc)"
cmake -S src/eval_service/llama_states -B build/llama_states \
  -DLLAMA_DIR="$PWD/llama-prism" -DCMAKE_BUILD_TYPE=Release
cmake --build build/llama_states
echo "LLAMA OK $(llama-prism/build/bin/llama-cli --version 2>&1 | head -1)"
