#!/usr/bin/env bash
# Build .venv on the GPU box: vLLM 0.31.0 (CUDA 13 wheels) with PR #56822, ingot with the eval
# extra, and the links FlashInfer's kernel compiler needs. Use it instead of `uv sync` there: a
# sync drops vLLM, which is not in the lock file. Idempotent.
#   bash scripts/setup_eval_box.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for tool in curl patch c++; do
  command -v "$tool" >/dev/null || { echo "missing $tool (apt-get install -y curl patch build-essential)"; exit 1; }
done
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d .venv ] || uv venv --python 3.12 .venv
PY=.venv/bin/python
# --link-mode=copy: the patch below edits vLLM in place, which must not reach uv's cache or other venvs
uv pip install --python $PY --link-mode=copy "vllm==0.31.0" --torch-backend=auto \
  flash-linear-attention ninja
uv pip install --python $PY --link-mode=copy -e ".[eval]"

SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
# PR #56822 bills hidden-state extraction at its own width: 0.72 instead of 12.6 MiB per token
# at 64 layers, without which only a few requests fit a GPU
grep -q _split_kv_cache_groups_to_single_layer "$SP/vllm/v1/core/kv_cache_utils.py" \
  || patch --no-backup-if-mismatch -p1 -d "$SP" < scripts/vllm_pr56822.diff

# The pip CUDA wheels ship libfoo.so.13 without the libfoo.so that FlashInfer's `-lfoo` links against.
CU=$SP/nvidia/cu13
[ -e "$CU/lib64" ] || ln -s lib "$CU/lib64"
(
  cd "$CU/lib"
  for lib in libcudart libnvrtc libcublas libcublasLt; do ln -sfn "$lib.so.13" "$lib.so"; done
  ln -sfn "$(ls libnvrtc-builtins.so.13.* | head -1)" libnvrtc-builtins.so
)
probe=$(mktemp -d)
echo 'int main(){return 0;}' > "$probe/t.c"
c++ "$probe/t.c" -L"$CU/lib" -lcudart -lnvrtc -lcublas -lcublasLt -o "$probe/t"
rm -rf "$probe"

$PY -c "
import torch, vllm
from vllm.v1.core import kv_cache_utils
import eval_service.forcer
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
assert (x @ x).abs().sum() > 0 and hasattr(kv_cache_utils, '_split_kv_cache_groups_to_single_layer')
print('vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.device_count(), 'GPUs')"
echo "EVAL BOX OK"
