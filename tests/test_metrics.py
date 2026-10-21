"""The final head keeps its logits in float32: two logits 0.0625 apart stay apart, where bf16
output (a 0.125 step between 16 and 32) rounded them onto the same value.

    .venv/bin/python -m pytest -m gpu tests/test_metrics.py"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_validate import shard

torch = pytest.importorskip("torch")


@pytest.mark.gpu
def test_logits_are_not_rounded_to_bf16(tmp_path: Path):
    from eval_service.metrics import FinalHead

    # after the norm the state is [1, 1]: a logit is its head row's sum, 25 and 25.0625
    head = torch.tensor([[12.5, 12.5], [12.5, 12.5625]], dtype=torch.bfloat16)
    norm = torch.zeros(2, dtype=torch.bfloat16)
    raw = lambda t: t.view(torch.int16).numpy().tobytes()  # noqa: E731
    tensors = {
        "lm_head.weight": ("BF16", [2, 2], raw(head)),
        "model.norm.weight": ("BF16", [2], raw(norm)),
    }
    shard(tmp_path / "a.safetensors", tensors)
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"rms_norm_eps": 1e-6}}))
    weight_map = {"lm_head.weight": "a.safetensors", "model.norm.weight": "a.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    h = torch.tensor([[3.0, 3.0]], dtype=torch.bfloat16, device="cuda")

    lp = FinalHead(tmp_path, "cuda").logprobs(h)
    assert lp.dtype == torch.float32
    assert abs((lp[0, 1] - lp[0, 0]).item() - 0.0625) < 1e-4
    rounded = (h @ head.cuda().T).float()  # what the eval did before: both logits read 25.0
    assert rounded[0, 1] == rounded[0, 0]
