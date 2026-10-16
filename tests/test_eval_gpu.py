"""The eval on real models: needs the 8-GPU box (scripts/setup_eval_box.sh) and the teacher and
genesis from the hub. `uv run` would sync vLLM away, so:

    .venv/bin/python -m pytest -m gpu tests/test_eval_gpu.py"""

from __future__ import annotations

import pytest

from common import models
from common.config import chain_spec
from eval_service import run, servers
from eval_service.dataset import Sample

SAMPLES = [
    Sample("s1", [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]),
    Sample("s2", [{"role": "user", "content": "Write a Python function that adds two numbers."}]),
    Sample("s3", [{"role": "user", "content": "What is 17 * 23? Show your steps."}]),
]


@pytest.mark.gpu
@pytest.mark.network
def test_genesis_is_close_and_cannot_beat_itself():
    spec = chain_spec()
    teacher = models.snapshot(spec.teacher_repo, spec.teacher_revision)
    genesis = models.snapshot(spec.genesis_repo, spec.genesis_revision)
    try:
        duel = run.duel(teacher, genesis, genesis, SAMPLES)
    finally:
        servers.stop_all()
    assert 0 < duel.challenger.kl < 0.05 and 0 < duel.challenger.dist < 0.2, duel
    assert duel.noise.kl < duel.challenger.kl, "batching noise must stay below fp8's own drift"
    assert not duel.won, "a model cannot beat itself by both margins"
