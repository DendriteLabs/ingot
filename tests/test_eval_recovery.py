"""A failed sample is left out and the eval goes on, within the cap; the draw sizes prompts without
rendering short ones; orphaned state files are swept."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from eval_service import config, run, servers
from eval_service.dataset import Sample
from eval_service.run import Result, _Evaluation

SAMPLES = [Sample(f"s{i}", []) for i in range(5)]
OK: Result = ({}, {"sample_id": "ok"})


class Scripted(_Evaluation):
    """An _Evaluation without servers: its _sample returns, or raises, outcomes[i]."""

    def __init__(self, outcomes: Sequence[Result | Exception]) -> None:
        self.outcomes = outcomes
        self.compare_on, self.preemptions_at_start = {}, 0.0

    def _preemptions(self) -> float:
        return 0.0

    def _sample(self, i: int, sample: Sample) -> Result:
        outcome = self.outcomes[i]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_failed_samples_are_left_out_up_to_the_cap(monkeypatch):
    monkeypatch.setattr(config, "MAX_FAILED_SAMPLES", 1)
    no_room = ValueError("a 58416-token prompt leaves no room")
    assert Scripted([OK, no_room, OK, OK, OK]).run(SAMPLES) == [OK] * 4

    with pytest.raises(RuntimeError, match="2 samples failed"):
        Scripted([OK, no_room, RuntimeError("500 from vLLM"), OK, OK]).run(SAMPLES)

    monkeypatch.setattr(config, "MAX_FAILED_SAMPLES", 10)
    with pytest.raises(RuntimeError, match="no sample was scored"):
        Scripted([no_room] * 5).run(SAMPLES)


class Words:
    """A tokenizer whose tokens are words, with a two-token frame around the messages."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<u> " + " ".join(m["content"] for m in messages) + " </u>"

    def encode(self, text, add_special_tokens):
        return text.split()


def test_fits_keeps_the_whole_answer_budget(monkeypatch):
    monkeypatch.setattr(config, "MAX_MODEL_LEN", 10)
    monkeypatch.setattr(config, "MAX_TOKENS", 4)  # 6 prompt tokens, 2 of them the frame
    check = run.fits(Words())
    assert check(Sample("rendered", [{"role": "user", "content": "a b c"}])), "5 tokens"
    assert not check(Sample("long", [{"role": "user", "content": "a b c d e"}])), "7 tokens"
    assert check(Sample("short in bytes", [{"role": "user", "content": "abcd"}]))


def test_sweep_removes_abandoned_state_files_only(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(servers, "HIDDEN_STATES_DIR", tmp_path)
    old, fresh = tmp_path / "gpu0" / "old.safetensors", tmp_path / "remote" / "fresh.safetensors"
    lone_lock = tmp_path / "gpu1" / "lone.safetensors.lock"
    for path in (old, Path(f"{old}.lock"), fresh, Path(f"{fresh}.lock"), lone_lock):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    stale = time.time() - 2 * config.STATES_TTL_S
    for path in (old, Path(f"{old}.lock"), lone_lock):
        os.utime(path, (stale, stale))

    assert servers.sweep(config.STATES_TTL_S) == 2
    assert fresh.exists() and Path(f"{fresh}.lock").exists(), "a live request's file stays"
    assert not old.exists() and not Path(f"{old}.lock").exists() and not lone_lock.exists()
    assert servers.sweep() == 1 and not fresh.exists(), "at a start nothing is live"
