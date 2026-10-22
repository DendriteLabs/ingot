"""A q2 submission is genesis's header except the storage type of the weights genesis
quantizes."""

from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import pytest

from common.gguf import read_header
from common.gguf.reader import STRING, U32
from common.gguf.types import BF16, F32, PQ2_0, Q8_0
from model_validation import gguf_checks
from tests.test_gguf import _kv, write_gguf


@pytest.fixture
def genesis(tmp_path, monkeypatch):
    (tmp_path / "genesis").mkdir()
    g = write_gguf(tmp_path / "genesis" / "g.gguf")
    pin = gguf_checks.pinned(read_header(g))
    monkeypatch.setattr(gguf_checks, "genesis", lambda family, quant: pin)
    return pin


def submission(tmp_path: Path, **kw) -> Path:
    (tmp_path / "m").mkdir(exist_ok=True)
    write_gguf(tmp_path / "m" / "model.gguf", **kw)
    return tmp_path / "m"


def run(model_dir: Path, spec, limit: int | None = None) -> str | None:
    pair = spec.pair("qwen38_27b", "q2")
    if limit is not None:
        pair = replace(pair, weight_limit=limit)
    return gguf_checks.validate_gguf(model_dir, pair, "qwen38_27b")


def test_genesis_itself_passes(tmp_path, genesis, spec):
    assert run(submission(tmp_path), spec) is None


@pytest.mark.parametrize(
    ("kv", "reason"),
    [
        ([_kv("general.alignment", U32, 32), _kv("general.name", STRING, "mine")], "differs"),
        ([_kv("general.alignment", U32, 32)], "missing"),
        (
            [
                _kv("general.alignment", U32, 32),
                _kv("general.name", STRING, "tiny"),
                _kv("x.extra", U32, 1),
            ],
            "does not have",
        ),
    ],
)
def test_metadata_is_genesis(tmp_path, genesis, spec, kv, reason):
    assert reason in str(run(submission(tmp_path, kv=kv), spec))


def test_over_the_size_of_genesis(tmp_path, genesis, spec):
    model = submission(tmp_path)
    size = (model / "model.gguf").stat().st_size
    assert "genesis has" in str(run(model, spec, limit=size - 1))
    assert run(model, spec, limit=size) is None


def test_tensors(genesis):
    want = {"a.weight": [F32, [4, 2]], "b.weight": [PQ2_0, [128, 2]]}
    check = gguf_checks.check_tensors
    assert check({"a.weight": [F32, [4, 2]], "b.weight": [Q8_0, [128, 2]]}, want) is None
    assert "keeps it F32" in str(check({**want, "a.weight": [BF16, [4, 2]]}, want))
    assert "shape" in str(check({**want, "b.weight": [PQ2_0, [256, 1]]}, want))
    assert "missing" in str(check({"a.weight": want["a.weight"]}, want))
    assert "does not have" in str(check({**want, "c.weight": [F32, [1]]}, want))


BIG = "a weight of {} in size, at most 10000"


def _patch(model: Path, tensor: str, at: int, raw: bytes) -> None:
    path = model / "model.gguf"
    g = read_header(path)
    data = bytearray(path.read_bytes())
    start = g.data_start + g.tensors[tensor].offset + at
    data[start : start + len(raw)] = raw
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    ("tensor", "at", "raw", "reason"),
    [
        ("a.weight", 4, struct.pack("<f", float("nan")), "a.weight: a value is NaN or infinite"),
        ("a.weight", 0, struct.pack("<f", float("inf")), "a.weight: a value is NaN or infinite"),
        ("b.weight", 34, struct.pack("<e", float("inf")), "b.weight: a scale is NaN or infinite"),
        ("b.weight", 0, struct.pack("<e", 20000.0), "b.weight: " + BIG.format(20000)),
        ("a.weight", 4, struct.pack("<f", -1e5), "a.weight: " + BIG.format(100000)),
    ],
)
def test_values_are_finite_and_scales_bounded(tmp_path, genesis, spec, tensor, at, raw, reason):
    model = submission(tmp_path)
    _patch(model, tensor, at, raw)
    assert run(model, spec) == reason


CODE_3 = "b.weight: a weight uses code 3 (+2); q2 weights are -1, 0 or +1"


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        # one byte holds 4 codes, weight k at bits 2k: 0b10 is +1, 0b11 (+2) only in the top field
        (bytes([0b10_10_10_10]), None),
        (bytes([0b11_00_00_00]), CODE_3),
        (bytes([0b00_00_00_11]), CODE_3),
    ],
)
def test_q2_stays_ternary(tmp_path, genesis, spec, raw, reason):
    model = submission(tmp_path)
    _patch(model, "b.weight", 34 + 2 + 31, raw)  # the last code byte of the second block
    assert run(model, spec) == reason


def test_large_values_within_the_cap_pass(tmp_path, genesis, spec):
    model = submission(tmp_path)
    _patch(model, "b.weight", 0, struct.pack("<e", 9000.0))
    _patch(model, "a.weight", 0, struct.pack("<f", -9999.0))
    assert run(model, spec) is None


def test_the_file_itself_may_be_given(tmp_path, genesis, spec):
    model = submission(tmp_path)
    pair = spec.pair("qwen38_27b", "q2")
    assert gguf_checks.validate_gguf(model / "model.gguf", pair, "qwen38_27b") is None
