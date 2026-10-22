"""chain.toml: the quants, the open one, and what each format's submission may hold."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from common.config import REPO_ROOT, chain_spec

RAW = (REPO_ROOT / "chain.toml").read_text()


def load(tmp_path: Path, change) -> None:
    raw = tomllib.loads(RAW)
    change(raw)
    path = tmp_path / "chain.toml"
    path.write_text(_dumps(raw))
    chain_spec.__wrapped__(path)


def _dumps(raw: dict, prefix: str = "") -> str:
    """Just enough TOML for chain.toml: tables of strings, ints and string lists."""
    scalars = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    out = f"[{prefix}]\n" if prefix and scalars else ""
    for key, value in scalars.items():
        if isinstance(value, list):
            out += f"{key} = [{', '.join(repr_(v) for v in value)}]\n"
        else:
            out += f"{key} = {repr_(value)}\n"
    for key, value in raw.items():
        if isinstance(value, dict):
            out += _dumps(value, f"{prefix}.{key}" if prefix else key)
    return out


def repr_(value: object) -> str:
    return f'"{value}"' if isinstance(value, str) else str(value)


def test_the_real_chain_toml_opens_q2():
    spec = chain_spec()
    assert (spec.family, spec.quant) == ("qwen38_27b", "q2")
    assert spec.open == spec.pair("qwen38_27b", "q2")
    assert spec.open.format == "gguf" and spec.open.weight_limit == 7_206_168_928
    fp8 = spec.pair("qwen38_27b", "fp8")
    assert (fp8.format, fp8.weight_limit, fp8.genesis_file) == ("safetensors", None, None)
    assert spec.closed("qwen38_27b", "fp8") and not spec.closed("qwen38_27b", "q2")
    assert spec.upload_limit("qwen38_27b", "fp8") == 35_000_000_000
    assert (spec.king_name("qwen38_27b", "fp8"), spec.king_name("qwen38_27b", "q2")) == (
        "27B-FP8",
        "27B-Q2",
    )


def test_an_unlisted_pair_raises():
    with pytest.raises(ValueError, match="qwen38_27b.gptq is not an allowed pair"):
        chain_spec().pair("qwen38_27b", "gptq")


def test_a_copy_of_the_real_chain_toml_loads(tmp_path: Path):
    load(tmp_path, lambda raw: None)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda r: r["quants"]["qwen38_27b"]["q2"].update(king_name="27B-FP8"), "unique"),
        (lambda r: r["quants"]["qwen38_27b"]["q2"].pop("genesis_file"), "genesis_file"),
        (lambda r: r["quants"]["qwen38_27b"]["q2"].update(vision_repo="org/repo"), "repo@revision"),
        (lambda r: r["quants"]["qwen38_27b"]["q2"].update(format="exl2"), "format"),
        (lambda r: r["quants"]["qwen38_27b"]["q2"].update(upload_limit=0), "upload_limit"),
        (lambda r: r["quants"]["qwen38_27b"]["q2"].update(king_name="27B/Q2"), "king_name"),
        (lambda r: r["competition"].update(active_quant="q3"), "does not list"),
    ],
)
def test_a_bad_quant_entry_raises(tmp_path: Path, change, error: str):
    with pytest.raises(ValueError, match=error):
        load(tmp_path, change)


def test_allows_file_per_format():
    spec = chain_spec()
    assert spec.allows_file("model.gguf", "gguf")
    for bad in ("a/model.gguf", "config.json", "model.safetensors"):
        assert not spec.allows_file(bad, "gguf")
    assert spec.allows_file("config.json", "safetensors")
    assert not spec.allows_file("model.gguf", "safetensors")
