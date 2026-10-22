"""A safetensors submission's index and shards agree tensor by tensor: a tensor in two shards would
be checked in one while vLLM loads the other."""

from __future__ import annotations

import json
from pathlib import Path

from model_validation.checks import check_index

INDEX = "model.safetensors.index.json"


def shard(path: Path, names: list[str]) -> None:
    header: dict = {"__metadata__": {"format": "pt"}}
    for i, name in enumerate(names):
        header[name] = {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
    head = json.dumps(header).encode()
    path.write_bytes(len(head).to_bytes(8, "little") + head + b"\0\0" * len(names))


def model(root: Path, shards: dict[str, list[str]], weight_map: dict[str, str]) -> Path:
    root.mkdir()
    for name, names in shards.items():
        shard(root / name, names)
    (root / INDEX).write_text(json.dumps({"weight_map": weight_map}))
    return root


def test_each_tensor_once_in_the_shard_the_index_names(tmp_path):
    ok = model(tmp_path / "ok", {"a.safetensors": ["x"], "b.safetensors": ["y"]},
               {"x": "a.safetensors", "y": "b.safetensors"})  # fmt: skip
    assert check_index(ok) is None


def test_a_tensor_in_two_shards_is_rejected(tmp_path):
    dup = model(tmp_path / "dup", {"a.safetensors": ["x", "y"], "z.safetensors": ["y"]},
                {"x": "a.safetensors", "y": "z.safetensors"})  # fmt: skip
    assert check_index(dup) == "y is in both a.safetensors and z.safetensors"


def test_the_index_must_name_each_tensors_real_shard(tmp_path):
    wrong = model(tmp_path / "wrong", {"a.safetensors": ["x"], "b.safetensors": ["y"]},
                  {"x": "b.safetensors", "y": "a.safetensors"})  # fmt: skip
    assert "does not match the shards' tensors: x" in str(check_index(wrong))
    unlisted = model(tmp_path / "unlisted", {"a.safetensors": ["x", "hidden"]},
                     {"x": "a.safetensors"})  # fmt: skip
    assert "does not match the shards' tensors: hidden" in str(check_index(unlisted))
