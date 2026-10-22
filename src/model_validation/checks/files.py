"""A safetensors submission is exactly config.json, model.safetensors.index.json and the
top-level *.safetensors shards the index names; a gguf one is one top-level *.gguf.
`check_files` runs on the bucket listing before download; `check_json` bounds the two json files
and `check_index` matches the shards against the index (both safetensors only)."""

from __future__ import annotations

import json
from pathlib import Path

from common.config import ChainSpec
from common.models import safetensors_header

INDEX = "model.safetensors.index.json"
MAX_JSON_BYTES = 1_000_000
_INDEX_KEYS = {"metadata": dict, "weight_map": dict}
_INDEX_METADATA = {"total_size", "total_parameters"}


def check_files(files: list[str], spec: ChainSpec, fmt: str) -> str | None:
    """`fmt` is the declared pair's format; a gguf submission is one top-level *.gguf file."""
    if fmt == "gguf":
        if nested := [f for f in files if "/" in f]:
            return f"files in subfolders: {', '.join(nested)}"
        if extra := [f for f in files if not spec.allows_file(f, fmt)]:
            return f"unexpected files: {', '.join(extra)}"
        if len(files) != 1:
            return f"a gguf submission is exactly one .gguf file, not {len(files)}"
        return None
    if nested := [f for f in files if "/" in f]:
        return f"files in subfolders: {', '.join(nested)}"
    if missing := [f for f in spec.required_files if f not in files]:
        return f"missing required files: {', '.join(missing)}"
    if not any(f.endswith(".safetensors") for f in files):
        return "no safetensors weights"
    if extra := [f for f in files if not spec.allows_file(f, fmt)]:
        return f"unexpected files: {', '.join(extra)}"
    return None


def check_json(model_dir: Path) -> str | None:
    for name in ("config.json", INDEX):
        if (size := (model_dir / name).stat().st_size) > MAX_JSON_BYTES:
            return f"{name} is {size} bytes, at most {MAX_JSON_BYTES}"
    index = json.loads((model_dir / INDEX).read_text())
    if not isinstance(index, dict) or index.keys() - _INDEX_KEYS.keys():
        return f"{INDEX} may only hold {', '.join(_INDEX_KEYS)}"
    for key, kind in _INDEX_KEYS.items():
        if not isinstance(index.get(key, {}), kind):
            return f"{INDEX} {key} must be an object"
    if extra := sorted(index.get("metadata", {}).keys() - _INDEX_METADATA):
        return f"{INDEX} metadata has unexpected keys: {', '.join(extra)}"
    if not all(isinstance(v, str) for v in index.get("weight_map", {}).values()):
        return f"{INDEX} weight_map values must be shard names"
    return None


def check_index(model_dir: Path) -> str | None:
    """The index and the shards agree tensor by tensor: every tensor is in exactly one shard, the
    one the index names. A name in two shards would be checked, counted and fingerprinted in one
    while the server loads the other."""
    weight_map = json.loads((model_dir / INDEX).read_text())["weight_map"]
    shards = {p.name for p in model_dir.glob("*.safetensors")}
    indexed = set(weight_map.values())
    if missing := sorted(indexed - shards):
        return f"{INDEX} names shards that are missing: {', '.join(missing)}"
    if extra := sorted(shards - indexed):
        return f"safetensors files not in {INDEX}: {', '.join(extra)}"
    found: dict[str, str] = {}
    for shard in sorted(shards):
        for name in safetensors_header(model_dir / shard):
            if name in found:
                return f"{name} is in both {found[name]} and {shard}"
            found[name] = shard
    if found != weight_map:
        names = found.keys() | weight_map.keys()
        wrong = sorted(n for n in names if found.get(n) != weight_map.get(n))
        return f"{INDEX} does not match the shards' tensors: {', '.join(wrong[:5])}"
    return None
