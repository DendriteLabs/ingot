"""The genesis tensors, re-quantized as the miner likes: every weight keeps its name and
shape; a text-model weight may be BF16 or F32, or FP8 where genesis has it in FP8 (vLLM serves
nothing else quantized: an FP8 embedding does not load), with the projections vLLM fuses all in
one precision; tensors the eval never exercises keep the genesis dtype. An FP8 weight carries
`<name>_scale_inv` shaped by the declared block (two ints >= 128), and
config.modules_to_not_convert must agree with the file: vLLM builds each module from that list,
so a mismatch does not load. Weight values are free."""

from __future__ import annotations

import math
from typing import Any

from model_validation.families import Family, Layout

MIN_BLOCK = 128
FP8 = "F8_E4M3"
SIDE_SUFFIX = "_scale_inv"
WEIGHT_DTYPES = frozenset({FP8, "BF16", "F32"})
SCALE_DTYPES = frozenset({"BF16", "F32"})


def check_layout(layout: Layout, family: Family, config: dict[str, Any]) -> str | None:
    try:
        block = block_size(config)
    except ValueError as exc:
        return str(exc)
    weights = {n: v for n, v in layout.items() if not n.endswith(SIDE_SUFFIX)}
    sides = {n: v for n, v in layout.items() if n.endswith(SIDE_SUFFIX)}
    genesis = {n: v for n, v in family.genesis_layout.items() if not n.endswith(SIDE_SUFFIX)}
    if missing := sorted(genesis.keys() - weights.keys()):
        return f"missing tensors: {_some(missing)}"
    if extra := sorted(weights.keys() - genesis.keys()):
        return f"unexpected tensors: {_some(extra)}"
    skip = _not_converted(config)
    for name, (dtype, shape) in weights.items():
        g_dtype, g_shape = genesis[name]
        if shape != g_shape:
            return f"{name}: shape {list(shape)}, genesis has {list(g_shape)}"
        if family.dtype_frozen(name):
            if dtype != g_dtype:
                return f"{name}: dtype {dtype}, genesis has {g_dtype} (frozen part)"
        elif dtype not in WEIGHT_DTYPES or (dtype == FP8 and len(shape) != 2):
            return f"{name}: dtype {dtype} is not allowed"
        side = sides.pop(name + SIDE_SUFFIX, None)
        module = name.removesuffix(".weight")
        if dtype == FP8:
            if g_dtype != FP8:
                return f"{name}: FP8, but genesis keeps it {g_dtype}; vLLM cannot load it quantized"
            if side is None:
                return f"{name}: FP8 weight without {SIDE_SUFFIX}"
            if side[0] not in SCALE_DTYPES:
                return f"{name}{SIDE_SUFFIX}: dtype {side[0]} is not allowed"
            expected = tuple(math.ceil(n / b) for n, b in zip(shape, block, strict=True))
            if side[1] != expected:
                return (
                    f"{name}{SIDE_SUFFIX}: shape {list(side[1])} does not match block {list(block)}"
                )
            if _covered(module, skip):
                return f"{name}: FP8 but listed in modules_to_not_convert"
        else:
            if side is not None:
                return f"{name}{SIDE_SUFFIX}: scale on a non-FP8 weight"
            if g_dtype == FP8 and not _covered(module, skip):
                return f"{name}: not FP8 but missing from modules_to_not_convert"
    if sides:
        return f"scale tensors without a weight: {_some(sorted(sides))}"
    return _split_fused(weights, family.fused)


def _split_fused(weights: Layout, fused: tuple[tuple[str, ...], ...]) -> str | None:
    """A group vLLM loads as one layer with only some members FP8 does not load."""
    groups: dict[tuple[str, tuple[str, ...]], dict[str, bool]] = {}
    for name, (dtype, _shape) in weights.items():
        module, _, leaf = name.removesuffix(".weight").rpartition(".")
        for group in fused:
            if leaf in group:
                groups.setdefault((module, group), {})[leaf] = dtype == FP8
    for (module, group), fp8 in sorted(groups.items()):
        if len(set(fp8.values())) > 1:
            parts = ", ".join(
                f"{leaf} {'FP8' if fp8[leaf] else 'not'}" for leaf in group if leaf in fp8
            )
            return f"{module}: {parts}; vLLM loads them as one layer, so all FP8 or none"
    return None


def block_size(config: dict[str, Any]) -> tuple[int, int]:
    block = config.get("quantization_config", {}).get("weight_block_size")
    if (
        not isinstance(block, list)
        or len(block) != 2
        or not all(isinstance(b, int) and b >= MIN_BLOCK for b in block)
    ):
        raise ValueError(
            f"quantization_config.weight_block_size is {block!r}, need two ints >= {MIN_BLOCK}"
        )
    return block[0], block[1]


def _not_converted(config: dict[str, Any]) -> tuple[str, ...]:
    listed = config.get("quantization_config", {}).get("modules_to_not_convert") or []
    return tuple(x for x in listed if isinstance(x, str))


def _covered(module: str, skip: tuple[str, ...]) -> bool:
    return any(module == s or module.startswith(s + ".") for s in skip)


def _some(names: list[str], limit: int = 3) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} (+{len(names) - limit} more)"
