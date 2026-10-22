"""The checks for a gguf submission (q2, and a later q1): one file whose header is genesis's
except for the storage type of the weights genesis quantizes. Genesis's facts are pinned in the
family package as genesis_<quant>.json (scripts/pin_gguf.py), as `pinned()` gives any header.

Metadata: every key, its type and value equal genesis's, no key added. That keeps the
architecture, the tokenizer and chat template, and the prism.hadamard rotation (which names the
weights stored rotated) out of the miner's hands. Tensors: genesis's names and shapes; a weight
genesis stores quantized (PQ2_0 in q2) may use any type in common.gguf.TYPES (the fork rotates
activations by tensor name, not by type); one genesis keeps wide keeps its type. Values: every
one finite, every block scale at most SCALE_CAP (check_values). Nothing here
is particular to one quant: a q1 genesis brings its own pinned header."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from common.config import Quant
from common.gguf import TYPES, Gguf, GgufError, check_container, read_header
from common.gguf.types import BF16, F16, F32, MAX_CODE, PQ2_0, SCALE_AT, WIDE

# The largest |value| any weight may hold once dequantized: a block type's |d| × its largest
# code, a wide type's value itself. Above fp16's 65504 the fork's fp16 paths (KV cache, cuBLAS
# casts) overflow to inf and NaN, which a model could arm on rare inputs. Generous on purpose:
# genesis peaks at 0.44 (a PQ2_0 scale) and 139 (ssm_a), and an honest model gains nothing from
# orders of magnitude more, RMSNorm undoing a scale anyway.
VALUE_CAP = 1e4
# PQ2_0 stores each weight as a 2-bit code c, decoded (c - 1) * d: 0, 1, 2 are -1, 0, +1, and the
# fork decodes c = 3 as +2. q2 is a ternary round: a fourth level would give a miner a 4-level
# model at the same bytes, so a PQ2_0 weight may not use code 3 (genesis never does).
TERNARY = {PQ2_0}
CHUNK = 64 << 20


def pinned(g: Gguf) -> dict[str, Any]:
    """A header as plain JSON: scalars by value, arrays by element type, length and sha256."""
    kv: dict[str, Any] = {}
    for key, v in sorted(g.kv.items()):
        if v.elem_type is None:
            kv[key] = [v.type, v.value]
        else:
            raw = json.dumps(v.value, separators=(",", ":"), ensure_ascii=False).encode()
            kv[key] = [v.type, v.elem_type, len(v.value), hashlib.sha256(raw).hexdigest()]
    tensors = {t.name: [t.ggml_type, list(t.shape)] for t in g.tensors.values()}
    return {"kv": kv, "tensors": dict(sorted(tensors.items())), "file_size": g.file_size}


def pin_file(quant: str) -> str:
    return f"genesis_{quant}.json"


@lru_cache
def genesis(family: str, quant: str) -> dict[str, Any]:
    root = files(f"model_validation.families.{family}")
    return json.loads(root.joinpath(pin_file(quant)).read_text())


def validate_gguf(model_dir: Path, pair: Quant, family: str) -> str | None:
    """None for a legal gguf submission, else the reason. check_files has already passed.
    `model_dir` may be the .gguf file itself."""
    return next((reason for _, reason in report(model_dir, pair, family) if reason), None)


def report(model_dir: Path, pair: Quant, family: str) -> list[tuple[str, str | None]]:
    """Each check by name with its reason (None: passed), in order; a header that does not
    parse ends the list, the later checks need it."""
    path = model_dir if model_dir.is_file() else next(model_dir.glob("*.gguf"))
    try:
        g = read_header(path)
    except GgufError as e:
        return [("header", str(e))]
    want, got = genesis(family, pair.name), pinned(g)
    size = None
    if pair.weight_limit is not None and g.file_size > pair.weight_limit:
        size = f"the file is {g.file_size} bytes, genesis has {pair.weight_limit}"
    out: list[tuple[str, str | None]] = [
        ("header", None),
        ("container", check_container(g, path)),
        ("size", size),
        ("metadata", check_metadata(got["kv"], want["kv"])),
        ("tensors", check_tensors(got["tensors"], want["tensors"])),
    ]
    # values read the tensors where the container says they are
    out.append(("values", check_values(path, g) if out[1][1] is None else "not checked"))
    return out


def check_metadata(got: dict[str, Any], want: dict[str, Any]) -> str | None:
    if extra := sorted(got.keys() - want.keys()):
        return f"metadata keys genesis does not have: {', '.join(extra[:5])}"
    if missing := sorted(want.keys() - got.keys()):
        return f"metadata keys missing: {', '.join(missing[:5])}"
    if differ := [k for k in want if got[k] != want[k]]:
        return f"metadata differs from genesis: {', '.join(differ[:5])}"
    return None


def check_tensors(got: dict[str, Any], want: dict[str, Any]) -> str | None:
    if extra := sorted(got.keys() - want.keys()):
        return f"tensors genesis does not have: {', '.join(extra[:5])}"
    if missing := sorted(want.keys() - got.keys()):
        return f"tensors missing: {', '.join(missing[:5])}"
    for name, (want_type, want_shape) in want.items():
        got_type, got_shape = got[name]
        if got_shape != want_shape:
            return f"{name}: shape {got_shape}, genesis has {want_shape}"
        if got_type != want_type and want_type in WIDE:
            return f"{name}: stored as {TYPES[got_type].name}, genesis keeps it {_type(want_type)}"
    return None


def _type(ggml_type: int) -> str:
    return TYPES[ggml_type].name if ggml_type in TYPES else str(ggml_type)


def check_values(path: Path, g: Gguf) -> str | None:
    """Every value finite and at most VALUE_CAP in size once dequantized, and no PQ2_0 code 3: a
    wide tensor read whole, a block type by its fp16 scales (times MAX_CODE) and, for TERNARY
    types, its codes. Read in chunks of CHUNK bytes, so the embedding's millions of blocks cost no
    more memory than a small tensor."""
    for t in g.tensors.values():
        kind = TYPES[t.ggml_type]
        size = math.prod(t.shape) // kind.block_elems * kind.block_bytes
        raw = np.memmap(path, np.uint8, "r", g.data_start + t.offset, (size,))
        step = CHUNK // kind.block_bytes * kind.block_bytes
        for at in range(0, size, step):
            chunk = raw[at : at + step]
            if t.ggml_type in SCALE_AT:
                blocks = chunk.reshape(-1, kind.block_bytes)
                d = blocks[:, SCALE_AT[t.ggml_type] : SCALE_AT[t.ggml_type] + 2].copy().view("<f2")
                values, what = np.abs(d.astype(np.float32)) * MAX_CODE[t.ggml_type], "scale"
                # a 2-bit field is 3 when both its bits are set: bit 2k and bit 2k + 1
                if t.ggml_type in TERNARY and (blocks[:, 2:] & (blocks[:, 2:] >> 1) & 0x55).any():
                    return f"{t.name}: a weight uses code 3 (+2); q2 weights are -1, 0 or +1"
            else:
                values, what = np.abs(_wide(chunk, t.ggml_type)), "value"
            if not np.isfinite(values).all():
                return f"{t.name}: a {what} is NaN or infinite"
            if (big := float(values.max(initial=0.0))) > VALUE_CAP:
                return f"{t.name}: a weight of {big:g} in size, at most {VALUE_CAP:g}"
    return None


def _wide(chunk: np.ndarray, ggml_type: int) -> np.ndarray:
    if ggml_type == F32:
        return chunk.view("<f4")
    if ggml_type == F16:
        return chunk.view("<f2").astype(np.float32)
    if ggml_type == BF16:
        return (chunk.view("<u2").astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"no reader for {TYPES[ggml_type].name}")
