"""Our own strict GGUF v3 header parser. Every count and length is bounded before anything is
allocated, so a hostile header fails with a reason instead of exhausting memory. Arrays may
hold up to 4M items: Qwen3.5's tokenizer has ~248k tokens and a similar number of merges, so
4M leaves room without letting an array dominate the 64 MB header budget."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from common.gguf.types import TYPES

MAGIC = b"GGUF"
VERSION = 3
DEFAULT_ALIGNMENT = 32
MAX_HEADER = 64 * 1024 * 1024
MAX_TENSORS = 10_000
MAX_KV = 10_000
MAX_STRING = 1024 * 1024
MAX_ARRAY = 4_000_000
MAX_DIMS = 4

U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
_SCALARS: dict[int, str] = {
    U8: "<B",
    I8: "<b",
    U16: "<H",
    I16: "<h",
    U32: "<I",
    I32: "<i",
    F32: "<f",
    BOOL: "<B",
    U64: "<Q",
    I64: "<q",
    F64: "<d",
}


class GgufError(ValueError):
    """A header the miner must fix; the message is the reason shown to them."""


@dataclass(frozen=True)
class Value:
    type: int
    value: Any
    elem_type: int | None = None  # set for arrays


@dataclass(frozen=True)
class TensorInfo:
    name: str
    ggml_type: int
    shape: tuple[int, ...]  # GGUF order: dim 0 is the contiguous row
    offset: int  # relative to data_start


@dataclass
class Gguf:
    version: int
    alignment: int
    kv: dict[str, Value]
    tensors: dict[str, TensorInfo]
    header_end: int
    data_start: int
    file_size: int


class _Reader:
    def __init__(self, f: BinaryIO, limit: int) -> None:
        self.f = f
        self.pos = 0
        self.limit = limit

    def take(self, n: int) -> bytes:
        if self.pos + n > self.limit:
            raise GgufError(
                f"the header is larger than {self.limit} bytes or the file is truncated"
            )
        data = self.f.read(n)
        if len(data) != n:
            raise GgufError("the file is truncated inside the header")
        self.pos += n
        return data

    def scalar(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.scalar("<Q")
        if n > MAX_STRING:
            raise GgufError(f"a string of {n} bytes is over the {MAX_STRING}-byte cap")
        try:
            return self.take(n).decode("utf-8")
        except UnicodeDecodeError:
            raise GgufError("a string is not valid UTF-8") from None

    def value(self, vtype: int, key: str) -> Value:
        if vtype == STRING:
            return Value(vtype, self.string())
        if vtype in _SCALARS:
            v = self.scalar(_SCALARS[vtype])
            if vtype == BOOL:
                if v > 1:
                    raise GgufError(f"{key}: a bool is {v}, not 0 or 1")
                v = bool(v)
            return Value(vtype, v)
        if vtype == ARRAY:
            etype = self.scalar("<I")
            n = self.scalar("<Q")
            if n > MAX_ARRAY:
                raise GgufError(f"{key}: an array of {n} items is over the {MAX_ARRAY} cap")
            if etype == STRING:
                return Value(vtype, [self.string() for _ in range(n)], etype)
            if etype not in _SCALARS:
                raise GgufError(f"{key}: array element type {etype} is not allowed")
            items = [self.value(etype, key).value for _ in range(n)]
            return Value(vtype, items, etype)
        raise GgufError(f"{key}: unknown value type {vtype}")


def read_header(path: str | Path, *, max_header: int = MAX_HEADER) -> Gguf:
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        r = _Reader(f, min(max_header, file_size))
        if r.take(4) != MAGIC:
            raise GgufError("not a GGUF file (the magic is not GGUF)")
        version = r.scalar("<I")
        if version != VERSION:
            raise GgufError(f"GGUF version {version}; only version 3 little-endian is accepted")
        n_tensors = r.scalar("<Q")
        n_kv = r.scalar("<Q")
        if n_tensors > MAX_TENSORS:
            raise GgufError(f"{n_tensors} tensors is over the {MAX_TENSORS} cap")
        if n_kv > MAX_KV:
            raise GgufError(f"{n_kv} metadata keys is over the {MAX_KV} cap")

        kv: dict[str, Value] = {}
        for _ in range(n_kv):
            key = r.string()
            if key in kv:
                raise GgufError(f"duplicate metadata key {key}")
            kv[key] = r.value(r.scalar("<I"), key)

        tensors: dict[str, TensorInfo] = {}
        for _ in range(n_tensors):
            name = r.string()
            if name in tensors:
                raise GgufError(f"duplicate tensor {name}")
            n_dims = r.scalar("<I")
            if not 1 <= n_dims <= MAX_DIMS:
                raise GgufError(f"{name}: {n_dims} dimensions, expected 1 to {MAX_DIMS}")
            shape = tuple(r.scalar("<Q") for _ in range(n_dims))
            if any(d == 0 for d in shape):
                raise GgufError(f"{name}: a dimension is zero")
            ggml_type = r.scalar("<I")
            if ggml_type not in TYPES:
                raise GgufError(f"{name}: ggml type {ggml_type} is not allowed")
            tensors[name] = TensorInfo(name, ggml_type, shape, r.scalar("<Q"))

    alignment = DEFAULT_ALIGNMENT
    if (a := kv.get("general.alignment")) is not None:
        if a.type != U32 or a.value == 0 or a.value & (a.value - 1):
            raise GgufError("general.alignment must be a u32 power of two")
        alignment = a.value
    data_start = -(-r.pos // alignment) * alignment
    return Gguf(version, alignment, kv, tensors, r.pos, data_start, file_size)
