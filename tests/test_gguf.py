"""The GGUF reader and container checks on tiny files built by a writer in this test, plus the
real genesis header fetched with HTTP range requests."""

from __future__ import annotations

import struct
import urllib.request
from pathlib import Path

import pytest

from common.gguf import GgufError, check_container, read_header
from common.gguf.reader import ARRAY, STRING, U32
from common.gguf.types import F32, PQ2_0

ALIGN = 32


def _str(s: str | bytes) -> bytes:
    b = s.encode() if isinstance(s, str) else s
    return struct.pack("<Q", len(b)) + b


def _kv(key: str | bytes, vtype: int, value) -> bytes:
    out = _str(key) + struct.pack("<I", vtype)
    if vtype == STRING:
        return out + _str(value)
    if vtype == ARRAY:
        return out + struct.pack("<IQ", STRING, len(value)) + b"".join(_str(v) for v in value)
    return out + struct.pack("<I", value)


def write_gguf(
    path: Path,
    *,
    magic: bytes = b"GGUF",
    version: int = 3,
    kv: list[bytes] | None = None,
    tensors: list[tuple[str, int, tuple[int, ...], int]] | None = None,
    data: bytes | None = None,
    n_tensors: int | None = None,
) -> Path:
    """tensors: (name, ggml type, shape, offset). Without `data`, writes each tensor's bytes
    (0x11) at its offset, zero padding between, and pads the end to the alignment."""
    kv = (
        [_kv("general.alignment", U32, ALIGN), _kv("general.name", STRING, "tiny")]
        if kv is None
        else kv
    )
    if tensors is None:
        tensors = [("a.weight", F32, (4, 2), 0), ("b.weight", PQ2_0, (128, 2), 32)]
    head = magic + struct.pack(
        "<IQQ", version, len(tensors) if n_tensors is None else n_tensors, len(kv)
    )
    head += b"".join(kv)
    for name, ggml_type, shape, offset in tensors:
        head += _str(name) + struct.pack("<I", len(shape))
        head += b"".join(struct.pack("<Q", d) for d in shape)
        head += struct.pack("<IQ", ggml_type, offset)
    head += b"\0" * (-len(head) % ALIGN)
    if data is None:
        body = bytearray()
        for _, ggml_type, shape, offset in tensors:
            size = shape[0] * shape[1] * 4 if ggml_type == F32 else shape[1] * 34
            body += b"\0" * (offset - len(body)) + b"\x11" * size
        data = bytes(body) + b"\0" * (-len(body) % ALIGN)
    path.write_bytes(head + data)
    return path


def check(path: Path) -> str | None:
    try:
        g = read_header(path)
    except GgufError as exc:
        return str(exc)
    return check_container(g, path)


def test_valid_file_passes(tmp_path: Path) -> None:
    p = write_gguf(tmp_path / "m.gguf")
    assert check(p) is None
    g = read_header(p)
    assert g.tensors["b.weight"].shape == (128, 2)
    assert g.kv["general.alignment"].type == U32


def _bad(tmp_path: Path, **kw) -> str | None:
    return check(write_gguf(tmp_path / "m.gguf", **kw))


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        ({"magic": b"GGUG"}, "magic"),
        ({"version": 2}, "version 2"),
        ({"n_tensors": 10_001}, "over the 10000 cap"),
        ({"kv": [_kv("k", STRING, b"\xff\xfe")]}, "UTF-8"),
        ({"kv": [_kv("k", U32, 1), _kv("k", U32, 2)]}, "duplicate metadata key k"),
        ({"kv": [_kv("k", 13, 1)]}, "unknown value type 13"),
        ({"kv": [_kv("general.alignment", U32, 24)]}, "power of two"),
        ({"tensors": [("a", F32, (4, 2), 0), ("a", F32, (4, 2), 32)]}, "duplicate tensor a"),
        ({"tensors": [("a", 2, (32, 1), 0)]}, "ggml type 2 is not allowed"),
        ({"tensors": [("a", F32, (4, 2), 0), ("b", F32, (4, 2), 48)]}, "not a multiple"),
        ({"tensors": [("a", F32, (4, 2), 0), ("b", F32, (4, 2), 64)]}, "gap"),
        ({"tensors": [("a", F32, (16, 2), 0), ("b", F32, (4, 2), 64)]}, "overlaps"),
        ({"tensors": [("a", PQ2_0, (64, 2), 0)]}, "do not split"),
        ({"tensors": [("a", F32, (4, 2), 0)], "data": b"\x11" * 64}, "but the file has"),
        (
            {"tensors": [("a", F32, (4, 1), 0)], "data": b"\x11" * 16 + b"\x00" * 15 + b"\x01"},
            "padding",
        ),
        ({"tensors": [("a", F32, (4, 2), 0)], "data": b"\x11" * 16}, "but the file has"),
    ],
)
def test_bad_file_fails(tmp_path: Path, kw: dict, reason: str) -> None:
    got = _bad(tmp_path, **kw)
    assert got is not None and reason in got, got


def test_truncated_header(tmp_path: Path) -> None:
    p = write_gguf(tmp_path / "m.gguf")
    p.write_bytes(p.read_bytes()[:60])
    assert "truncated" in (check(p) or "")


def test_header_over_budget(tmp_path: Path) -> None:
    p = write_gguf(tmp_path / "m.gguf")
    with pytest.raises(GgufError, match="larger than 40 bytes"):
        read_header(p, max_header=40)


def test_array_over_cap(tmp_path: Path) -> None:
    key = _str("tokenizer.ggml.tokens") + struct.pack("<IIQ", ARRAY, STRING, 4_000_001)
    assert "over the 4000000 cap" in (_bad(tmp_path, kv=[key]) or "")


def test_nonzero_padding_after_header(tmp_path: Path) -> None:
    p = write_gguf(tmp_path / "m.gguf")
    g = read_header(p)
    raw = bytearray(p.read_bytes())
    raw[g.header_end] = 1
    p.write_bytes(bytes(raw))
    assert "padding" in (check(p) or "")


GENESIS_URL = (
    "https://huggingface.co/dendriteholdings/Ingot-genesis-27B-Q2/resolve/"
    "88c0a49cdf41018703181cff534669e113323d67/Ingot-genesis-27B-Q2.gguf"
)
GENESIS_SIZE = 7206168928


def _fetch_header(path: Path) -> None:
    """Fetches a growing prefix until the header parses, into a sparse file of the real size."""
    n = 8 * 1024 * 1024
    while True:
        req = urllib.request.Request(GENESIS_URL, headers={"Range": f"bytes=0-{n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            head = resp.read()
        with open(path, "wb") as f:
            f.write(head)
            f.truncate(GENESIS_SIZE)
        try:
            read_header(path, max_header=len(head))
            return
        except GgufError:
            if n >= 64 * 1024 * 1024:
                raise
            n *= 2


@pytest.mark.network
def test_genesis_header(tmp_path: Path) -> None:
    p = tmp_path / "genesis.gguf"
    _fetch_header(p)
    g = read_header(p)
    assert g.version == 3
    assert len(g.tensors) > 0
    last = max(g.tensors.values(), key=lambda t: t.offset)
    from common.gguf.layout import tensor_bytes

    end = g.data_start + last.offset + (tensor_bytes(last.shape, last.ggml_type) or 0)
    assert end == GENESIS_SIZE
