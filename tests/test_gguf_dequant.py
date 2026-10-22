"""Exact dequantization of the accepted gguf types, the fork's Hadamard rotation, the gguf output
head the eval rebuilds the logits with, and the llama-states group's forced reads."""

from __future__ import annotations

import struct
import threading
from pathlib import Path

import numpy as np
import pytest

from common.gguf import read_header
from common.gguf.dequant import hadamard, rotation, tensor
from common.gguf.reader import ARRAY, BOOL, I32, STRING, U32
from common.gguf.reader import F32 as KV_F32
from common.gguf.types import BF16, F32, PQ2_0, Q8_0, TYPES
from tests.test_gguf import _str

ALIGN = 32


def _kv(key: str, vtype: int, value, elem: int | None = None) -> bytes:
    out = _str(key) + struct.pack("<I", vtype)
    if vtype == STRING:
        return out + _str(value)
    if vtype == ARRAY:
        assert elem is not None
        body = b"".join(_str(v) if elem == STRING else struct.pack("<i", v) for v in value)
        return out + struct.pack("<IQ", elem, len(value)) + body
    return out + struct.pack({KV_F32: "<f", BOOL: "<B"}.get(vtype, "<I"), value)


def write(path: Path, kv: list[bytes], tensors: list[tuple[str, int, tuple[int, ...], bytes]]):
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + b"".join(kv)
    body, offset = b"", 0
    for name, ggml_type, shape, data in tensors:
        head += _str(name) + struct.pack("<I", len(shape))
        head += b"".join(struct.pack("<Q", d) for d in shape)
        head += struct.pack("<IQ", ggml_type, offset)
        pad = b"\0" * (-len(data) % ALIGN)
        body += data + pad
        offset += len(data) + len(pad)
    head += b"\0" * (-len(head) % ALIGN)
    path.write_bytes(head + body)
    return path


def pq2_0(values: np.ndarray, d: float) -> bytes:
    """One row of PQ2_0 blocks of codes in {0,1,2,3} (value (code - 1) * d)."""
    codes = values.reshape(-1, 128).astype(np.uint8)
    out = b""
    for block in codes:
        qs = bytes(int(sum(int(block[4 * i + k]) << (2 * k) for k in range(4))) for i in range(32))
        out += np.float16(d).tobytes() + qs
    return out


def test_every_accepted_type_dequantizes_exactly(tmp_path):
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, (2, 256))
    q8 = rng.integers(-128, 128, (1, 32)).astype(np.int8)
    wide = rng.standard_normal((2, 4)).astype(np.float32)
    bf16 = np.array([1.5, -2.0, 0.15625, 3.0], np.float32)
    path = write(
        tmp_path / "m.gguf",
        [],
        [
            ("p", PQ2_0, (256, 2), b"".join(pq2_0(row, 0.5) for row in codes)),
            ("q", Q8_0, (32, 1), np.float16(0.25).tobytes() + q8.tobytes()),
            ("f", F32, (4, 2), wide.tobytes()),
            ("b", BF16, (4,), (bf16.view(np.uint32) >> 16).astype("<u2").tobytes()),
        ],
    )
    g = read_header(path)
    np.testing.assert_array_equal(tensor(path, g, "p"), (codes - 1) * 0.5)
    np.testing.assert_array_equal(tensor(path, g, "q"), q8.astype(np.float32) * 0.25)
    np.testing.assert_array_equal(tensor(path, g, "f"), wide)
    np.testing.assert_array_equal(tensor(path, g, "b"), bf16)
    assert set(TYPES) == {F32, 1, BF16, Q8_0, PQ2_0}, "a new type needs its dequantizer here"


def test_hadamard_is_the_normalized_sylvester_matrix():
    h2 = np.array([[1, 1], [1, -1]], np.float32)
    want = np.kron(np.kron(h2, h2), h2) / np.sqrt(8)
    np.testing.assert_allclose(hadamard(8), want, atol=1e-7)
    np.testing.assert_allclose(hadamard(1024) @ hadamard(1024), np.eye(1024), atol=1e-5)


def _folded(tmp_path: Path, folded: list[str]) -> Path:
    kv = [
        _kv("general.architecture", STRING, "qwen35"),
        _kv("qwen35.attention.layer_norm_rms_epsilon", KV_F32, 1e-6),
        _kv("prism.hadamard.weight_names", ARRAY, folded, STRING),
        _kv("prism.hadamard.block_size", U32, 4),
        _kv("prism.hadamard.sign_mode", STRING, "explicit"),
        _kv("prism.hadamard.sign_widths", ARRAY, [4, 8], I32),
        _kv("prism.hadamard.sign_values", ARRAY, [1, -1, 1, -1] + [1] * 8, I32),
    ]
    rng = np.random.default_rng(1)
    head = rng.standard_normal((3, 8)).astype(np.float32)
    norm = (1 + rng.random(8)).astype(np.float32)
    path = write(
        tmp_path / "model.gguf",
        kv,
        [
            ("output.weight", F32, (8, 3), head.tobytes()),
            ("output_norm.weight", F32, (8,), norm.tobytes()),
        ],
    )
    return path


def test_rotation_finds_the_sign_vector_of_the_input_width(tmp_path):
    g = read_header(_folded(tmp_path, ["output.weight"]))
    folded = rotation(g, "output.weight")
    assert folded is not None
    block, signs = folded
    assert block == 4 and signs.tolist() == [1.0] * 8
    assert rotation(g, "output_norm.weight") is None


def test_gguf_head_applies_norm_rotation_and_weight(tmp_path):
    torch = pytest.importorskip("torch")
    from eval_service.metrics import GgufHead

    path = _folded(tmp_path, ["output.weight"])
    g = read_header(path)
    w, norm = tensor(path, g, "output.weight"), tensor(path, g, "output_norm.weight")
    h = np.random.default_rng(2).standard_normal((5, 8)).astype(np.float32)
    x = h / np.sqrt((h**2).mean(-1, keepdims=True) + 1e-6) * norm
    x = (x.reshape(5, 2, 4) @ hadamard(4)).reshape(5, 8)  # signs of width 8 are all +1
    logits = x @ w.T
    want = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    got = GgufHead(tmp_path, "cpu").logprobs(torch.from_numpy(h)).numpy()
    np.testing.assert_allclose(got, want, atol=1e-5)


class FakeServer:
    def __init__(self) -> None:
        self.reads: list[tuple[list[int], int]] = []

    def read(self, tokens: list[int], first: int) -> str:
        self.reads.append((tokens, first))
        return "/dev/shm/x.safetensors"


def test_llama_group_reads_the_forced_answer_only():
    from eval_service.servers import LlamaGroup

    group = LlamaGroup.__new__(LlamaGroup)
    fake = FakeServer()
    group.role, group.waiting = "challenger", [0]
    group.servers = [fake]  # ty: ignore[invalid-assignment]
    group.cv = threading.Condition()
    prompt, answer = [1, 2, 3], [4, 5]
    tokens, path = group.complete(prompt, len(answer), 0, prompt + answer)
    assert tokens == answer and path.endswith(".safetensors")
    assert fake.reads == [([1, 2, 3, 4], 2)], "rows prompt + answer[:-1] from the last prompt row"
    with pytest.raises(ValueError, match="only reads"):
        group.complete(prompt, 2, 0)


def test_a_model_that_hangs_while_loading_fails_at_the_deadline(tmp_path):
    import subprocess
    import time

    from eval_service.servers import LlamaServer, ServerExited

    server = LlamaServer.__new__(LlamaServer)
    server.gpu, server.log = 7, tmp_path / "gpu7.log"
    server.log.write_text("")
    # stands in for llama-states stuck in the fork's loader: never prints its ready line
    server.process = subprocess.Popen(
        ["sleep", "60"], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    started = time.monotonic()
    with pytest.raises(ServerExited, match="not loaded before the deadline"):
        server.wait_ready(time.monotonic() + 0.5)
    assert time.monotonic() - started < 30, "the deadline, not the hang, ends the wait"
    assert server.process.poll() is not None, "and the stuck process is killed"


def test_a_read_that_hangs_or_dies_is_blamed_on_its_gpu(tmp_path, monkeypatch):
    import subprocess
    import threading

    from eval_service import config
    from eval_service.servers import LlamaServer, ServerExited

    monkeypatch.setattr(config, "LLAMA_READ_S", 0.5)
    monkeypatch.setattr(config, "LLAMA_S_PER_1K", 0)

    def server(command: list[str]) -> LlamaServer:
        s = LlamaServer.__new__(LlamaServer)
        s.gpu, s.log, s.dir, s.lock = 5, tmp_path / "gpu5.log", tmp_path, threading.Lock()
        s.log.write_text("CUDA error: an illegal memory access")
        s.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        return s

    hung = server(["sleep", "60"])  # takes the request, never answers
    with pytest.raises(ServerExited, match="did not finish") as caught:
        hung.read([1, 2, 3], 1)
    assert caught.value.gpu == 5 and hung.process.poll() is not None
    dead = server(["true"])  # exits at once: a crash mid-eval
    dead.process.wait()
    with pytest.raises(ServerExited, match="illegal memory access"):
        dead.read([1, 2, 3], 1)
