"""The dedup reads a gguf model's weights in the teacher's Hugging Face layout: out of the fork's
Hadamard folding and back from the converter's tiled value-head order."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common.gguf.dequant import hadamard  # noqa: E402
from common.gguf.reader import ARRAY, BOOL, I32, STRING, U32  # noqa: E402
from common.gguf.types import F32  # noqa: E402
from model_dedup.gguf_checkpoint import GgufCheckpoint  # noqa: E402
from tests.test_gguf_dequant import _kv, write  # noqa: E402

B = 4  # Hadamard block
N_K, REP, K_DIM, V_DIM = 2, 2, 2, 2  # 2 key heads, each shared by 2 value heads of 2
N_V = N_K * REP
QK, V = 2 * N_K * K_DIM, N_V * V_DIM  # in_proj_qkv rows: q and k, then v


def fold(w: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """W' with W' H diag(s) = W: W diag(s) H (H symmetric and orthonormal, s = ±1)."""
    rows, width = w.shape
    return ((w * signs).reshape(rows, width // B, B) @ hadamard(B)).reshape(rows, width)


def tile(w: np.ndarray, axis: int, head: int) -> np.ndarray:
    """The converter's reorder: value heads grouped [n_k, rep, head] → tiled [rep, n_k, head]."""
    w = np.moveaxis(w, axis, 0)
    rest = w.shape[1:]
    w = w.reshape(N_K, REP, head, *rest).swapaxes(0, 1).reshape(-1, *rest)
    return np.moveaxis(w, 0, axis)


def heads_kv(folded: list[str], grouped: bool) -> list[bytes]:
    kv = [
        _kv("general.architecture", STRING, "qwen35"),
        _kv("qwen35.block_count", U32, 1),
        _kv("qwen35.ssm.time_step_rank", U32, N_V),
        _kv("qwen35.ssm.group_count", U32, N_K),
        _kv("qwen35.ssm.state_size", U32, K_DIM),
        _kv("qwen35.ssm.inner_size", U32, V),
    ]
    if folded:
        kv += [
            _kv("prism.hadamard.weight_names", ARRAY, folded, STRING),
            _kv("prism.hadamard.inverse_weight_names", ARRAY, ["token_embd.weight"], STRING),
            _kv("prism.hadamard.block_size", U32, B),
            _kv("prism.hadamard.sign_mode", STRING, "explicit"),
            _kv("prism.hadamard.sign_widths", ARRAY, [8], I32),
            _kv("prism.hadamard.sign_values", ARRAY, [1, -1, 1, 1, -1, -1, 1, -1], I32),
            _kv("prism.hadamard.gdn_v_grouped", BOOL, int(grouped)),
        ]
    return kv


SIGNS = np.array([1, -1, 1, 1, -1, -1, 1, -1], np.float32)


def test_folded_weights_read_back_in_the_teachers_layout(tmp_path):
    rng = np.random.default_rng(0)
    qkv = rng.standard_normal((QK + V, 8)).astype(np.float32)  # reads the 8-wide residual
    out = rng.standard_normal((8, V)).astype(np.float32)  # reads the 8 value channels
    embd = rng.standard_normal((10, 8)).astype(np.float32)
    norm = (1 + rng.random(8)).astype(np.float32)
    stored_qkv = np.concatenate([qkv[:QK], tile(qkv[QK:], 0, V_DIM)])
    write(
        tmp_path / "m.gguf",
        heads_kv(["blk.0.attn_qkv.weight", "blk.0.ssm_out.weight"], grouped=True),
        [
            ("blk.0.attn_qkv.weight", F32, (8, QK + V), fold(stored_qkv, SIGNS).tobytes()),
            # folded: kept grouped (the runtime permutes the activation instead)
            ("blk.0.ssm_out.weight", F32, (V, 8), fold(out, SIGNS).tobytes()),
            ("token_embd.weight", F32, (8, 10), fold(embd, SIGNS).tobytes()),
            ("blk.0.attn_norm.weight", F32, (8,), norm.tobytes()),
        ],
    )
    ck = GgufCheckpoint(tmp_path, "cpu")
    assert ck.names() == [
        "embed_tokens.weight",
        "layers.0.linear_attn.in_proj_qkv.weight",
        "layers.0.linear_attn.out_proj.weight",
    ]
    for name, want in (
        ("layers.0.linear_attn.in_proj_qkv.weight", qkv),
        ("layers.0.linear_attn.out_proj.weight", out),
        ("embed_tokens.weight", embd),
        ("layers.0.input_layernorm.weight", norm - 1),
    ):
        np.testing.assert_allclose(ck.get(name).numpy(), want, atol=1e-5, err_msg=name)
    assert len(ck.tensors_hash()) == 64


def test_unfolded_value_heads_are_untiled(tmp_path):
    rng = np.random.default_rng(1)
    z = rng.standard_normal((V, 8)).astype(np.float32)
    a = rng.standard_normal((N_V, 8)).astype(np.float32)
    out = rng.standard_normal((8, V)).astype(np.float32)
    write(
        tmp_path / "m.gguf",
        heads_kv([], grouped=False),
        [
            ("blk.0.attn_gate.weight", F32, (8, V), tile(z, 0, V_DIM).tobytes()),
            ("blk.0.ssm_alpha.weight", F32, (8, N_V), tile(a, 0, 1).tobytes()),
            ("blk.0.ssm_out.weight", F32, (V, 8), tile(out, 1, V_DIM).tobytes()),
        ],
    )
    ck = GgufCheckpoint(tmp_path, "cpu")
    for name, want in (
        ("layers.0.linear_attn.in_proj_z.weight", z),
        ("layers.0.linear_attn.in_proj_a.weight", a),
        ("layers.0.linear_attn.out_proj.weight", out),
    ):
        np.testing.assert_array_equal(ck.get(name).numpy(), want, err_msg=name)
