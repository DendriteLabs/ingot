"""A gguf file as the dedup reads it: the same names()/get()/tensors_hash() as Checkpoint, every
weight dequantized to float32, taken out of the fork's Hadamard folding and put back into the
Hugging Face layout, so it is fingerprinted against the safetensors teacher like any submission.

The fork computes a folded weight W' as y = W' H (s ⊙ x); a looked-up embedding row z becomes
s ⊙ (H z). So W = W' H diag(s), and an embedding row is H z ⊙ s: the same per-block product.

The converter (llama-prism conversion/qwen.py) stores the linear-attention value heads tiled
([rep, n_k, hd]) where Qwen has them grouped by key head ([n_k, rep, hd]): the V rows of
in_proj_qkv, the rows of in_proj_z, in_proj_a and in_proj_b, and the columns of out_proj, except
a folded out_proj, which keeps the grouped order (gdn_v_grouped: the runtime permutes its input
instead). get() undoes that. Norm gains are stored as Qwen's 1 + w; get() returns w.
"""

from __future__ import annotations

import hashlib
import mmap
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from common.gguf import TYPES, read_header
from common.gguf.dequant import ROTATION_PREFIX, hadamard, rotation, tensor
from model_dedup.layout import MIN_DIM

_BLOCK = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_qkv": "linear_attn.in_proj_qkv",
    "attn_gate": "linear_attn.in_proj_z",
    "ssm_alpha": "linear_attn.in_proj_a",
    "ssm_beta": "linear_attn.in_proj_b",
    "ssm_out": "linear_attn.out_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
}
_TOP = {
    "token_embd.weight": "embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "norm.weight",
}
_BLK = re.compile(r"blk\.(\d+)\.(\w+)\.weight")


def canonical(key: str) -> str | None:
    """gguf tensor name → the dedup's name, or None for tensors it ignores (MTP, conv, norms of
    heads, ssm_a/dt)."""
    if key in _TOP:
        return _TOP[key]
    m = _BLK.fullmatch(key)
    if m and m[2] in _BLOCK:
        return f"layers.{m[1]}.{_BLOCK[m[2]]}.weight"
    return None


class GgufCheckpoint:
    def __init__(self, path: Path, device: str) -> None:
        (self.file,) = Path(path).glob("*.gguf")
        self.device = torch.device(device)
        self.g = read_header(self.file)
        kv = self.g.kv
        arch = kv["general.architecture"].value
        # blocks past block_count are the MTP head's, which the dedup ignores as for safetensors
        blocks = int(kv[f"{arch}.block_count"].value)
        self._key = {
            c: k
            for k in self.g.tensors
            if (c := canonical(k)) and not (k.startswith("blk.") and int(k.split(".")[1]) >= blocks)
        }
        grouped = kv.get(ROTATION_PREFIX + "gdn_v_grouped")
        self._grouped = bool(grouped and grouped.value)
        n_v, n_k = (int(kv[f"{arch}.ssm.{k}"].value) for k in ("time_step_rank", "group_count"))
        self._heads = Heads(
            n_k=n_k,
            rep=n_v // n_k,
            k_dim=int(kv[f"{arch}.ssm.state_size"].value),
            v_dim=int(kv[f"{arch}.ssm.inner_size"].value) // n_v,
        )

    def names(self) -> list[str]:
        """The fingerprinted tensors: 2-D, both sides at least MIN_DIM."""
        return sorted(
            name
            for name, key in self._key.items()
            if len(shape := self.g.tensors[key].shape) == 2 and min(shape) >= MIN_DIM
        )

    def get(self, name: str) -> torch.Tensor:
        key = self._key[name]
        w = torch.from_numpy(tensor(self.file, self.g, key)).to(self.device)
        if w.dim() == 1:
            return w - 1 if key.endswith("norm.weight") else w
        folded = rotation(self.g, key)
        if folded is not None:
            block, signs = folded
            h = torch.from_numpy(hadamard(block)).to(self.device)
            rows, width = w.shape
            w = (w.reshape(rows, width // block, block) @ h).reshape(rows, width)
            w = w * torch.from_numpy(signs).to(self.device)
        return self._hf_order(key, w, folded is not None)

    def _hf_order(self, key: str, w: torch.Tensor, folded: bool) -> torch.Tensor:
        hd = self._heads
        if hd.rep == 1 or not key.startswith("blk."):
            return w
        kind = key.split(".")[2]
        if kind == "attn_qkv":
            qk = 2 * hd.n_k * hd.k_dim
            return torch.cat([w[:qk], _untile(w[qk:], 0, hd, hd.v_dim)])
        if kind == "attn_gate":
            return _untile(w, 0, hd, hd.v_dim)
        if kind in ("ssm_alpha", "ssm_beta"):
            return _untile(w, 0, hd, 1)
        if kind == "ssm_out" and not (folded and self._grouped):
            return _untile(w, 1, hd, hd.v_dim)
        return w

    def tensors_hash(self) -> str:
        """sha256 over every tensor's name, type, shape and bytes, sorted by name."""
        digests = []
        with self.file.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for t in self.g.tensors.values():
                kind = TYPES[t.ggml_type]
                n = 1
                for d in t.shape:
                    n *= d
                start = self.g.data_start + t.offset
                h = hashlib.sha256(f"{t.name}|{kind.name}|{list(t.shape)}|".encode())
                h.update(mm[start : start + n // kind.block_elems * kind.block_bytes])
                digests.append(h.hexdigest())
        return hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest()


@dataclass(frozen=True)
class Heads:
    """Linear attention: n_k key heads of k_dim, each shared by rep value heads of v_dim."""

    n_k: int
    rep: int
    k_dim: int
    v_dim: int


def _untile(w: torch.Tensor, dim: int, hd: Heads, head: int) -> torch.Tensor:
    """Value heads along `dim` from the converter's tiled order [rep, n_k, head] back to Qwen's
    grouped [n_k, rep, head] (conversion/qwen.py _reorder_v_heads, inverted)."""
    shape = list(w.shape)
    x = w.reshape(*shape[:dim], hd.rep, hd.n_k, head, *shape[dim + 1 :])
    return x.transpose(dim, dim + 1).contiguous().reshape(shape)
