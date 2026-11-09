"""A PQ2_0 (ternary, Hadamard-folded) gguf as a PyTorch Qwen3.5 model, exact or trainable.

Every folded weight is kept as ternary codes {-1,0,1} and FP16 scales (one per 128 inputs per
output row) and reads its input as H (s * x) per 1024-block, as the fork does; the embedding table
is stored latent and restored row by row (s * (H z)) at load. Norms are stored as (1 + w) in the
gguf and as w in transformers; ssm_a is -exp(A_log); the value-side rows of the gated-deltanet
layers are stored in the fork's tiled head order and re-ordered here. Trainable mode keeps a float
latent per code, in units of its group's scale, and a float scale per group, and rounds the latent
to {-1,0,1} with a straight-through gradient: a starting point for your own training, which
pack.py writes back into the file.

    from miner.ternary.model import load_model
    model = load_model(gguf, teacher_dir)  # on the CPU; dispatch to GPUs or shard with FSDP
    python -m miner.ternary.model          # self-test of the transform"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common.gguf import TYPES, read_header
from common.gguf.dequant import pq2_codes, rotation, tensor
from common.gguf.types import PQ2_0

GROUP = 128


def fwht(x: torch.Tensor, block: int) -> torch.Tensor:
    """The normalized Sylvester-Walsh-Hadamard transform, per `block`-long piece of the last dim."""
    shape = x.shape
    n = shape[-1]
    y = x.reshape(-1, n // block, block)
    h = 1
    while h < block:
        y = y.reshape(-1, n // block, block // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return y.reshape(shape) * (1.0 / math.sqrt(block))


class PrismLinear(nn.Module):
    """y = W' (H (s * x)), W' ternary codes times group scales; exact (codes, scales buffers) or
    trainable (latent float weight and float scales, straight-through rounding)."""

    signs: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor

    def __init__(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        signs: torch.Tensor,
        block: int,
        trainable: bool,
        compute_dtype: torch.dtype = torch.bfloat16,
        frozen_as: str = "exact",
    ):
        """frozen_as: "exact" keeps int8 codes and fp16 scales as buffers (the numerics check);
        "bf16" keeps one dequantized bf16 parameter without gradient (shardable under FSDP)."""
        super().__init__()
        out_f, in_f = codes.shape
        self.in_features, self.out_features, self.block, self.compute_dtype = (
            in_f,
            out_f,
            block,
            compute_dtype,
        )
        self.register_buffer("signs", signs.to(compute_dtype), persistent=True)
        self.trainable = trainable
        self.frozen_as = frozen_as
        self.codes_fixed = False  # trainable scales with the codes held (the scales-only control)
        if trainable:
            # the latent code, in units of its group's scale: it starts at the code and the code
            # flips once it has moved half a step
            self.latent = nn.Parameter(codes.float())
            self.scale = nn.Parameter(scales.float())
            self.codes0 = nn.Parameter(codes.to(torch.int8), requires_grad=False)
        elif frozen_as == "bf16":
            w = (codes.to(torch.float32) * scales.float().repeat_interleave(GROUP, dim=1)).to(
                compute_dtype
            )
            self.weight = nn.Parameter(w, requires_grad=False)
        else:
            self.register_buffer("codes", codes.to(torch.int8), persistent=True)
            self.register_buffer("scales", scales.to(torch.float16), persistent=True)

    def quantized(self) -> torch.Tensor:
        """The weight as the kernels see it: codes times scales."""
        if not self.trainable:
            if self.frozen_as == "bf16":
                return self.weight
            return self.codes.to(self.compute_dtype) * self.scales.to(
                self.compute_dtype
            ).repeat_interleave(GROUP, dim=1)
        scale = self.scale.abs().clamp_min(1e-8).repeat_interleave(GROUP, dim=1)
        if self.codes_fixed:
            return (self.codes0.to(torch.float32) * scale).to(self.compute_dtype)
        u = self.latent
        codes = torch.clamp(torch.round(u), -1.0, 1.0)
        # straight-through: the forward pass uses the codes, the gradient reaches the latent
        return ((u + (codes - u).detach()) * scale).to(self.compute_dtype)

    def current_codes(self) -> torch.Tensor:
        with torch.no_grad():
            if not self.trainable:
                return self.codes
            if self.codes_fixed:
                return self.codes0
            return torch.clamp(torch.round(self.latent), -1.0, 1.0).to(torch.int8)

    def flipped(self) -> int:
        return int((self.current_codes() != self.codes0).sum()) if self.trainable else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        xr = fwht(x.float() * self.signs.float(), self.block).to(
            self.compute_dtype
        )  # the transform in fp32
        return F.linear(xr, self.quantized())


def read_pq2(path: Path, g, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Codes as int8 [out, in] in {-1,0,1} and FP16 scales [out, in/128] of a PQ2_0 tensor."""
    t = g.tensors[name]
    if t.ggml_type != PQ2_0:
        raise ValueError(f"{name} is {TYPES[t.ggml_type].name}, not PQ2_0")
    in_f, out_f = int(t.shape[0]), int(np.prod(t.shape[1:]))
    blocks = out_f * in_f // GROUP
    raw = np.memmap(path, np.uint8, "r", g.data_start + t.offset, (blocks * 34,)).reshape(-1, 34)
    codes = np.empty((blocks, GROUP), np.int8)
    step = 1 << 16
    for s in range(0, blocks, step):
        c = pq2_codes(raw[s : s + step])
        if (c == 3).any():
            raise ValueError(f"{name}: code 3 is not ternary")
        codes[s : s + step] = c.astype(np.int8) - 1
    scales = raw[:, :2].copy().view("<f2").reshape(out_f, in_f // GROUP)
    return torch.from_numpy(codes.reshape(out_f, in_f)), torch.from_numpy(scales.astype(np.float16))


def signs_for(g, name: str) -> tuple[int, torch.Tensor]:
    rot = rotation(g, name)
    if rot is None:
        raise ValueError(f"{name}: not a Hadamard-folded weight")
    return rot[0], torch.from_numpy(np.ascontiguousarray(rot[1]))


def tiled_to_grouped(n_rows: int, per_head: int, n_k: int, n_v: int) -> torch.Tensor:
    """HF row h -> gguf row, for value-side rows the fork keeps in tiled head order (head = rep *
    n_k + k) while transformers uses grouped order (head = k * rep + rep_i); `per_head` rows per
    head."""
    rep = n_v // n_k
    idx = torch.arange(n_rows)
    head, within = idx // per_head, idx % per_head
    k, r = head // rep, head % rep
    return (r * n_k + k) * per_head + within


def row_perm(gname: str, text) -> torch.Tensor | None:
    """The gguf->HF row map of a tensor stored in tiled head order, else None. Rows of the z gate,
    beta, alpha, A_log, dt_bias and the v part (after 2 * key_dim rows) of the qkv projection and
    the conv channels."""
    n_k, n_v, hd = (
        text.linear_num_key_heads,
        text.linear_num_value_heads,
        text.linear_value_head_dim,
    )
    key_dim = text.linear_num_key_heads * text.linear_key_head_dim
    value_dim = n_v * hd
    if gname.endswith("attn_gate.weight"):
        return tiled_to_grouped(value_dim, hd, n_k, n_v)
    if gname.endswith("attn_qkv.weight") or gname.endswith("ssm_conv1d.weight"):
        head = torch.arange(2 * key_dim)
        return torch.cat([head, 2 * key_dim + tiled_to_grouped(value_dim, hd, n_k, n_v)])
    if any(
        gname.endswith(x) for x in ("ssm_beta.weight", "ssm_alpha.weight", "ssm_a", "ssm_dt.bias")
    ):
        return tiled_to_grouped(n_v, 1, n_k, n_v)
    return None


def linear_map(layer_types: list[str]) -> dict[str, str]:
    """HF linear module path -> gguf tensor name, for the folded weights."""
    m = {"lm_head": "output.weight"}
    for i, kind in enumerate(layer_types):
        b = f"blk.{i}."
        if kind == "linear_attention":
            m[f"model.layers.{i}.linear_attn.in_proj_qkv"] = b + "attn_qkv.weight"
            m[f"model.layers.{i}.linear_attn.in_proj_z"] = b + "attn_gate.weight"
            m[f"model.layers.{i}.linear_attn.out_proj"] = b + "ssm_out.weight"
        else:
            m[f"model.layers.{i}.self_attn.q_proj"] = b + "attn_q.weight"
            m[f"model.layers.{i}.self_attn.k_proj"] = b + "attn_k.weight"
            m[f"model.layers.{i}.self_attn.v_proj"] = b + "attn_v.weight"
            m[f"model.layers.{i}.self_attn.o_proj"] = b + "attn_output.weight"
        m[f"model.layers.{i}.mlp.gate_proj"] = b + "ffn_gate.weight"
        m[f"model.layers.{i}.mlp.up_proj"] = b + "ffn_up.weight"
        m[f"model.layers.{i}.mlp.down_proj"] = b + "ffn_down.weight"
    return m


def small_map(layer_types: list[str]) -> dict[str, tuple[str, str]]:
    """HF parameter path -> (gguf tensor, transform) for everything that is not a folded linear."""
    m = {"model.norm.weight": ("output_norm.weight", "norm")}
    for i, kind in enumerate(layer_types):
        b = f"blk.{i}."
        m[f"model.layers.{i}.input_layernorm.weight"] = (b + "attn_norm.weight", "norm")
        m[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            b + "post_attention_norm.weight",
            "norm",
        )
        if kind == "linear_attention":
            p = f"model.layers.{i}.linear_attn."
            m[p + "in_proj_b.weight"] = (b + "ssm_beta.weight", "plain")
            m[p + "in_proj_a.weight"] = (b + "ssm_alpha.weight", "plain")
            m[p + "conv1d.weight"] = (b + "ssm_conv1d.weight", "conv")
            m[p + "dt_bias"] = (b + "ssm_dt.bias", "plain")
            m[p + "A_log"] = (b + "ssm_a", "a_log")
            m[p + "norm.weight"] = (b + "ssm_norm.weight", "plain")
        else:
            m[f"model.layers.{i}.self_attn.q_norm.weight"] = (b + "attn_q_norm.weight", "norm")
            m[f"model.layers.{i}.self_attn.k_norm.weight"] = (b + "attn_k_norm.weight", "norm")
    return m


def small_tensors(gguf: Path, g, text) -> dict[str, torch.Tensor]:
    """HF parameter path -> float32 value of every tensor that is not a folded linear, in
    transformers' layout (head order, norm offsets, A_log, conv shape)."""
    out = {}
    for hf_path, (gname, how) in small_map(list(text.layer_types)).items():
        v = torch.from_numpy(tensor(gguf, g, gname))
        perm = row_perm(gname, text)
        if perm is not None:
            v = v[perm]
        if how == "norm":
            v = v - 1.0
        elif how == "a_log":
            v = torch.log(-v)
        elif how == "conv":
            v = v.reshape(v.shape[0], 1, v.shape[1])  # [channels, 1, kernel]
        out[hf_path] = v
    return out


def small_to_gguf(v: torch.Tensor, how: str) -> torch.Tensor:
    """The inverse of small_tensors' transform for one tensor (head order aside)."""
    v = v.float()
    if how == "norm":
        return v + 1.0
    if how == "a_log":
        return -torch.exp(v)
    if how == "conv":
        return v.reshape(v.shape[0], -1)
    return v


def set_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parent = root.get_submodule(".".join(path.split(".")[:-1])) if "." in path else root
    setattr(parent, path.split(".")[-1], module)


def load_model(
    gguf: Path,
    config_dir: Path,
    trainable: bool = False,
    compute_dtype: torch.dtype = torch.bfloat16,
    train_only: set[str] | None = None,
    frozen_as: str = "exact",
    skeleton: bool = False,
    log=print,
):
    """The gguf as a Qwen3_5ForCausalLM on the CPU. `train_only`: HF module paths that get a
    trainable layer when trainable (None = all folded linears). `skeleton`: the same modules with
    empty tensors on the meta device (shapes from the gguf header only), for ranks that receive the
    weights by broadcast."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    g = read_header(gguf)
    cfg = AutoConfig.from_pretrained(config_dir)
    text = getattr(cfg, "text_config", cfg)
    text.dtype = compute_dtype
    with init_empty_weights():
        model = Qwen3_5ForCausalLM._from_config(
            text, attn_implementation="sdpa", dtype=compute_dtype
        )
    layer_types = list(text.layer_types)

    # folded linears
    for hf_path, gname in linear_map(layer_types).items():
        if skeleton:
            t = g.tensors[gname]
            in_f, out_f = int(t.shape[0]), int(np.prod(t.shape[1:]))
            with torch.device("meta"):
                codes, scales = (
                    torch.empty((out_f, in_f), dtype=torch.int8),
                    torch.empty((out_f, in_f // GROUP), dtype=torch.float16),
                )
                rot = rotation(g, gname)
                if rot is None:
                    raise ValueError(f"{gname}: not a Hadamard-folded weight")
                block, signs = rot[0], torch.empty((in_f,), dtype=torch.float32)
            train = trainable and (train_only is None or hf_path in train_only)
            with torch.device("meta"):
                set_module(
                    model,
                    hf_path,
                    PrismLinear(codes, scales, signs, block, train, compute_dtype, frozen_as),
                )
            continue
        codes, scales = read_pq2(gguf, g, gname)
        perm = row_perm(gname, text)
        if perm is not None:
            codes, scales = codes[perm], scales[perm]
        block, signs = signs_for(g, gname)
        train = trainable and (train_only is None or hf_path in train_only)
        set_module(
            model,
            hf_path,
            PrismLinear(codes, scales, signs, block, train, compute_dtype, frozen_as),
        )
    if skeleton:
        with torch.device("meta"):
            model.model.embed_tokens = nn.Embedding(
                text.vocab_size, text.hidden_size, text.pad_token_id, dtype=compute_dtype
            )
            model.model.embed_tokens.weight.requires_grad_(False)
            for hf_path in small_map(layer_types):
                p = dict(model.named_parameters())[hf_path]
                set_param(
                    model,
                    hf_path,
                    nn.Parameter(torch.empty(p.shape, dtype=compute_dtype), requires_grad=False),
                )
        log(f"skeleton of {gguf.name} on the meta device")
        return model
    # the latent embedding: rows restored to the primal basis
    codes, scales = read_pq2(gguf, g, "token_embd.weight")
    block, signs = signs_for(g, "token_embd.weight")
    rows = codes.float() * scales.float().repeat_interleave(GROUP, dim=1)
    emb = torch.empty_like(rows, dtype=compute_dtype)
    step = 8192
    for s in range(0, rows.shape[0], step):
        emb[s : s + step] = (fwht(rows[s : s + step], block) * signs).to(compute_dtype)
    model.model.embed_tokens = nn.Embedding.from_pretrained(
        emb, freeze=True, padding_idx=text.pad_token_id
    )
    # everything else
    params = dict(model.named_parameters())
    for hf_path, v in small_tensors(gguf, g, text).items():
        p = params[hf_path]
        if tuple(p.shape) != tuple(v.shape):
            raise ValueError(
                f"{hf_path}: expected {tuple(p.shape)}, the gguf gives {tuple(v.shape)}"
            )
        set_param(model, hf_path, nn.Parameter(v.to(compute_dtype), requires_grad=False))
    # conv bias, if the class has one
    for i, kind in enumerate(layer_types):
        if kind == "linear_attention":
            conv = model.model.layers[i].linear_attn.conv1d
            if conv.bias is not None:
                conv.bias = nn.Parameter(
                    torch.zeros(conv.bias.shape, dtype=compute_dtype), requires_grad=False
                )
    left = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if left:
        raise RuntimeError(f"{len(left)} parameters never loaded, e.g. {left[:5]}")
    model.model.rotary_emb = type(model.model.rotary_emb)(config=text)  # buffers were on meta
    log(
        f"loaded {gguf.name}: {sum(1 for m in model.modules() if isinstance(m, PrismLinear))} "
        "folded linears, "
        f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f} B parameters materialized"
    )
    return model


def set_param(root: nn.Module, path: str, p: nn.Parameter) -> None:
    mod = root.get_submodule(".".join(path.split(".")[:-1]))
    setattr(mod, path.split(".")[-1], p)


def layer_device_map(model, n_gpus: int) -> dict[str, int]:
    """Decoder layers spread evenly over the GPUs, embedding first, norm and head last."""
    layers = len(model.model.layers)
    per = -(-layers // n_gpus)
    m = {"model.embed_tokens": 0, "model.rotary_emb": 0}
    for i in range(layers):
        m[f"model.layers.{i}"] = min(i // per, n_gpus - 1)
    m["model.norm"] = n_gpus - 1
    m["lm_head"] = n_gpus - 1
    return m


if __name__ == "__main__":
    # self-test of the transform against the reference matrix
    from common.gguf.dequant import hadamard

    for block in (8, 64, 1024):
        x = torch.randn(3, 2 * block, dtype=torch.float64)
        ref = torch.from_numpy(hadamard(block)).double()
        want = torch.cat([x[:, :block] @ ref.T, x[:, block:] @ ref.T], dim=1)
        got = fwht(x, block)
        print(block, "max abs diff", float((want - got).abs().max()))
