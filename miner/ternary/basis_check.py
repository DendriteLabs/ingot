"""Which basis and head order are the folded weights in? De-rotate a few gguf tensors every
plausible way and compare them row by row with the teacher's BF16 weights (the right one
correlates, the others do not); then, for the gated-deltanet value-side rows, test the fork's
tiled head order against transformers' grouped order. Run this on any new base before trusting
the loader's conventions.

    .venv/bin/python -m miner.ternary.basis_check --gguf genesis.gguf"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.gguf import read_header
from common.gguf.dequant import tensor
from miner.ternary.model import GROUP, fwht, read_pq2, signs_for, tiled_to_grouped

PAIRS = {  # gguf tensor -> HF parameter
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "blk.0.attn_qkv.weight": "model.layers.0.linear_attn.in_proj_qkv.weight",
    "blk.0.ssm_out.weight": "model.layers.0.linear_attn.out_proj.weight",
    "blk.0.ffn_down.weight": "model.layers.0.mlp.down_proj.weight",
    "blk.3.attn_q.weight": "model.layers.3.self_attn.q_proj.weight",
}


def hf_tensor(teacher_dir: Path, name: str) -> torch.Tensor:
    from safetensors import safe_open

    index = json.loads((teacher_dir / "model.safetensors.index.json").read_text())["weight_map"]
    for cand in (
        name,
        "model.language_model." + name.removeprefix("model."),
        "language_model." + name,
    ):
        if cand in index:
            with safe_open(teacher_dir / index[cand], "pt") as f:
                return f.get_tensor(cand).float()
    raise KeyError(f"{name} not in the teacher index; examples: {list(index)[:5]}")


def cos_rows(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1).mean())


def teacher_snapshot() -> Path:
    from common import models
    from common.config import chain_spec

    spec = chain_spec()
    return models.snapshot(spec.teacher_repo, spec.teacher_revision)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, help="default: the teacher in chain.toml")
    p.add_argument("--rows", type=int, default=2048)
    a = p.parse_args()
    teacher_dir = a.teacher_dir or teacher_snapshot()
    g = read_header(a.gguf)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    def derot(gname, rows=None):
        codes, scales = read_pq2(a.gguf, g, gname)
        block, signs = signs_for(g, gname)
        if rows:
            codes, scales = codes[:rows], scales[:rows]
        w = (codes.float() * scales.float().repeat_interleave(GROUP, dim=1)).to(dev)
        return w, signs.to(dev), block

    print("basis: stored row z vs the teacher's row (mean cosine over the first rows)")
    for gname, hname in PAIRS.items():
        z, signs, block = derot(gname, a.rows)
        ref = hf_tensor(teacher_dir, hname)[: a.rows].to(dev)
        cands = {
            "z": z,
            "s*(Hz)": fwht(z, block) * signs,
            "H(s*z)": fwht(z * signs, block),
            "Hz": fwht(z, block),
        }
        print(
            f"  {gname:24} " + "  ".join(f"{k} {cos_rows(v, ref):+.3f}" for k, v in cands.items()),
            flush=True,
        )

    print(
        "head order of the value-side rows: identity vs the tiled map "
        "(fraction of rows whose best match agrees)"
    )
    n_k, n_v, hd = 16, 48, 128
    for gname, hname, start, per_head in (
        ("blk.0.attn_gate.weight", "model.layers.0.linear_attn.in_proj_z.weight", 0, hd),
        (
            "blk.0.attn_qkv.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            2 * n_k * hd,
            hd,
        ),
        ("blk.0.ssm_beta.weight", "model.layers.0.linear_attn.in_proj_b.weight", 0, 1),
    ):
        if gname.endswith("ssm_beta.weight"):
            z = torch.from_numpy(tensor(a.gguf, g, gname)).to(dev)
        else:
            w, signs, block = derot(gname)
            z = fwht(w, block) * signs
        z = z[start:]
        ref = hf_tensor(teacher_dir, hname)[start:].to(dev)
        sim = torch.nn.functional.normalize(z, dim=1) @ torch.nn.functional.normalize(ref, dim=1).T
        best = sim.argmax(1).cpu()
        ident = torch.arange(z.shape[0])
        perm = tiled_to_grouped(z.shape[0], per_head, n_k, n_v)
        inv = torch.empty_like(perm)
        inv[perm] = ident
        print(
            f"  {gname:24} identity {(best == ident).float().mean():.3f}  "
            f"tiled {(best == inv).float().mean():.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
