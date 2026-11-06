"""Pack new codes, scales and small tensors into the gguf container: a byte copy of the base file
with the data of the changed tensors overwritten in place, so names, shapes, types, metadata and
size stay those of the base. Rows the loader re-ordered (the tiled value-side tensors) are put
back in the fork's order.

From Python, after training a model loaded with model.load_model(..., trainable=True):

    from miner.ternary.pack import delta_from_model, pack
    pack(Path("king.gguf"), delta_from_model(model), Path("mine/model.gguf"), model.config)

From a saved delta (safetensors with "<gguf tensor>:codes" int8 [out, in] and "<gguf
tensor>:scales" fp16 [out, in/128] in transformers' row order, and "small:<parameter>" floats):

    .venv/bin/python -m miner.ternary.pack --base king.gguf --delta delta.safetensors \\
        --out mine/model.gguf
    .venv/bin/ingot check-model --path mine/model.gguf"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from common.gguf import TYPES, read_header
from common.gguf.types import BF16, F32, PQ2_0
from miner.ternary.model import (
    GROUP,
    PrismLinear,
    linear_map,
    read_pq2,
    row_perm,
    small_map,
    small_tensors,
    small_to_gguf,
)


def delta_from_model(model) -> dict[str, torch.Tensor]:
    """The values to pack from a model loaded with load_model(..., trainable=True) in one process:
    each trainable linear's current codes and scales, and every small tensor that has a gradient."""
    layer_types = list(model.config.layer_types)
    out: dict[str, torch.Tensor] = {}
    for hf_path, gname in linear_map(layer_types).items():
        m = model.get_submodule(hf_path)
        if isinstance(m, PrismLinear) and m.trainable:
            out[gname + ":codes"] = m.current_codes().cpu()
            out[gname + ":scales"] = m.scale.detach().abs().to(torch.float16).cpu()
    small = set(small_map(layer_types))
    for name, p in model.named_parameters():
        if name in small and p.requires_grad:
            out["small:" + name] = p.detach().float().cpu()
    return out


def pack_pq2(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """[out, in] codes in {-1,0,1} and [out, in/128] fp16 scales -> block bytes [n_blocks, 34]."""
    out_f, in_f = codes.shape
    c = (codes.astype(np.int16) + 1).astype(np.uint8).reshape(out_f * in_f // GROUP, GROUP // 4, 4)
    qs = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    blocks = np.empty((qs.shape[0], 34), np.uint8)
    blocks[:, :2] = scales.astype("<f2").reshape(-1, 1).view(np.uint8)
    blocks[:, 2:] = qs
    return blocks


def unpermute(v: np.ndarray, gname: str, text) -> np.ndarray:
    perm = row_perm(gname, text)
    if perm is None:
        return v
    back = np.empty_like(v)
    back[perm.numpy()] = v
    return back


def pack(base: Path, delta: dict[str, torch.Tensor], out: Path, text) -> int:
    """Write `delta` into a byte copy of `base` at `out`, check the round trip, and return the
    number of tensors written. `text` is the model's text config (layer types, head counts)."""
    g = read_header(base)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base, out)
    written = 0
    with open(out, "r+b") as f:
        for key, val in delta.items():
            if key.endswith(":codes"):
                gname = key[: -len(":codes")]
                t = g.tensors[gname]
                if t.ggml_type != PQ2_0:
                    raise ValueError(f"{gname} is not PQ2_0 in the base")
                codes = unpermute(val.numpy(), gname, text)
                scales = unpermute(delta[gname + ":scales"].float().numpy(), gname, text)
                in_f, out_f = int(t.shape[0]), int(np.prod(t.shape[1:]))
                if codes.shape != (out_f, in_f):
                    raise ValueError(f"{gname}: codes {codes.shape}, tensor {(out_f, in_f)}")
                data = pack_pq2(codes, scales).tobytes()
            elif key.startswith("small:"):
                gname, how = small_map(list(text.layer_types))[key[len("small:") :]]
                t = g.tensors[gname]
                v = unpermute(small_to_gguf(val, how).numpy(), gname, text)
                if t.ggml_type == F32:
                    data = v.astype("<f4").tobytes()
                elif t.ggml_type == BF16:
                    bf = torch.from_numpy(np.ascontiguousarray(v)).to(torch.bfloat16)
                    data = bf.view(torch.int16).numpy().astype("<i2").tobytes()
                else:
                    raise ValueError(f"{gname} is {TYPES[t.ggml_type].name}, expected F32 or BF16")
            else:
                continue
            slot = (
                int(np.prod(t.shape))
                // TYPES[t.ggml_type].block_elems
                * TYPES[t.ggml_type].block_bytes
            )
            if len(data) != slot:
                raise ValueError(f"{gname}: {len(data)} bytes for a {slot}-byte slot")
            f.seek(g.data_start + t.offset)
            f.write(data)
            written += 1
    check(out, delta, text)
    return written


def check(out: Path, delta: dict[str, torch.Tensor], text) -> None:
    """Read the packed values back: codes and scales exactly, small tensors to bf16 rounding."""
    g = read_header(out)
    for key, val in delta.items():
        if key.endswith(":codes"):
            gname = key[: -len(":codes")]
            codes, scales = read_pq2(out, g, gname)
            perm = row_perm(gname, text)
            if perm is not None:
                codes, scales = codes[perm], scales[perm]
            want = delta[gname + ":scales"].to(torch.float16)
            if not (torch.equal(codes, val.to(torch.int8)) and torch.equal(scales, want)):
                raise ValueError(f"{gname}: the packed codes or scales do not read back")
    small = {k[len("small:") :]: v for k, v in delta.items() if k.startswith("small:")}
    if small:
        back = small_tensors(out, g, text)
        worst = max(float((back[k].float() - v.float()).abs().max()) for k, v in small.items())
        if worst > 1e-2:
            raise ValueError(f"the small tensors read back {worst:.2e} off")


def teacher_snapshot() -> Path:
    from common import models
    from common.config import chain_spec

    spec = chain_spec()
    return models.snapshot(spec.teacher_repo, spec.teacher_revision)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--delta", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, help="default: the teacher in chain.toml")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from safetensors.torch import load_file
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(a.teacher_dir or teacher_snapshot())
    text = getattr(config, "text_config", config)
    n = pack(a.base, load_file(str(a.delta)), a.out, text)
    print(f"{n} tensors written into a copy of {a.base.name} -> {a.out}, read back ok")


if __name__ == "__main__":
    main()
