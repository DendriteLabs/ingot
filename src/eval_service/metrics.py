"""How a model's answer tokens compare with the teacher's: next-token KL and top-1 agreement from
the last layer's state through each model's own head. Summed per sample; the caller averages."""

from __future__ import annotations

import fcntl
import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from eval_service import config

# rows per GPU step: the comparison shares its GPU with a server and has only a few GiB
CHUNK = 64


class FinalHead:
    """Final RMSNorm + lm_head of a checkpoint, applied to last-layer states. The norm is
    zero-centred, y = x * rsqrt(mean(x²) + eps) * (1 + w), in float32."""

    def __init__(self, model_dir: Path, device: str) -> None:
        cfg = json.loads((model_dir / "config.json").read_text())
        self.eps = cfg.get("text_config", cfg)["rms_norm_eps"]
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]

        def tensor(pattern: str) -> torch.Tensor:
            key = next(k for k in index if re.fullmatch(pattern, k))
            with safe_open(str(model_dir / index[key]), "pt", device=device) as f:
                return f.get_tensor(key)

        self.norm = tensor(r"(model\.)?(language_model\.)?norm\.weight").float()
        # BF16 by model validation (lm_head's dtype is frozen): the eval reproduces the logits
        # from the last hidden state, which a quantized head would not allow
        self.head = tensor(r"(model\.)?(language_model\.)?lm_head\.weight")

    def same(self, other: FinalHead) -> bool:
        return torch.equal(self.head, other.head) and torch.equal(self.norm, other.norm)

    def logprobs(self, h: torch.Tensor) -> torch.Tensor:
        h = h.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.norm)
        logits = torch.mm(h.to(torch.bfloat16), self.head.T, out_dtype=torch.float32)
        return torch.log_softmax(logits, -1)


class GgufHead:
    """The output head of a gguf model as the llama-prism fork applies it to the last layer's
    state: RMSNorm with the stored gain (gguf stores Qwen's 1 + w), the prism.hadamard rotation of
    the input side when output.weight is stored folded, then output.weight. All in float32 from
    the exact dequantized weight: llama-states writes only hidden states, the logits are made
    here on the fly."""

    def __init__(self, model_dir: Path, device: str) -> None:
        eps, norm, head, rot = _gguf_head(model_dir)
        self.eps = eps
        self.norm, self.head = norm.to(device), head.to(device)
        self.rot: tuple[torch.Tensor, torch.Tensor] | None = None
        if rot is not None:
            self.rot = (rot[0].to(device), rot[1].to(device))

    def logprobs(self, h: torch.Tensor) -> torch.Tensor:
        h = h.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm
        if self.rot is not None:
            hm, signs = self.rot
            rows, width = h.shape
            h = ((h * signs).reshape(rows, width // hm.shape[0], hm.shape[0]) @ hm).reshape(
                rows, width
            )
        with _no_tf32():
            logits = torch.mm(h, self.head.T)
        return torch.log_softmax(logits, -1)


# The king's model usually faces several challengers in a row: its head is dequantized once, on the
# CPU, and copied to each comparison GPU from there.
@functools.lru_cache(maxsize=3)
def _gguf_head(
    model_dir: Path,
) -> tuple[float, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    from common.gguf import read_header
    from common.gguf.dequant import hadamard, rotation, tensor

    (path,) = model_dir.glob("*.gguf")
    g = read_header(path)
    if "output.weight" not in g.tensors:
        raise ValueError(f"{path.name}: no output.weight (a tied head is not supported)")
    arch = g.kv["general.architecture"].value
    eps = float(g.kv[f"{arch}.attention.layer_norm_rms_epsilon"].value)
    norm = torch.from_numpy(tensor(path, g, "output_norm.weight"))
    head = torch.from_numpy(tensor(path, g, "output.weight"))
    folded = rotation(g, "output.weight")
    rot = None
    if folded:
        block, signs = folded
        rot = (torch.from_numpy(hadamard(block)), torch.from_numpy(signs))
    return eps, norm, head, rot


class _no_tf32:
    def __enter__(self) -> None:
        self.was = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def __exit__(self, *exc: object) -> None:
        torch.backends.cuda.matmul.allow_tf32 = self.was


def head(model_dir: Path, device: str) -> FinalHead | GgufHead:
    """The final head of a safetensors checkpoint or of a gguf file's folder."""
    if any(model_dir.glob("*.gguf")):
        return GgufHead(model_dir, device)
    return FinalHead(model_dir, device)


def heads(
    models: dict[str, Path], devices: list[str]
) -> dict[str, dict[str, FinalHead | GgufHead]]:
    """Per device, role → final head. A model with the teacher's norm and lm_head shares the
    teacher's copy, which leaves the comparison GPU more room; a gguf model has its own."""
    teacher = _final_head(models["teacher"], "cpu")
    shared = {
        role: role == "teacher" or (not any(d.glob("*.gguf")) and teacher.same(FinalHead(d, "cpu")))
        for role, d in models.items()
    }
    out: dict[str, dict[str, FinalHead | GgufHead]] = {}
    for dev in devices:
        on_dev = _final_head(models["teacher"], dev)
        out[dev] = {role: on_dev if shared[role] else head(d, dev) for role, d in models.items()}
        out[dev]["teacher_rerun"] = on_dev
    return out


@functools.lru_cache(maxsize=16)
def _final_head(model_dir: Path, device: str) -> FinalHead:
    """The teacher's head, the same for every eval: loaded once per device."""
    return FinalHead(model_dir, device)


def load_states(path: str, read: list[int], first: int) -> torch.Tensor:
    """[rows from `first`, layers, hidden] a server wrote for one request, checked against the
    tokens it read. vLLM writes every row; llama-states only those from its `first_row`."""
    with open(path + ".lock") as lock:
        # the server holds this lock exclusively until the file is complete
        fcntl.flock(lock, fcntl.LOCK_SH)
        data = load_file(path)
    states = data["hidden_states"]
    offset = int(data["first_row"][0]) if "first_row" in data else 0
    if (
        data["token_ids"].tolist() != read
        or states.shape[0] != len(read) - offset
        or not 0 <= offset <= first
        or states.shape[1] != len(config.LAYERS)
    ):
        raise ValueError(
            f"{path}: states {tuple(states.shape)} from row {offset} do not match {len(read)}"
            f" tokens from row {first} x {len(config.LAYERS)} layers"
        )
    return states[first - offset :]


@dataclass
class Sums:
    """One model's sums over one sample's answer tokens: every answer token has a prediction."""

    predictions: int
    kl: float = 0.0
    top1: float = 0.0

    def summary(self) -> dict[str, Any]:
        """Per-token means, as one sample's record."""
        p = max(self.predictions, 1)
        return {"kl": _g7(self.kl / p), "top1_agreement": _g7(self.top1 / p)}


@torch.no_grad()
def compare(
    states: dict[str, torch.Tensor],
    heads: dict[str, FinalHead | GgufHead],
    answer: list[int],
    device: str,
) -> dict[str, Sums]:
    """Sums for every role in `states` ([rows, layers, hidden] from the last prompt row, whose state
    predicts the first answer token, as load_states returns them), each against
    `states["teacher"]`, over the rows that predict answer tokens: the KL of the next-token
    distributions from the last layer's state through each model's own head."""
    out = {role: Sums(len(answer)) for role in states}
    teacher = states["teacher"]
    for c in range(0, len(answer), CHUNK):
        rows = slice(c, c + CHUNK)
        ref = heads["teacher"].logprobs(teacher[rows, -1].to(device))
        ref_p, ref_top = ref.exp(), ref.argmax(-1)
        for role, s in out.items():
            own = states[role][rows, -1].to(device)
            lp = ref if role == "teacher" else heads[role].logprobs(own)
            s.kl += (ref_p * (ref - lp)).sum().item()
            s.top1 += (lp.argmax(-1) == ref_top).sum().item()
    return out


def _g7(x: float) -> float:
    return float(f"{x:.7g}")
