"""Put a submission's weights into the teacher's coordinates.

Every function-preserving symmetry of Qwen3.8 is either global on the residual stream (a scale,
a signed permutation or a rotation once the norm gains are folded into the readers), undone once
from the embeddings, or local to one tensor's hidden axis (permutations and diagonal scalings of
neurons, heads, channels and norm gains), undone per tensor by matching and a rescale fit.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from model_dedup.layout import norm_for, role

RESCALE_ITERS = 6
Basis = tuple[torch.Tensor, float] | None


def residual_transform(embed: torch.Tensor, ref_embed: torch.Tensor) -> Basis:
    """(R, c) with embed @ R ≈ c * ref_embed, R orthogonal: Procrustes on the embedding matrices.

    Submissions keep the embeddings in BF16 and 248k rows pin a 5120-dim rotation down exactly.
    Fitting it on hidden states instead overfits and hides real quantization error.
    """
    if embed.shape == ref_embed.shape and torch.equal(embed, ref_embed):
        return None  # keeps an untouched basis bit-exact
    # float64: summed over 248k rows, float32 left R ~1e-3 off a signed permutation
    cross = torch.zeros(embed.shape[1], embed.shape[1], device=embed.device, dtype=torch.float64)
    for i in range(0, embed.shape[0], 16384):
        cross += embed[i : i + 16384].double().T @ ref_embed[i : i + 16384].double()
    u, s, vh = torch.linalg.svd(cross)
    return (u @ vh).float(), float(s.sum() / ref_embed.double().pow(2).sum())


def fold_gain(name: str, w: torch.Tensor, ckpt: Any) -> torch.Tensor:
    """Multiply a reader by the zero-centred gain (1 + w) of the norm that feeds it.

    Done on both sides before the basis is undone: rotate the residual stream, then re-pick the
    norm gains s with the readers scaled by 1/s, and the readers carry Q diag(1/s) Q^T, a dense
    matrix no row/column fit removes. With each model's own gains folded in it is a rotation.
    """
    norm = norm_for(name)
    if norm is None:
        return w
    return w * (1 + ckpt.get(norm).float().to(w.device))[None, :]


def undo_residual(name: str, w: torch.Tensor, basis: Basis) -> torch.Tensor:
    if basis is None:
        return w
    r, c = basis
    kind = role(name)
    if kind in ("reader", "head"):
        return w @ r  # RMSNorm removed the scale from what the readers see
    if kind == "writer":
        return r.T @ w / c
    if kind == "embed":
        return w @ r / c
    return w


def match(cand: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, float]:
    """One-to-one order of cand's rows lining them up with ref's, by |cosine|.

    Absolute: sign flips are symmetries too (up_proj against down_proj, V against O); the
    rescale fit recovers the sign. Ties keep a row in place, so an unpermuted tensor is untouched.
    """
    cs = (F.normalize(ref, dim=1) @ F.normalize(cand, dim=1).T).abs()
    best, idx = cs.max(dim=1)
    own = torch.arange(idx.numel(), device=idx.device)
    idx = torch.where(cs.diagonal() >= best, own, idx)
    if idx.unique().numel() != idx.numel():
        used = torch.zeros(cand.shape[0], dtype=torch.bool, device=cand.device)
        for j in torch.argsort(best, descending=True).tolist():
            i = int(idx[j])
            if used[i]:
                row = cs[j].clone()
                row[used] = -1
                i = int(row.argmax())
                idx[j] = i
            used[i] = True
    return idx, float((idx == own).float().mean())


def rescale(w: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Divide out the row and column scaling that best maps w onto ref, and the squared distance
    to ref before and after. Fits m + n numbers out of m * n, so a real error keeps its energy."""
    before = float((w - ref).pow(2).sum())
    if before == 0:
        return w, 0.0, 0.0
    for _ in range(RESCALE_ITERS):
        w = w * _axis_scale(w, ref, 0)
        w = w * _axis_scale(w, ref, 1)
    return w, before, float((w - ref).pow(2).sum())


def canonicalize(
    name: str, w: torch.Tensor, ref: torch.Tensor, basis: Basis
) -> tuple[torch.Tensor, dict[str, float]]:
    w = undo_residual(name, w, basis)
    info: dict[str, float] = {}
    kind = role(name)
    if kind == "reader":
        idx, info["ident"] = match(w, ref)
        w = w[idx]
    elif kind == "writer":
        idx, info["ident"] = match(w.T, ref.T)
        w = w[:, idx]
    w, info["rs_before"], info["rs_after"] = rescale(w, ref)
    return w, info


def _axis_scale(w: torch.Tensor, ref: torch.Tensor, dim: int) -> torch.Tensor:
    num, den = (w * ref).sum(dim), (w * w).sum(dim)
    return torch.where(den > 0, num / den, torch.ones_like(den)).unsqueeze(dim)
