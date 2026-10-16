"""Is a fingerprint another banked model's? Every honest submission is the teacher plus its own
quantization error, so the question is whether the error is someone else's.

Independent quantizations make nearly uncorrelated errors. A copy keeps its source's error
through any function-preserving disguise; noise of k times that error only dilutes it to
corr = 1 / sqrt(1 + k^2), while the copy's own error grows by the same sqrt(1 + k^2).
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import numpy as np

from model_dedup import config


@dataclass(frozen=True)
class Verdict:
    reason: str | None  # None, COPY (identical tensors) or NEAR-COPY (the same error)
    of: str | None  # bank id of the copied model
    message: str
    corr: float  # best error correlation in the shortlist
    copied_frac: float  # best share of the error copied tensor by tensor
    notes: tuple[str, ...]

    @property
    def duplicate(self) -> bool:
        return self.reason is not None


def sketches(fp: dict) -> dict[str, np.ndarray]:
    return {
        e["name"]: np.frombuffer(base64.b64decode(e["s"]), dtype=np.float32).reshape(e["k"], e["k"])
        for e in fp["tensors"]
    }


def similarity(a: dict, b: dict) -> dict[str, float]:
    """corr: correlation of the two errors over the whole model. ratio: |D_a − D_b| over the
    smaller |D|, about k for a copy with noise k. copied_frac: energy share of the tensors whose
    own correlation is >= PART_CORR, which a pooled corr dilutes when only some are copied."""
    sa, sb = sketches(a), sketches(b)
    dot = na = nb = dist = copied = energy = 0.0
    for name in (n for n in sa if n in sb):
        x, y = sa[name].astype(np.float64), sb[name].astype(np.float64)
        d, ex, ey = float((x * y).sum()), float((x * x).sum()), float((y * y).sum())
        dot, na, nb = dot + d, na + ex, nb + ey
        dist += float(np.square(x - y).sum())
        # a tensor neither model changed carries no evidence
        if ex > 0 and ey > 0:
            w = math.sqrt(ex * ey)
            energy += w
            copied += w if d / w >= config.PART_CORR else 0.0
    if na == 0 or nb == 0:
        return dict(corr=0.0, ratio=math.inf, copied_frac=0.0)
    return dict(
        corr=dot / math.sqrt(na * nb),
        ratio=math.sqrt(dist / min(na, nb)),
        copied_frac=copied / energy if energy else 0.0,
    )


def decide(fp: dict, shortlist: dict[str, dict]) -> Verdict:
    """COPY (identical tensors) > NEAR-COPY (the same error, whole or in part) > unique.

    Closeness to the teacher is no evidence: a near-teacher model is a good one, and the size
    limit already stops the teacher itself.
    """
    for bank_id, other in shortlist.items():
        if (other["version"], other["key_id"]) != (fp["version"], fp["key_id"]):
            raise ValueError(f"{bank_id}: fingerprint made under another secret or version")
    notes = _notes(fp)
    sims = {bank_id: similarity(fp, other) for bank_id, other in shortlist.items()}
    corr = max((s["corr"] for s in sims.values()), default=0.0)
    frac = max((s["copied_frac"] for s in sims.values()), default=0.0)
    for bank_id, other in shortlist.items():
        if other["tensors_hash"] == fp["tensors_hash"]:
            return Verdict("COPY", bank_id, f"identical tensors to {bank_id}", corr, frac, notes)
    if sims:
        best = max(sims, key=lambda b: sims[b]["corr"])
        if sims[best]["corr"] >= config.COPY_CORR:
            msg = (
                f"its quantization error correlates {sims[best]['corr']:.3f} with {best}'s"
                f" (relative distance {sims[best]['ratio']:.2f})"
            )
            return Verdict("NEAR-COPY", best, msg, corr, frac, notes)
        part = max(sims, key=lambda b: sims[b]["copied_frac"])
        if sims[part]["copied_frac"] >= config.COPIED_FRAC:
            msg = (
                f"{sims[part]['copied_frac']:.0%} of its quantization error is {part}'s,"
                f" tensor by tensor (overall correlation {sims[part]['corr']:.3f})"
            )
            return Verdict("NEAR-COPY", part, msg, corr, frac, notes)
    return Verdict(None, None, "unique quantization error", corr, frac, notes)


def _notes(fp: dict) -> tuple[str, ...]:
    """The disguises canonicalization undid, for the record."""
    out = []
    res = fp["residual"]
    if not res["identity"] and (res["off_identity"] > 1e-2 or abs(res["scale"] - 1) > 1e-3):
        kind = "signed permutation" if res["perm_like"] > 0.99 else "rotation"
        out.append(f"residual basis: {kind}, scale {res['scale']:.4f}")
    if fp["identity_frac"] < 0.99:
        out.append(f"permuted hidden axes, identity fraction {fp['identity_frac']:.3f}")
    if fp["rescale_frac"] > 0.5:
        out.append(f"{fp['rescale_frac']:.0%} of the difference was row/column scaling")
    return tuple(out)
