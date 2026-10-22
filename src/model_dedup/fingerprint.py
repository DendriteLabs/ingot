"""A fingerprint: secret sketches of a model's quantization error after canonicalization.

Per tensor, the error D = canonical(model) − teacher is kept as psi @ D @ omega (64 x 64). psi and
omega have N(0, 1/64) entries seeded by an HMAC of the validator's secret, so a miner cannot
shape an error the sketch misses; they preserve energy and inner products in expectation, so two
errors compare through their sketches alone. They are drawn on the CPU: no GPU dependence.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.config import Settings
from model_dedup.canon import Basis, canonicalize, fold_gain, residual_transform
from model_dedup.checkpoint import Checkpoint
from model_dedup.gguf_checkpoint import GgufCheckpoint
from model_dedup.layout import EMBED

# Part of every fingerprint: a change makes the bank incomparable and it has to be reseeded.
VERSION = "qd-v1"
K = 64


def load_secret(s: Settings) -> bytes:
    """INGOT_DEDUP_SECRET; exits when it is under 32 bytes."""
    secret = s.dedup_secret.encode()
    if len(secret) < 32:
        raise SystemExit("INGOT_DEDUP_SECRET must be at least 32 bytes")
    return secret


def key_id(secret: bytes) -> str:
    return hashlib.sha256(secret).hexdigest()[:8]


def fingerprint_dir(model_dir: Path, *, teacher_dir: Path, secret: bytes, device: str) -> dict:
    """Against the safetensors teacher; a gguf model is read in the teacher's layout
    (GgufCheckpoint)."""
    with torch.no_grad():
        return fingerprint(
            open_checkpoint(model_dir, device), open_checkpoint(teacher_dir, device), secret
        )


def open_checkpoint(path: Path, device: str) -> Checkpoint | GgufCheckpoint:
    return GgufCheckpoint(path, device) if any(path.glob("*.gguf")) else Checkpoint(path, device)


def fingerprint(
    model: Checkpoint | GgufCheckpoint, teacher: Checkpoint | GgufCheckpoint, secret: bytes
) -> dict:
    t0 = time.time()
    basis = residual_transform(model.get(EMBED), teacher.get(EMBED))
    entries = [
        _sketch_tensor(
            name,
            fold_gain(name, model.get(name), model),
            fold_gain(name, teacher.get(name), teacher),
            basis,
            secret,
        )
        for name in teacher.names()
        if name in set(model.names())
    ]
    tnorm2 = sum(e["tnorm"] ** 2 for e in entries)
    dnorm2 = sum(e["dnorm"] ** 2 for e in entries)
    idents = [e["ident"] for e in entries if "ident" in e]
    before = sum(e["rs_before"] for e in entries)
    after = sum(e["rs_after"] for e in entries)
    return dict(
        version=VERSION,
        key_id=key_id(secret),
        tensors_hash=model.tensors_hash(),
        rel_to_teacher=math.sqrt(dnorm2 / tnorm2),
        identity_frac=float(np.mean(idents)) if idents else 1.0,
        rescale_frac=1 - after / before if before > 0 else 0.0,
        residual=_residual_info(basis),
        secs=round(time.time() - t0, 1),
        tensors=entries,
    )


def projections(name: str, shape: tuple[int, ...], secret: bytes) -> tuple:
    """psi (k x m) and omega (n x k): the same for every model under one secret."""
    m, n = shape
    k = min(K, m, n)
    psi = torch.randn(k, m, generator=_generator(secret, name, "psi")) / math.sqrt(k)
    omega = torch.randn(n, k, generator=_generator(secret, name, "omega")) / math.sqrt(k)
    return psi, omega


def _sketch_tensor(
    name: str, w: torch.Tensor, ref: torch.Tensor, basis: Basis, secret: bytes
) -> dict[str, Any]:
    w, info = canonicalize(name, w, ref, basis)
    d = w - ref
    psi, omega = projections(name, tuple(d.shape), secret)
    s = (psi.to(d.device) @ (d @ omega.to(d.device))).cpu().numpy().astype(np.float32)
    return dict(
        name=name,
        k=s.shape[0],
        s=base64.b64encode(s.tobytes()).decode(),
        dnorm=float(d.norm()),
        tnorm=float(ref.norm()),
        **info,
    )


def _residual_info(basis: Basis) -> dict[str, Any]:
    if basis is None:
        return dict(identity=True, scale=1.0, off_identity=0.0, perm_like=1.0)
    r, c = basis
    eye = torch.eye(r.shape[0], device=r.device)
    return dict(
        identity=False,
        scale=c,
        off_identity=float((r - eye).norm() / math.sqrt(r.shape[0])),
        perm_like=float(r.abs().max(dim=1).values.mean()),  # 1.0 for a signed permutation
    )


def _generator(secret: bytes, name: str, which: str) -> torch.Generator:
    digest = hmac.new(secret, f"{VERSION}:{which}:{name}".encode(), hashlib.sha256).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
