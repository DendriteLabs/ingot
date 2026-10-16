"""A model directory as the dedup reads it: language-model weights, dequantized to float32."""

from __future__ import annotations

import hashlib
import mmap
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from common.models import safetensors_header
from model_dedup.layout import MIN_DIM, canonical

FP8 = "F8_E4M3"
SCALE = "_scale_inv"


class Checkpoint:
    def __init__(self, path: Path, device: str) -> None:
        self.path = Path(path)
        self.device = torch.device(device)
        self._file: dict[str, str] = {}
        self._meta: dict[str, tuple[str, tuple[int, ...]]] = {}
        for shard in sorted(self.path.glob("*.safetensors")):
            for key, info in safetensors_header(shard).items():
                self._file[key] = shard.name
                self._meta[key] = (info["dtype"], tuple(info["shape"]))
        self._key = {c: k for k in self._file if (c := canonical(k))}
        self._handles: dict[str, Any] = {}

    def names(self) -> list[str]:
        """The fingerprinted tensors: 2-D, both sides at least MIN_DIM."""
        return sorted(
            name
            for name, key in self._key.items()
            if len(self._meta[key][1]) == 2 and min(self._meta[key][1]) >= MIN_DIM
        )

    def get(self, name: str) -> torch.Tensor:
        key = self._key[name]
        w = self._raw(key).float()
        if self._meta[key][0] == FP8:
            w = w * _per_element(self._raw(key + SCALE).float(), w.shape)
        return w

    def tensors_hash(self) -> str:
        """sha256 over every tensor's name, dtype, shape and bytes; shard names and order do not
        count, so a re-sharded copy hashes the same."""
        digests = []
        for shard in sorted(self.path.glob("*.safetensors")):
            header = safetensors_header(shard)
            with shard.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                start = 8 + int.from_bytes(mm[:8], "little")
                for key, info in header.items():
                    a, b = info["data_offsets"]
                    h = hashlib.sha256(f"{key}|{info['dtype']}|{info['shape']}|".encode())
                    h.update(mm[start + a : start + b])
                    digests.append(h.hexdigest())
        return hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest()

    def _raw(self, key: str) -> torch.Tensor:
        fname = self._file[key]
        if fname not in self._handles:
            self._handles[fname] = safe_open(self.path / fname, "pt", device=str(self.device))
        return self._handles[fname].get_tensor(key)


def _per_element(scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """One scale per weight block, spread over the block's elements (the edge blocks may be cut)."""
    rows, cols = shape
    scale = scale.repeat_interleave(-(-rows // scale.shape[0]), 0)[:rows]
    return scale.repeat_interleave(-(-cols // scale.shape[1]), 1)[:, :cols]
