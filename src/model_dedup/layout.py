"""Which Qwen3.8 tensors are fingerprinted, and on which axis each meets the residual stream."""

from __future__ import annotations

PREFIX = "model.language_model."
EMBED = "embed_tokens.weight"
HEAD = "lm_head.weight"
# Linears that read the residual stream (it is their columns) and that write it (their rows).
READERS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "gate_proj",
    "up_proj",
)
WRITERS = ("o_proj", "out_proj", "down_proj")
# Norms are 1-D and conv1d is 3-D: too little to fingerprint.
MIN_DIM = 8


def canonical(key: str) -> str | None:
    """Checkpoint key → the name the dedup uses, or None for tensors it ignores (vision, MTP)."""
    if key == HEAD:
        return key
    if key.startswith(PREFIX) and key.endswith(".weight"):
        return key[len(PREFIX) :]
    return None


def role(name: str) -> str:
    if name == EMBED:
        return "embed"
    if name == HEAD:
        return "head"
    proj = name.rsplit(".", 2)[-2]
    if proj in READERS:
        return "reader"
    if proj in WRITERS:
        return "writer"
    return "other"


def norm_for(name: str) -> str | None:
    """The RMSNorm whose output this reader consumes."""
    if name == HEAD:
        return "norm.weight"
    if role(name) != "reader":
        return None
    layer = ".".join(name.split(".", 2)[:2])
    return f"{layer}.{'post_attention_layernorm' if '.mlp.' in name else 'input_layernorm'}.weight"
