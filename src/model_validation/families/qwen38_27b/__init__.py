"""Qwen3.8-27B: teacher Qwen/Qwen3.8-27B (bf16), genesis dendriteholdings/Ingot-genesis-27B-FP8.
The pinned files and GENESIS_WEIGHT_BYTES are regenerated with
`uv run scripts/pin_family.py qwen38_27b`.

Every genesis config value is frozen except FREE_KEYS: the miner's own quantization choices
and fields with no effect on inference. Tensors under FROZEN_DTYPE_PREFIXES keep their genesis
dtype: the parts the text eval never exercises, so they cannot be shrunk to free bytes, and
lm_head, which the eval applies itself in BF16 to the last hidden state."""

GENESIS_WEIGHT_BYTES = 30_866_866_928

FREE_KEYS = (
    "quantization_config.weight_block_size",
    "quantization_config.modules_to_not_convert",
    "transformers_version",
    "text_config.use_cache",
    "text_config.initializer_range",
    "text_config.attention_dropout",
    "vision_config.initializer_range",
)

FROZEN_DTYPE_PREFIXES = ("model.visual.", "mtp.", "lm_head.")

# vLLM's packed_modules_mapping for Qwen3.5: each group is one layer there, so one precision
FUSED = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z"),
    ("in_proj_b", "in_proj_a"),
)
