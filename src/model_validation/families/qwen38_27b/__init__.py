"""Qwen3.8-27B: teacher Qwen/Qwen3.8-27B (bf16), genesis dendriteholdings/Ingot-genesis-27B-FP8.
The pinned files are regenerated with `uv run scripts/pin_family.py qwen38_27b`.

Every genesis config value is frozen except FREE_KEYS: the miner's own quantization choices
and fields with no effect on inference. Tensors under FROZEN_DTYPE_PREFIXES keep their genesis
dtype: the parts the text eval never exercises, and lm_head, which the eval applies itself in
BF16 to the last hidden state. The parts under UNCOUNTED_PREFIXES are left out of the size
count on both sides: the text model may hold at most what genesis's text model holds."""

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
UNCOUNTED_PREFIXES = ("model.visual.", "mtp.")

# vLLM's FP8 loader reads modules_to_not_convert for linear layers only; these it builds as
# embeddings (ParallelLMHead is one), unquantized whatever the list says
EMBEDDINGS = ("model.language_model.embed_tokens", "model.visual.pos_embed", "lm_head")

# vLLM's packed_modules_mapping for Qwen3.5: each group is one layer there, so one precision
FUSED = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("in_proj_qkv", "in_proj_z"),
    ("in_proj_b", "in_proj_a"),
)
