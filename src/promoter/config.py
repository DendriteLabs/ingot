LEASE_S = 3600
MAX_ATTEMPTS = 3
IDLE_S = 5.0
KEEP_KINGS = 1  # downloaded kings kept on disk per pair, newest first
# DFlash 2 drafter per family, linked from gguf king cards: (repo, recommended file)
DRAFTERS = {"qwen38_27b": ("dendriteholdings/Ingot-27B-DFlash2", "Ingot-27B-DFlash2-Q8_0.gguf")}
MLX_DRAFTERS = {"qwen38_27b": "dendriteholdings/Ingot-27B-DFlash2-MLX"}
