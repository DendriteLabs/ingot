"""Decode speed of an Ingot q2 king, measured on king IV: the gguf files on one RTX PRO 6000
Blackwell and on a Mac Studio M3 Ultra, the MLX pack on the same Mac. Every q2
king has the same layout and sizes, so the figures hold for all of them."""

from __future__ import annotations

FORK = "https://github.com/DendriteLabs/llama.cpp"
GPU = "NVIDIA RTX PRO 6000 Blackwell (server edition, 96 GB)"
MAC = "Mac Studio M3 Ultra (80-core GPU, 512 GB)"

# (decode tok/s, prompt tok/s) of the PQ2_0 file on GPU
GPU_PQ2 = ("137", "3,865")
# the PTQ1_0 file on GPU: decode measured 6% above PQ2_0 in a paired run, prompt speed the same
GPU_PTQ1 = ("145", "3,865")
OTHER_GPUS = (
    "Other GPUs: PrismML reports 142 tok/s for Bonsai 2 PQ2_0 on an RTX 5090; we have not "
    "measured other cards."
)

# build label → {packing: (decode tok/s, prompt tok/s)} on the Mac
MAC_ROWS = (
    ("PrismML llama.cpp fork, stock", {"PQ2_0": ("44.5", "432"), "PTQ1_0": ("43.1", "368")}),
    (
        f"[Dendrite fork]({FORK}), Metal patch",
        {"PQ2_0": ("56.4", "432"), "PTQ1_0": ("52.8", "368")},
    ),
)

# (engine, mode, decode tok/s) of the MLX pack on the Mac
MLX_ROWS = (
    ("MTPLX 2.12 (mlx 0.32) server", "plain", "55.3"),
    ("MTPLX 2.12 (mlx 0.32) server", "speculative", "66.6"),
    ("mlx-lm 0.31 graph with MTPLX's ternary kernels, in-process", "plain", "59.9"),
)
MLX_VS_GGUF = (MAC_ROWS[1][1]["PQ2_0"][0], GPU_PQ2[0])


def gguf_section(pq2: str, ptq1: str | None, mlx_repo: str | None) -> str:
    """The gguf card's Speed section; PTQ1_0 rows only with a PTQ1_0 file, the pointer to the
    MLX card only once that repo exists."""
    files = [("PQ2_0", pq2)] + ([("PTQ1_0", ptq1)] if ptq1 else [])
    text = (
        "\n## Speed\n\n"
        "Decode tokens per second at batch 1, measured on Ingot king IV with the Dendrite "
        "fork at the commit above. Every q2 king has the same file sizes and layout, so these "
        "numbers hold for all of them within run-to-run noise.\n\n"
        "### GPU\n\n"
        "`llama-bench`, 128 generated tokens (`tg128`) and a 512-token prompt (`pp512`), one "
        "GPU, `-ngl 99 -fa 1`.\n\n"
        "| GPU | file | decode tok/s | prompt tok/s |\n| --- | --- | --- | --- |\n"
    )
    for packing, name in files:
        decode, prompt = GPU_PQ2 if packing == "PQ2_0" else GPU_PTQ1
        text += f"| {GPU} | `{name}` | {decode} | {prompt} |\n"
    text += (
        f"\n{OTHER_GPUS}\n\n"
        "### Mac\n\n"
        f"{MAC}, `llama-bench`, `tg128` and `pp512`, Metal. Two builds: PrismML's fork as "
        f"published, and the [Dendrite fork]({FORK}) whose Metal patch fuses the Hadamard "
        "rotation and the linear-attention state update for this model family.\n\n"
        "| build | file | decode tok/s | prompt tok/s |\n| --- | --- | --- | --- |\n"
    )
    for build, rows in MAC_ROWS:
        for packing, name in files:
            decode, prompt = rows[packing]
            text += f"| {build} | `{name}` | {decode} | {prompt} |\n"
    if mlx_repo:
        text += (
            "\nThe same weights as an MLX pack run faster on Apple Silicon: see the speed table "
            f"of [{mlx_repo}](https://huggingface.co/{mlx_repo}).\n"
        )
    return text


def mlx_section(gguf_repo: str) -> str:
    """The MLX card's Speed section."""
    mac, gpu = MLX_VS_GGUF
    text = (
        "\n## Speed\n\n"
        f"Decode tokens per second at batch 1 on a {MAC}, measured on Ingot king IV. "
        "Every q2 king has the same layout and size, so the numbers hold for all of them within "
        'run-to-run noise. "Plain" is one token per forward pass; "speculative" uses MTPLX\'s '
        "multi-token draft head (`--generation-mode mtp --depth 1`). The server numbers are 256 "
        "generated tokens on a code prompt, timed over HTTP; the in-process numbers are 128 "
        "tokens with no server.\n\n"
        "| engine | mode | decode tok/s |\n| --- | --- | --- |\n"
    )
    for engine, mode, decode in MLX_ROWS:
        text += f"| {engine} | {mode} | {decode} |\n"
    text += (
        f"\nFor comparison, the gguf twin of these weights decodes at {mac} tok/s on the same Mac "
        f"in the [Dendrite llama.cpp fork]({FORK}) and at {gpu} tok/s on an RTX PRO 6000; see "
        f"[its card](https://huggingface.co/{gguf_repo}).\n"
    )
    return text
