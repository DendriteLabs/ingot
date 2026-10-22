"""Prove the GPU box can run an eval and see what it measures and costs: the teacher, a king and a
challenger on a few samples, through the same scoring as the eval service.

    bash scripts/setup_eval_box.sh
    .venv/bin/python scripts/eval_smoke.py [--samples N] [--dataset eval.jsonl] [--seed S]
        [--teacher DIR] [--king DIR] [--challenger DIR]

Models default to the teacher and genesis from chain.toml, genesis playing both king and
challenger, so the two must land equally close to the teacher. Prints KL and the hidden-state
distance per model, the noise floor and the wall time; writes one record per sample.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import models
from common.config import chain_spec
from eval_service import dataset, run, servers
from eval_service.dataset import Sample

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is 17 * 23? Show your steps.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=len(PROMPTS))
    parser.add_argument("--dataset", type=Path, help="JSONL of {id, messages}; default: built in")
    parser.add_argument("--seed", default="smoke", help="seed of the draw from --dataset")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--king", type=Path)
    parser.add_argument("--challenger", type=Path)
    args = parser.parse_args()

    spec = chain_spec()
    fp8 = spec.pair(spec.family, "fp8")  # vLLM serves the safetensors genesis
    teacher = args.teacher or models.snapshot(spec.teacher_repo, spec.teacher_revision)
    king, challenger = args.king, args.challenger
    if king is None or challenger is None:
        genesis = models.snapshot(fp8.genesis_repo, fp8.genesis_revision)
        king, challenger = king or genesis, challenger or genesis
    if args.dataset:
        samples = dataset.draw(dataset.load(args.dataset).samples, args.seed, args.samples)
    else:
        samples = [
            Sample(f"s{i}", [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}])
            for i in range(args.samples)
        ]

    started = time.perf_counter()
    try:
        duel = run.duel(teacher, challenger, king, samples)
    finally:
        servers.stop_all()
    for name, d in (("challenger", duel.challenger), ("king", duel.king), ("noise", duel.noise)):
        print(f"{name:10s} KL {d.kl:.5f}  distance {d.dist:.4f}")
    print(
        f"won: {duel.won}; {len(samples)} samples in {time.perf_counter() - started:.0f}s"
        " including server start-up"
    )
    out = Path("eval_smoke.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in duel.samples))
    print(f"per-sample records in {out}")


if __name__ == "__main__":
    main()
