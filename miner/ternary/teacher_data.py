"""Teacher answers to train on, as token ids: draws from the eval's corpus (the eval's own draw
and cuts) or every row of a JSONL of {id, domain, messages, tools}, answered by the teacher through
the eval's generation (vLLM, the domain's answer budget). Runs on a box set up like the eval box
(scripts/setup_eval_box.sh).

    .venv/bin/python -m miner.ternary.teacher_data --draw s1 s2 s3 --out teacher-data.jsonl
    .venv/bin/python -m miner.ternary.teacher_data --draw holdout --write-draw holdout.jsonl"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import models
from common.config import chain_spec
from eval_service import dataset, run, servers


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data", type=Path, nargs="*", default=[], help="JSONL draws in the duel format"
    )
    p.add_argument(
        "--draw",
        nargs="*",
        default=[],
        metavar="SEED",
        help="draw from the eval's corpus (chain.toml) with these seeds: one eval's samples each",
    )
    p.add_argument(
        "--max-prompt", type=int, default=12288, help="skip samples whose rendered prompt is longer"
    )
    p.add_argument("--workers", type=int, default=256)
    p.add_argument("--write-draw", type=Path, help="write the drawn samples here as JSONL and stop")
    p.add_argument("--out", type=Path, help="the answers, as JSONL of token ids")
    a = p.parse_args()
    spec = chain_spec()
    teacher = models.snapshot(spec.teacher_repo, spec.teacher_revision)
    tok = run.load_tokenizer(teacher)
    samples, seen = [], set()
    if a.draw:
        corpus = dataset.fetch(spec, Path("eval.jsonl"))
        for seed in a.draw:
            for s in corpus.draw(seed, 0, run.fits(tok)):
                if s.id not in seen:
                    seen.add(s.id)
                    samples.append(s)
    for f in a.data:
        for s in dataset.load(f).samples:
            if s.id not in seen:
                seen.add(s.id)
                samples.append(s)
    if a.write_draw:
        rows = [
            {
                "id": s.id,
                "domain": s.domain,
                "source": s.source,
                "language": s.language,
                "messages": s.messages,
                "tools": s.tools,
            }
            for s in samples
        ]
        a.write_draw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{len(rows)} samples -> {a.write_draw}")
        return
    if a.out is None:
        p.error("--out is required unless --write-draw is given")
    prompts = [run._render(tok, s.messages, s.tools) for s in samples]
    keep = [i for i, pr in enumerate(prompts) if len(pr) <= a.max_prompt]
    print(f"{len(samples)} samples, {len(keep)} with prompts <= {a.max_prompt} tokens", flush=True)
    done = set()
    if a.out.exists():
        done = {json.loads(line)["id"] for line in a.out.read_text().splitlines()}
        keep = [i for i in keep if samples[i].id not in done]
        print(f"{len(done)} already answered, {len(keep)} to go", flush=True)
    groups = servers.serve({"teacher": teacher})
    started = time.perf_counter()
    out = a.out.open("a")
    count = {"n": 0, "tokens": 0}

    def answer(i):
        s = samples[i]
        budget = run.answer_budget(s)
        tokens, path = run.generate(groups["teacher"], prompts[i], i, s.id, budget, budget)
        servers.discard(path)
        return i, tokens

    try:
        with ThreadPoolExecutor(a.workers) as pool:
            for i, tokens in pool.map(answer, keep):
                s = samples[i]
                out.write(
                    json.dumps(
                        {"id": s.id, "domain": s.domain, "prompt": prompts[i], "answer": tokens}
                    )
                    + "\n"
                )
                out.flush()
                count["n"] += 1
                count["tokens"] += len(tokens)
                if count["n"] % 50 == 0:
                    print(
                        f"{count['n']} answered, {count['tokens'] / 1e6:.2f} M answer tokens, "
                        f"{time.perf_counter() - started:.0f}s",
                        flush=True,
                    )
    finally:
        servers.stop_all()
    print(
        f"DATA DONE: {count['n']} answers, {count['tokens'] / 1e6:.2f} M answer tokens in "
        f"{time.perf_counter() - started:.0f}s -> {a.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
