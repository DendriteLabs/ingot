"""Does the PyTorch model compute what the fork computes? The fork reads n prompts forced through
llama-states (the eval's own path), the gguf head turns the last layer's states into log-probs; the
PyTorch model reads the same tokens; per block and per position, cosine, KL both ways and top-1
agreement. Runs on a box set up like the eval box.

    .venv/bin/python -m miner.ternary.numerics_check --gguf-dir <dir with one gguf> \
        --prompts draw.jsonl --n 8 --max-tokens 300 --gpus 1 --first-gpu 7 --layers 1,4,64 \
        --out numerics.json"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from common import models
from common.config import chain_spec
from eval_service import metrics, run, servers
from miner.ternary.model import layer_device_map, load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf-dir", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument(
        "--head", type=int, default=8, help="tokens treated as the prompt; the rest is read forced"
    )
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument(
        "--gpus", type=int, default=6, help="GPUs for the PyTorch model (the teacher's, idle here)"
    )
    p.add_argument("--first-gpu", type=int, default=0)
    p.add_argument(
        "--layers",
        default="64",
        help="blocks whose outputs to compare, 1-based as config.LAYERS (64 = last)",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from eval_service import config

    setattr(config, "LAYERS", tuple(int(x) for x in a.layers.split(",")))  # noqa: B010
    spec = chain_spec()
    teacher = models.snapshot(spec.teacher_repo, spec.teacher_revision)
    tok = run.load_tokenizer(teacher)
    rows = [json.loads(line) for line in a.prompts.read_text().splitlines()][: a.n]
    seqs = []
    for r in rows:
        ids = run._render(tok, r["messages"], r.get("tools"))[: a.head + a.max_tokens]
        seqs.append((ids[: a.head], ids[a.head :]))
    gguf = next(a.gguf_dir.glob("*.gguf"))
    print(
        f"{len(seqs)} sequences, {sum(len(x) for _, x in seqs)} forced tokens; {gguf.name}",
        flush=True,
    )

    # phase 1: the fork
    t0 = time.perf_counter()
    groups = servers.serve({"king": a.gguf_dir})
    device = f"cuda:{a.first_gpu + a.gpus - 1}"
    head = metrics.heads({"teacher": teacher, "model": a.gguf_dir}, [device])[device]["model"]
    fork = []
    try:
        for i, (prompt, ans) in enumerate(seqs):
            read, path = groups["king"].complete(prompt, len(ans), i, prompt + ans)
            if read != ans:
                raise RuntimeError(f"sequence {i}: the fork left the forced answer")
            states = metrics.load_states(path, prompt + ans[:-1], len(prompt) - 1)
            servers.discard(path)
            fork.append(
                (
                    head.logprobs(states[:, -1].to(device)).to(torch.float16).cpu(),
                    states.float().cpu(),
                )
            )
            print(f"fork read {i}: {len(ans)} rows", flush=True)
    finally:
        servers.stop_all()
    print(f"fork phase {time.perf_counter() - t0:.0f}s", flush=True)
    del head
    torch.cuda.empty_cache()

    # phase 2: PyTorch
    from accelerate import dispatch_model

    t0 = time.perf_counter()
    model = load_model(gguf, teacher)
    model = dispatch_model(
        model, device_map={k: v + a.first_gpu for k, v in layer_device_map(model, a.gpus).items()}
    )
    model.eval()
    print(f"model on {a.gpus} GPUs in {time.perf_counter() - t0:.0f}s", flush=True)
    report: dict[str, Any] = {"gguf": str(gguf), "sequences": [], "layers": {}}
    tot_kl = tot_rkl = tot_agree = tot_n = 0.0
    captured = {}
    for li, layer in enumerate(model.model.layers):
        if li + 1 in config.LAYERS:
            layer.register_forward_hook(
                lambda m, inp, out, li=li: captured.__setitem__(
                    li + 1, (out[0] if isinstance(out, tuple) else out)
                )
            )
    with torch.no_grad():
        for i, (prompt, ans) in enumerate(seqs):
            ids = torch.tensor([prompt + ans], device=f"cuda:{a.first_gpu}")
            captured.clear()
            logits = (
                model(input_ids=ids, use_cache=False)
                .logits[0, len(prompt) - 1 : len(prompt) + len(ans) - 1]
                .float()
            )
            for k, lyr in enumerate(config.LAYERS):
                theirs_h = fork[i][1][:, k]
                ours_h = (
                    captured[lyr][0, len(prompt) - 1 : len(prompt) + len(ans) - 1].float().cpu()
                )
                cos = torch.nn.functional.cosine_similarity(ours_h, theirs_h, dim=1)
                rel = (ours_h - theirs_h).norm(dim=1) / theirs_h.norm(dim=1).clamp_min(1e-6)
                report["layers"].setdefault(str(lyr), []).append(
                    {
                        "cos_mean": cos.mean().item(),
                        "cos_min": cos.min().item(),
                        "rel_err_mean": rel.mean().item(),
                    }
                )
                print(
                    f"seq {i} block {lyr}: cos mean {cos.mean().item():.4f} "
                    f"min {cos.min().item():.4f}  "
                    f"rel err {rel.mean().item():.4f}",
                    flush=True,
                )
            ours = torch.log_softmax(logits, -1)
            theirs = fork[i][0].to(ours.device).float()
            kl = (theirs.exp() * (theirs - ours)).sum(-1)  # KL(fork || torch) per position
            rkl = (ours.exp() * (ours - theirs)).sum(-1)
            agree = (theirs.argmax(-1) == ours.argmax(-1)).float()
            p_actual = ours.gather(1, torch.tensor(ans, device=ours.device)[:, None])[:, 0].exp()
            p_actual_fork = theirs.gather(1, torch.tensor(ans, device=ours.device)[:, None])[
                :, 0
            ].exp()
            s = {
                "i": i,
                "tokens": len(ans),
                "kl_mean": kl.mean().item(),
                "kl_max": kl.max().item(),
                "rkl_mean": rkl.mean().item(),
                "top1_agree": agree.mean().item(),
                "p_actual_torch": p_actual.mean().item(),
                "p_actual_fork": p_actual_fork.mean().item(),
                "worst_positions": [
                    (int(j), round(float(kl[j]), 4)) for j in kl.topk(min(3, len(kl))).indices
                ],
            }
            report["sequences"].append(s)
            tot_kl += kl.sum().item()
            tot_rkl += rkl.sum().item()
            tot_agree += agree.sum().item()
            tot_n += len(ans)
            print(
                f"seq {i}: {len(ans)} tok  KL {s['kl_mean']:.5f} (max {s['kl_max']:.3f})  "
                f"rKL {s['rkl_mean']:.5f}  top1 {s['top1_agree']:.4f}  "
                f"p(actual) torch {s['p_actual_torch']:.4f} fork {s['p_actual_fork']:.4f}",
                flush=True,
            )
    report["kl_mean"], report["rkl_mean"], report["top1_agree"] = (
        tot_kl / tot_n,
        tot_rkl / tot_n,
        tot_agree / tot_n,
    )
    a.out.write_text(json.dumps(report, indent=1))
    print(
        f"ALL: KL(fork||torch) {report['kl_mean']:.5f}  reverse {report['rkl_mean']:.5f}  "
        f"top-1 agreement {report['top1_agree']:.4f} over {int(tot_n)} tokens -> {a.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
