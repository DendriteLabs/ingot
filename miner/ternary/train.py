"""A training loop for the codes, to improve: the student is the gguf as model.py builds it
(trainable linears keep a float latent per code and float group scales, rounding straight-through),
the loss is the eval's own measure, forward KL(teacher || student) per answer token on the
teacher's answers, the teacher a frozen BF16 copy; FSDP2 over the node's GPUs, one sequence per
rank per step. It reports the KL on held-out sequences and the number of codes that moved as it
goes, and exports codes, scales and small tensors for pack.py.

As it stands, on 1,400 teacher answers (3.5 M tokens) and 16 sequences a step, it does not improve
a king: the training KL does not fall and the held-out KL rises once codes start to flip. The
sharding, the loss and the export are done; the recipe (far more data per step, the rates, a
warm-up, or a different objective) is yours.

    .venv/bin/torchrun --nproc_per_node 8 -m miner.ternary.train --gguf king.gguf \
        --data teacher-data.jsonl --train attention --train-small --steps 200 --max-seq 6144 \
        --lr 0.02 --tight-memory --out run-attn

--train all needs --optimizer adafactor on 80 GB GPUs; --scales-only holds the codes."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint

from common.gguf import read_header
from miner.ternary.model import (
    PrismLinear,
    linear_map,
    load_model,
    set_param,
    small_map,
    small_tensors,
)

# the output head stays replicated and frozen: ranks call it a different number of times
TRAIN_SETS = {
    "all": lambda p: p != "lm_head",
    "attention": lambda p: ".self_attn." in p or ".linear_attn." in p,
    "mlp": lambda p: ".mlp." in p,
    "full-attention": lambda p: ".self_attn." in p,
}


def log(rank, *a):
    if rank == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_rows(
    path: Path, max_seq: int, seed: int, rank: int, world: int, holdout: int
) -> tuple[list[dict], list[dict]]:
    """This rank's training rows and held-out rows: the first `holdout` rows after a seeded shuffle
    (rounded down to a multiple of the ranks) are never trained on."""
    rows = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if len(r["answer"]) < 2:
            continue
        over = len(r["prompt"]) + len(r["answer"]) - max_seq
        if over > 0:
            cut = min(over, max(0, len(r["prompt"]) - 64))
            r["prompt"] = r["prompt"][cut:]  # keep the end of the context
            over -= cut
            if over > 0:
                if len(r["answer"]) - over < 32:
                    continue
                r["answer"] = r["answer"][: len(r["answer"]) - over]  # then the answer's tail
        rows.append(r)
    random.Random(seed).shuffle(rows)
    held = holdout - holdout % world
    return rows[held:][rank::world], rows[:held][rank::world]


def moved(student) -> float:
    """Codes on this rank's shards that differ from the ones the run started with."""
    n = 0
    with torch.no_grad():
        for m in student.modules():
            if isinstance(m, PrismLinear) and m.trainable and not m.codes_fixed:
                lat = m.latent.to_local() if hasattr(m.latent, "to_local") else m.latent
                c0 = m.codes0.to_local() if hasattr(m.codes0, "to_local") else m.codes0
                n += int((torch.clamp(torch.round(lat), -1, 1).to(torch.int8) != c0).sum())
    return float(n)


def chunked_kl(
    student_head, teacher_head, h_s, h_t, answer_ids, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum over positions of KL(teacher || student) and of the student's top-1 agreement, head
    applied chunk by chunk under activation checkpointing (the logits are 248k wide)."""

    def piece(hs, ht):
        with torch.no_grad():
            tl = torch.log_softmax(teacher_head(ht).float(), -1)
        sl = torch.log_softmax(student_head(hs).float(), -1)
        kl = (tl.exp() * (tl - sl)).sum(-1).sum()
        agree = (tl.argmax(-1) == sl.argmax(-1)).sum()
        return kl, agree.float()

    kl = torch.zeros((), device=h_s.device, dtype=torch.float32)
    agree = torch.zeros((), device=h_s.device, dtype=torch.float32)
    for s in range(0, h_s.shape[0], chunk):
        k, a = checkpoint(piece, h_s[s : s + chunk], h_t[s : s + chunk], use_reentrant=False)
        kl = kl + k
        agree = agree + a
    return kl, agree


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--teacher-dir", type=Path, help="teacher snapshot; default: chain.toml's")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--train", default="attention", choices=sorted(TRAIN_SETS))
    p.add_argument(
        "--train-small",
        action="store_true",
        help="train the tensors that are not folded linears too: norms, deltanet alpha, beta, "
        "conv, A_log, dt bias",
    )
    p.add_argument(
        "--scales-only",
        action="store_true",
        help="hold the codes, train the scales (the control run)",
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="the latent codes' learning rate, in code steps: a code flips after moving 0.5",
    )
    p.add_argument("--scale-lr", type=float, default=1e-5, help="the group scales' learning rate")
    p.add_argument("--small-lr", type=float, default=1e-4, help="the --train-small tensors' rate")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--optimizer",
        default="adamw",
        choices=("adamw", "adafactor"),
        help="adafactor keeps factored second moments only: the whole model fits 80 GB GPUs",
    )
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--tight-memory",
        action="store_true",
        help="release cached GPU memory after each forward (slower)",
    )
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--holdout", type=int, default=64, help="sequences kept out of training")
    p.add_argument("--eval-every", type=int, default=20, help="held-out KL every this many steps")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    # long sequences fragment the default allocator; this must be set before CUDA starts
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # one Triton cache per process: ranks compiling the same kernel into one cache race on its files
    cache = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
    os.environ["TRITON_CACHE_DIR"] = str(cache / f"rank{os.environ.get('LOCAL_RANK', '0')}")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(a.seed)
    from common import models
    from common.config import chain_spec

    teacher_dir = a.teacher_dir or models.snapshot(
        chain_spec().teacher_repo, chain_spec().teacher_revision
    )
    from transformers import Qwen3_5ForCausalLM

    # student: rank 0 reads the gguf, the others build the skeleton on the meta device; the full
    # state is captured on
    # rank 0 before sharding and broadcast into the shards
    want = TRAIN_SETS[a.train]
    train_only = {path for path in linear_map_names(teacher_dir) if want(path)}
    log(rank, f"{len(train_only)} trainable linears ({a.train}); loading the student")
    student = load_model(
        a.gguf,
        teacher_dir,
        trainable=True,
        train_only=train_only,
        frozen_as="bf16",
        skeleton=rank != 0,
        log=(print if rank == 0 else (lambda *x: None)),
    )
    if a.scales_only:
        for m in student.modules():
            if isinstance(m, PrismLinear) and m.trainable:
                m.codes_fixed = True
                m.latent.requires_grad_(False)
    small_names = set(small_map(list(student.config.layer_types)))
    if a.train_small:
        values = small_tensors(a.gguf, read_header(a.gguf), student.config) if rank == 0 else {}
        for n in small_names:
            if rank == 0:
                set_param(student, n, nn.Parameter(values[n].float(), requires_grad=True))
            else:
                prm = student.get_parameter(n)
                prm.data = prm.data.float()
                prm.requires_grad_(True)
    student_sd = {k: v for k, v in student.state_dict().items()} if rank == 0 else {}
    # teacher: built from the config on the meta device everywhere, weights read from its
    # safetensors on rank 0
    with torch.device("meta"):
        teacher = Qwen3_5ForCausalLM._from_config(
            student.config, attn_implementation="sdpa", dtype=torch.bfloat16
        )
    teacher.requires_grad_(False)
    teacher_sd = teacher_state(teacher_dir, dict(teacher.state_dict()).keys()) if rank == 0 else {}

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for m in (student, teacher):
        # the embedding, the final norm and the head get their own groups: the trainer calls the
        # inner model and
        # the head separately, and only a module's own group unshards it
        # the head is not sharded: it is applied per chunk of answer positions and the ranks have
        # different chunk
        # counts, and a sharded module's forward is a collective
        for sub in (m.model.embed_tokens, m.model.norm, *m.model.layers):
            fully_shard(sub, mp_policy=mp, reshard_after_forward=True)
        fully_shard(
            m, mp_policy=mp, reshard_after_forward=True, ignored_params=set(m.lm_head.parameters())
        )
    student.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    for m, name, sd in ((student, "student", student_sd), (teacher, "teacher", teacher_sd)):
        if rank != 0 or name == "teacher":
            m.to_empty(device=device)
        else:
            m.lm_head.to(device)  # rank 0's replicated head is still on the CPU
        set_model_state_dict(
            m, sd, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        )
        del sd
        for t in (*m.lm_head.parameters(), *m.lm_head.buffers()):  # the replicated head, by hand
            dist.broadcast(t.data, src=0)
        m.model.rotary_emb = type(m.model.rotary_emb)(config=m.config).to(device)
        m.eval() if name == "teacher" else m.train()
        log(rank, f"{name} sharded and on GPUs")
    del student_sd, teacher_sd
    dist.barrier()
    torch.cuda.empty_cache()
    log(rank, f"GPU memory after loading: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB")

    latent = [q for n, q in student.named_parameters() if q.requires_grad and n.endswith(".latent")]
    scales = [q for n, q in student.named_parameters() if q.requires_grad and n.endswith(".scale")]
    small = [q for n, q in student.named_parameters() if q.requires_grad and n in small_names]
    log(
        rank,
        f"trainable: {sum(q.numel() for q in latent) / 1e9:.2f} B latent, "
        f"{sum(q.numel() for q in scales) / 1e6:.0f} M scales, "
        f"{sum(q.numel() for q in small) / 1e6:.1f} M other",
    )
    groups = [
        {"params": latent, "lr": a.lr},
        {"params": scales, "lr": a.scale_lr},
        {"params": small, "lr": a.small_lr},
    ]
    if a.optimizer == "adamw":
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, foreach=True)
    else:
        from transformers.optimization import Adafactor

        opt = Adafactor(
            groups, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=0.0
        )
    base_lrs = [g["lr"] for g in opt.param_groups]

    rows, held = load_rows(a.data, a.max_seq, a.seed, rank, world, a.holdout)
    log(rank, f"{len(rows)} training and {len(held)} held-out sequences per rank")
    n_codes = sum(
        m.latent.numel() for m in student.modules() if isinstance(m, PrismLinear) and m.trainable
    )

    def held_out(step: int) -> None:
        if not held:
            return
        tot = torch.zeros(2, device=device)
        with torch.no_grad():
            for r in held:
                ids = torch.tensor([r["prompt"] + r["answer"]], device=device)
                first, last = len(r["prompt"]) - 1, len(r["prompt"]) - 1 + len(r["answer"])
                h_t = teacher.model(input_ids=ids, use_cache=False).last_hidden_state[0, first:last]
                h_s = student.model(input_ids=ids, use_cache=False).last_hidden_state[0, first:last]
                kl, _ = chunked_kl(student.lm_head, teacher.lm_head, h_s, h_t, r["answer"], a.chunk)
                tot += torch.stack((kl, torch.tensor(float(len(r["answer"])), device=device)))
                if a.tight_memory:
                    torch.cuda.empty_cache()
        dist.all_reduce(tot)
        log(
            rank,
            f"step {step}: held-out KL {tot[0].item() / tot[1].item():.5f} "
            f"(every token weighing the same, {int(tot[1].item())} tokens)",
        )

    held_out(0)
    a.out.mkdir(parents=True, exist_ok=True)
    stats = {"loss": 0.0, "tokens": 0, "agree": 0.0, "n": 0}
    t0 = time.perf_counter()
    cursor = 0
    for step in range(1, a.steps + 1):
        warm = min(1.0, step / max(1, a.warmup))
        for g, lr0 in zip(opt.param_groups, base_lrs, strict=True):
            g["lr"] = lr0 * warm
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            r = rows[cursor % len(rows)]
            cursor += 1
            ids = torch.tensor([r["prompt"] + r["answer"]], device=device)
            first = len(r["prompt"]) - 1
            last = first + len(r["answer"])
            with torch.no_grad():
                h_t = teacher.model(input_ids=ids, use_cache=False).last_hidden_state[0, first:last]
            if a.tight_memory:
                torch.cuda.empty_cache()
            h_s = student.model(input_ids=ids, use_cache=False).last_hidden_state[0, first:last]
            kl, agree = chunked_kl(student.lm_head, teacher.lm_head, h_s, h_t, r["answer"], a.chunk)
            loss = kl / len(r["answer"]) / a.grad_accum
            if a.tight_memory:
                torch.cuda.empty_cache()
            loss.backward()
            if a.tight_memory:
                torch.cuda.empty_cache()
            stats["loss"] += kl.item()
            stats["tokens"] += len(r["answer"])
            stats["agree"] += agree.item()
            stats["n"] += 1
        gn = torch.nn.utils.clip_grad_norm_([*latent, *scales, *small], a.clip)
        opt.step()
        if step % a.log_every == 0:
            tot = torch.tensor(
                [stats["loss"], stats["tokens"], stats["agree"], moved(student)], device=device
            )
            dist.all_reduce(tot)
            el = time.perf_counter() - t0
            log(
                rank,
                f"step {step}: KL/token {tot[0].item() / tot[1].item():.5f}  "
                f"top-1 {tot[2].item() / tot[1].item():.4f}  grad {gn.item():.3f}  "
                f"lr {opt.param_groups[0]['lr']:.2e}  "
                f"{tot[1].item() / el:.0f} answer tok/s (all ranks)  "
                f"codes moved {int(tot[3].item()):,} "
                f"({100 * tot[3].item() / max(1, n_codes):.3f}%)  "
                f"mem {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB",
            )
            stats = {"loss": 0.0, "tokens": 0, "agree": 0.0, "n": 0}
            t0 = time.perf_counter()
        if a.eval_every and step % a.eval_every == 0:
            held_out(step)
        if a.save_every and step % a.save_every == 0:
            export(student, a.out / f"step-{step}.safetensors", rank, teacher_dir)
    export(student, a.out / "final.safetensors", rank, teacher_dir)
    dist.barrier()
    dist.destroy_process_group()


def teacher_state(teacher_dir: Path, wanted) -> dict:
    """The text model's weights from the conditional-generation checkpoint, renamed to the causal
    LM's keys."""
    import json

    from safetensors import safe_open

    index = json.loads((teacher_dir / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = set(wanted)
    out = {}
    for key, file in index.items():
        new = key.replace("model.language_model.", "model.")
        if new in wanted:
            with safe_open(teacher_dir / file, "pt") as f:
                out[new] = f.get_tensor(key).to(torch.bfloat16)
    missing = wanted - set(out)
    if missing:
        raise KeyError(f"teacher checkpoint lacks {len(missing)} keys, e.g. {sorted(missing)[:5]}")
    return out


def linear_map_names(teacher_dir: Path) -> list[str]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(teacher_dir)
    return list(linear_map(list(getattr(cfg, "text_config", cfg).layer_types)))


def export(student, path: Path, rank: int, teacher_dir: Path) -> None:
    """Codes (int8) and scales (fp16) of every trainable linear, plus the trained small tensors,
    gathered to rank 0."""
    from safetensors.torch import save_file

    names = linear_map_names(teacher_dir)
    gmap = linear_map_for(teacher_dir)
    out, flips, total = {}, 0, 0
    for hf_path in names:
        mod = student.get_submodule(hf_path)
        if not isinstance(mod, PrismLinear) or not mod.trainable:
            continue
        latent = mod.latent.full_tensor() if hasattr(mod.latent, "full_tensor") else mod.latent
        scale = mod.scale.full_tensor() if hasattr(mod.scale, "full_tensor") else mod.scale
        codes0 = mod.codes0.full_tensor() if hasattr(mod.codes0, "full_tensor") else mod.codes0
        with torch.no_grad():
            s = scale.abs().clamp_min(1e-8)
            codes = (
                codes0.clone()
                if mod.codes_fixed
                else torch.clamp(torch.round(latent), -1, 1).to(torch.int8)
            )
            flips += int((codes != codes0).sum())
            total += codes.numel()
            if rank == 0:
                out[gmap[hf_path] + ":codes"] = codes.cpu()
                out[gmap[hf_path] + ":scales"] = s.to(torch.float16).cpu()
    small_names = set(small_map(list(student.config.layer_types)))
    for n, prm in student.named_parameters():
        if prm.requires_grad and n in small_names:
            t = prm.full_tensor() if hasattr(prm, "full_tensor") else prm
            if rank == 0:
                out["small:" + n] = t.detach().float().cpu()
    if rank == 0:
        out["meta:flipped"] = torch.tensor([flips, total])
        save_file(out, str(path))
        print(
            time.strftime("%H:%M:%S"),
            f"saved {path.name}: {len(out)} tensors, {flips:,} of {total:,} codes moved "
            f"({100 * flips / max(1, total):.3f}%)",
            flush=True,
        )


def linear_map_for(teacher_dir: Path) -> dict[str, str]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(teacher_dir)
    return linear_map(list(getattr(cfg, "text_config", cfg).layer_types))


if __name__ == "__main__":
    main()
