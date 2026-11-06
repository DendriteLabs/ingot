# Working with a q2 model

This folder shows how to work with a q2 model in PyTorch: load a king's GGUF as an exact model,
change what you want in it, write it back into a file the validator accepts, make training data
the way the eval makes its own, and see where the king reads the teacher worst. It is for miners
who know their way around PyTorch, KL divergence and multi-GPU training. What you do with it is
up to you.

## What you can change

A q2 submission is genesis's GGUF file with your own numbers in it. Almost all of those numbers
are ternary: every weight of every large matrix is −1, 0 or +1, and each run of 128 weights
shares one fp16 scale. Three kinds of values are yours to set:

| what | how many | what it does |
|---|---:|---|
| the codes | 26.9 billion | which way each weight points, or whether it is off |
| the scales | 210 million | how strongly each group of 128 weights acts |
| the unquantized tensors | 26 million | norms and a few small tensors in the linear-attention layers |

All three load as parameters you can train, and the packer writes any subset back.

## What's in it

| module | does |
|---|---|
| `model` | loads a king's GGUF as a PyTorch `Qwen3_5ForCausalLM`, exact or trainable |
| `train` | a training loop to start from: FSDP2, KL to the teacher, held-out KL, codes moved |
| `pack` | writes trained codes, scales and small tensors into a byte copy of the king's file |
| `teacher_data` | draws from the eval's corpus as the eval does and has the teacher answer |
| `weak_spots` | where the current king reads the teacher worst, from the site's eval records |
| `numerics_check` | compares the PyTorch model with the runtime on your box |
| `basis_check` | tests the file's rotation and head order against the original Qwen weights |

The loader keeps the codes in the file's rotated basis and rotates the activations the way the
runtime does, so what you train is the stored code itself. In trainable mode each code gets a
float latent in units of its group's scale: it starts at the code, and the code flips once the
latent has moved half a step; `--scales-only` holds the codes. Norms and the other unquantized
tensors load as ordinary parameters. The format details are in `llms.txt` ("The file format").

## Use it

```bash
bash scripts/setup_eval_box.sh && bash scripts/build_llama.sh

# where the king reads the teacher worst; this part needs no GPU
uv run python -m miner.ternary.weak_spots

# data: the king's worst samples, fresh draws, and a draw you keep for scoring
.venv/bin/python -m miner.ternary.weak_spots --out worst.jsonl --n 400
.venv/bin/python -m miner.ternary.teacher_data --data worst.jsonl --draw s1 s2 --out teacher-data.jsonl
.venv/bin/python -m miner.ternary.teacher_data --draw holdout --write-draw holdout.jsonl

# check that the PyTorch model matches the runtime on your box
.venv/bin/python -m miner.ternary.numerics_check --gguf-dir <dir with the king's gguf> \
    --prompts holdout.jsonl --n 8 --out numerics.json
```

## Train

`train` is a complete loop: it shards the student and a frozen BF16 teacher over 8 GPUs with
FSDP2, minimizes KL(teacher ‖ student) over the answer tokens, prints the KL on held-out
sequences and the number of codes that moved as it goes, and exports a delta for `pack`.

```bash
.venv/bin/torchrun --nproc_per_node 8 -m miner.ternary.train --gguf <king>.gguf \
    --data teacher-data.jsonl --train attention --train-small --steps 200 --lr 0.02 \
    --max-seq 6144 --tight-memory --out run-attn
.venv/bin/python -m miner.ternary.pack --base <king>.gguf --delta run-attn/final.safetensors \
    --out mine/model.gguf
```

What it does out of the box is a starting point, not a recipe. In our own run from King XXVII
(1,400 teacher answers, 3.5 M tokens, 16 sequences a step, 80 steps) the training KL did not fall,
no code moved at a learning rate of 0.01, and at 0.02 the held-out KL rose 7.5% once half a
million codes had flipped. If you go this way, the things we would try are more tokens per step,
a longer warm-up, lower rates for the scales and the small tensors, and a layer-wise objective in
place of end-to-end gradients.

| flag | trains |
|---|---|
| `--train full-attention` | only the 16 full-attention layers, the smallest set |
| `--train attention` | every attention and linear-attention projection (7.2 billion weights) |
| `--train mlp` | the MLP projections |
| `--train all` | every matrix except the output head (add `--optimizer adafactor` on 80 GB cards) |
| `--train-small` | adds the norms and the other unquantized tensors to any of the above |
| `--scales-only` | holds the codes and trains the scales |

Or write your own loop against `model` and `pack`:

```python
from pathlib import Path
from miner.ternary.model import load_model
from miner.ternary.pack import delta_from_model, pack

king = Path("king.gguf")
model = load_model(
    king, teacher_dir, trainable=True, train_only={"model.layers.3.self_attn.q_proj", ...}
)
# ... your training: KL(teacher ‖ model) on the answers in teacher-data.jsonl ...
pack(king, delta_from_model(model), Path("mine/model.gguf"), model.config)
```

`delta_from_model` reads a model trained in one process; under FSDP, gather the full latents and
scales first, or save a delta file as `train` does.

## Check and submit

```bash
# free, and catches anything the validator would reject
.venv/bin/ingot check-model --path mine/model.gguf

# optional: duel it against the king, worth it when a registration costs more than
# half an hour of your box
.venv/bin/python scripts/eval_smoke.py --dataset holdout.jsonl --samples 800 \
    --king <king dir> --challenger mine
```

## Tips

- **Train on what the king gets wrong.** Every eval scores the king on a fresh draw and the site
  publishes each sample's KL. `weak_spots` reads every duel the current king has played and shows
  where it reads the teacher worst, by domain, source and stratum, with each part's share of its
  score. With `--out` it rebuilds the worst samples exactly as the evals drew them; have the
  teacher answer those and weight them in your training data.
- **Watch held-out KL, not the training loss.** `train` prints it every 20 steps, with the number
  of codes moved; if the held-out KL climbs, stop and change the recipe, whatever the training
  loss says. A crown needs the margin in `llms.txt` over the king on a fresh draw.
- **Don't reuse someone else's exact recipe.** The same deterministic recipe on the same data
  gives the same model, and the second submission of it is rejected as a duplicate. Vary the
  seeds, the data or the settings.
