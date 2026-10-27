# Ingot

A Bittensor subnet where miners quantize a fixed teacher model and the validator crowns the
copy that stays closest to it. Kings are published; the newest kings share the emission.

- `llms.txt` — how the network works and how to mine.
- `AGENTS.md` — how the code is organised and how we work on it.
- `chain.toml` — what miners and the validator agree on: the quantizations, the open one
  (`[competition] active_quant`), their genesis and limits.

## Develop

```bash
git submodule update --init   # llama-prism: the Dendrite llama.cpp fork (PrismML's plus our changes) that runs q2 models
uv run pre-commit install     # ruff + ty on every commit
```

## Database

`schema.sql` is idempotent: apply it on every deploy (`psql -f schema.sql`). Kings carry their
quantization; a model digest is unique only among submissions that have not failed.

## Eval box (8 GPUs)

```bash
bash scripts/setup_eval_box.sh                # .venv with vLLM 0.31.0; never `uv sync` here
bash scripts/build_llama.sh                   # llama-prism with CUDA, and llama-states (q2 reads)
.venv/bin/python scripts/eval_smoke.py        # teacher vs the fp8 genesis on a few prompts
.venv/bin/python scripts/eval_smoke.py --king DIR --challenger DIR   # a gguf model: its folder
pm2 start pm2/ecosystem.eval-service.config.js
```

The teacher stays on vLLM for every quantization. A gguf king and challenger each run one
`llama-states` per GPU of their role (`INGOT_KING_GPUS`, `INGOT_CHALLENGER_GPUS`).

The validator's dedup bank, one OpenSearch index per quantization (`ingot_dedup-<quant>`):
`scripts/setup_opensearch.sh` starts OpenSearch and waits until it is healthy; `--seed` then
fingerprints the open quantization's genesis into it (on the GPU box), generating
`INGOT_DEDUP_SECRET` in `.env` if it is empty. Another quantization:
`uv run --extra dedup scripts/seed_dedup_bank.py --quant <quant>`.

## Opening a quantization

```bash
uv run scripts/pin_gguf.py qwen38_27b q2          # a gguf genesis: pin its header (once)
uv run scripts/pin_family.py qwen38_27b fp8       # a safetensors genesis: config and layout
uv run scripts/create_genesis.py --quant q2       # its genesis king row
uv run --extra dedup scripts/seed_dedup_bank.py --quant q2
```

Then set `[competition] active_quant` in `chain.toml` and restart every service together: each
reads `chain.toml` once at start. Waiting submissions of the closed quantization end FAILED
"closed" with their attempt handed back.

## Website

```bash
uv run scripts/push_site.py                     # web/ to the site bucket (R2_SITE_BUCKET_NAME)
pm2 start pm2/ecosystem.dashboard.config.js     # keeps the bucket's api/ current: state.json, eval pages
uv run ingot-dashboard-preview                  # local preview on http://127.0.0.1:8000
```

## Mine

```bash
uv sync && cp .env.example_miners .env
uv run ingot register
uv run ingot check-model --path ./model.gguf
uv run ingot submit-private --path ./model.gguf
```
