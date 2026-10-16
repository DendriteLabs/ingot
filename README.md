# Ingot

A Bittensor subnet where miners quantize a fixed teacher model and the validator crowns the
copy that stays closest to it. Kings are published; the newest kings share the emission.

- `llms.txt` — how the network works and how to mine.
- `AGENTS.md` — how the code is organised and how we work on it.

## Develop

```bash
uv run pre-commit install    # ruff + ty on every commit
```

## Eval box (8 GPUs)

```bash
bash scripts/setup_eval_box.sh                # .venv with vLLM 0.31.0; never `uv sync` here
.venv/bin/python scripts/eval_smoke.py        # teacher vs genesis on a few prompts
pm2 start pm2/ecosystem.eval-service.config.js
```

The validator's dedup bank: `scripts/setup_opensearch.sh` starts OpenSearch and waits until it
is healthy; `--seed` then fingerprints genesis into it (on the GPU box), generating
`INGOT_DEDUP_SECRET` in `.env` if it is empty.

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
uv run ingot check-model --path ./my-model
uv run ingot submit-private --path ./my-model
```
