"""Build the eval dataset from ingot_corpus: one JSONL line per usable prompt, as read by
eval_service.dataset.load (id, messages, domain, source, plus language for analysis).

    uv run --with pyarrow --with transformers --with jinja2 \
        python scripts/build_eval_corpus.py <corpus dir> <teacher dir> <out.jsonl>

Keeps one row per distinct prompt text (the smallest id), and only prompts whose rendered length
leaves the full answer budget, so no sample's answer is cut short by the context.
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from eval_service import config
from eval_service.run import _render

corpus, teacher, out = (Path(arg) for arg in sys.argv[1:4])
table = pq.read_table(sorted(glob.glob(str(corpus / "data" / "*.parquet"))))
rows = table.select(["dataset", "entry_name", "domain", "language", "user_prompt"]).to_pylist()
for row in rows:
    row["id"] = f"{row['domain']}/{row['dataset']}/{row['language']}/{row['entry_name']}"
rows.sort(key=lambda row: row["id"])
if len({row["id"] for row in rows}) != len(rows):
    raise ValueError("ids are not unique")

limit = config.MAX_MODEL_LEN - config.MAX_TOKENS
tokenizer = AutoTokenizer.from_pretrained(teacher)
# A byte-level BPE token covers at least one UTF-8 byte (a character is at most 4), so a prompt this
# short fits without being rendered.
frame = len(_render(tokenizer, [{"role": "user", "content": ""}]))
safe_chars = (limit - frame) // 4

seen: set[str] = set()
kept, dropped = collections.Counter(), collections.Counter()
lines = []
for row in rows:
    text = row["user_prompt"]
    if not text or text in seen:
        dropped["empty or duplicate"] += 1
        continue
    seen.add(text)
    messages = [{"role": "user", "content": text}]
    if len(text) > safe_chars and len(_render(tokenizer, messages)) > limit:
        dropped[f"over {limit} prompt tokens ({row['domain']})"] += 1
        continue
    record = {
        "id": row["id"],
        "domain": row["domain"],
        "source": row["dataset"],
        "language": row["language"],
        "messages": messages,
    }
    lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    kept[row["domain"]] += 1

data = "".join(lines).encode()
out.write_bytes(data)
print(f"kept {sum(kept.values()):,} prompts, sha256 {hashlib.sha256(data).hexdigest()}")
for domain, count in kept.most_common():
    print(f"  {domain:24} {count:9,}")
for reason, count in dropped.most_common():
    print(f"  dropped {count:9,}  {reason}")
