# ruff: noqa: E501  (the multiple-choice task text keeps its long lines)
"""Build corpus v4 from v3: five domains, each in the format its samples are scored in.

    uv run --with pyarrow --with huggingface_hub python scripts/build_corpus_v4.py \\
        --corpus <v3 data dir> --math <EleutherAI/hendrycks_math snapshot> --out <out dir>

  terminal  v3's LHTB rows as they are (the Terminal-Bench-Science rows join at draw time,
            chain.toml [dataset.extra])
  swe       v3's Open-SWE rows as they are
  tau3      simia_tau and areal_tau2 first-turn prompts (policy, tools, the user's opening message)
  science   supergpqa and mmlu_pro questions as an agent task: the question and its choices, the
            answer letter written to /app/answer.txt with the bash tool (mini-swe-agent format)
  math      hendrycks MATH train problems with a step-by-step, boxed-answer suffix
Everything else in v3 is left out.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common.agent_traces import BASH_TOOL, COLUMNS, SYSTEM, task_message

FILE_ROWS = 5_000
KEEP = {
    "lhtb_leaderboard": "terminal",
    "open_swe_minisweagent": "swe",
    "simia_tau": "tau3",
    "areal_tau2": "tau3",
    "supergpqa": "science",
    "mmlu_pro": "science",
}
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CHOICE = re.compile(r"^([A-J])\. (.*)$", re.MULTILINE)
MC_TASK = """What is the correct answer to this question:
{question}

Choices:
{choices}

Let's think step by step:

[First, reason through the problem carefully. Consider all relevant concepts, formulas, and principles. Break down complex problems into manageable steps. Evaluate each answer choice critically.]

Based on your reasoning above, determine the single, most likely answer choice. Then write ONLY the letter of your final answer ({letters}) to `/app/answer.txt`.

**Important:**
- This is a challenging graduate-level question
- Do not try to solve the question in a single step. Break it down into smaller steps.
- You should ONLY interact with the environment provided to you AND NEVER ASK FOR HUMAN HELP.
- The answer file must contain only a single uppercase letter ({letters}) with no additional text or explanation
- You can show your work and reasoning before writing the final answer, but the file `/app/answer.txt` should contain only the letter
"""


def mc_row(row: dict) -> dict | None:
    """A multiple-choice question as an agent task: the task message and the bash tool, no
    assistant turn (the teacher's answer is the first turn)."""
    prompt = row["user_prompt"] or ""
    found = list(CHOICE.finditer(prompt))
    if len(found) < 2:
        return None
    question = prompt[: found[0].start()].strip()
    letters = [m.group(1) for m in found]
    if letters != [chr(ord("A") + i) for i in range(len(letters))]:
        return None
    choices = "\n".join(f"  ({m.group(1)}) {m.group(2).strip()}" for m in found)
    task = MC_TASK.format(
        question=question, choices=choices, letters=", ".join(letters[:-1]) + f", or {letters[-1]}"
    )
    user = task_message(task)
    return {
        "dataset": row["dataset"],
        "split": "train",
        "entry_name": row["entry_name"],
        "domain": "science",
        "language": "en",
        "user_prompt": user["content"],
        "conversation": json.dumps(
            {"messages": [SYSTEM, user], "tools": BASH_TOOL}, ensure_ascii=False
        ),
    }


def v3_rows(corpus: Path):
    """The kept rows of every v3 file, file by file (a file's rows at a time keeps memory flat)."""

    def read(f: str) -> list[dict]:
        out = []
        for row in pq.read_table(f, columns=list(COLUMNS)).to_pylist():
            domain = KEEP.get(row["dataset"] or "")
            if not domain:
                continue
            if domain == "science":
                row = mc_row(row)
                if not row:
                    continue
            else:
                row = dict(row, domain=domain, conversation=row.get("conversation") or None)
            out.append(row)
        return out

    files = sorted(glob.glob(str(corpus / "*.parquet")))
    with ThreadPoolExecutor(8) as pool:
        yield from pool.map(read, files)


def math_rows(snapshot: Path) -> list[dict]:
    """hendrycks MATH train problems (every subject) with the step-by-step, boxed-answer suffix."""
    rows = []
    for f in sorted(snapshot.glob("*/train-*.parquet")):
        for i, r in enumerate(pq.read_table(f, columns=["problem", "level", "type"]).to_pylist()):
            rows.append(
                {
                    "dataset": "hendrycks_math",
                    "split": "train",
                    "entry_name": f"{f.parent.name}/{i}",
                    "domain": "math",
                    "language": "en",
                    "user_prompt": r["problem"].strip() + MATH_SUFFIX,
                    "conversation": None,
                }
            )
    return rows


class Writer:
    """Rows into files of FILE_ROWS, named with the total once it is known."""

    def __init__(self, out: Path) -> None:
        self.dir, self.pending, self.files, self.counts = out / "data", [], 0, collections.Counter()
        self.schema = pa.schema([(c, pa.large_string()) for c in COLUMNS])
        self.dir.mkdir(parents=True, exist_ok=True)

    def add(self, rows: list[dict]) -> None:
        self.pending += rows
        self.counts.update((r["domain"], r["dataset"]) for r in rows)
        while len(self.pending) >= FILE_ROWS:
            self.flush(self.pending[:FILE_ROWS])
            self.pending = self.pending[FILE_ROWS:]

    def flush(self, rows: list[dict]) -> None:
        pq.write_table(
            pa.Table.from_pylist(rows, self.schema),
            self.dir / f"chungus-{self.files:05d}.parquet",
            compression="zstd",
            row_group_size=500,
        )
        self.files += 1

    def close(self) -> None:
        if self.pending:
            self.flush(self.pending)
        for i in range(self.files):
            (self.dir / f"chungus-{i:05d}.parquet").rename(
                self.dir / f"chungus-{i:05d}-of-{self.files:05d}.parquet"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("corpus", "math", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    writer = Writer(a.out)
    for rows in v3_rows(a.corpus):
        writer.add(rows)
    writer.add(math_rows(a.math))
    writer.close()
    for (domain, dataset), n in sorted(writer.counts.items()):
        print(f"{domain:10} {dataset:24} {n:8,}")
    print(f"{sum(writer.counts.values()):,} rows, {writer.files} files in {writer.dir}")


if __name__ == "__main__":
    main()
