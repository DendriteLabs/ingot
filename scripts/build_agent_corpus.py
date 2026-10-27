"""Rebuild ingot_corpus's chungus split with agent trajectories in swe and terminal: every other
domain's rows as they were and, in place of those two, whole agent trajectories in the conversation
column, which the eval's draw cuts before one of their assistant turns (eval_service.dataset._cut).

    uv run --with pyarrow python scripts/build_agent_corpus.py <corpus dir> \
        <SWE-rebench-openhands-trajectories dir> <Nemotron-Terminal-Corpus dir> \
        <exclude.json> <out dir>

exclude.json is {"repos": [...], "texts": [...]}: SWE-bench Multilingual's repositories and
Terminal-Bench's task instructions, whose trajectories are left out.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

AGENT_FILE_ROWS = 5_000
COLUMNS = ("dataset", "split", "entry_name", "domain", "language", "user_prompt", "conversation")


def _drop_none(value: Any) -> Any:
    # pyarrow gives every tool the union of all tools' fields, the ones it lacks as None
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


def _message(m: dict[str, Any]) -> dict[str, Any]:
    """A trajectory message as the teacher's template takes it: string content without earlier
    reasoning (an agent never sends it back), tool-call arguments as dicts."""
    content = m.get("content") or ""
    if m["role"] == "assistant" and "</think>" in content:
        content = content.split("</think>")[-1].lstrip()
    out = {"role": m["role"], "content": content}
    if m.get("tool_calls"):
        calls = [
            (c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
            for c in m["tool_calls"]
        ]
        if not all(isinstance(args, dict) for _, args in calls):
            raise ValueError("tool-call arguments that are not an object")
        out["tool_calls"] = [
            {"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls
        ]
    return out


def _row(domain: str, dataset: str, key: str, raw: list[dict[str, Any]], tools: Any) -> dict | None:
    """The trajectory as a corpus row; None when it is malformed or has no turn to cut before."""
    try:
        messages = [_message(m) for m in raw]
    except (ValueError, KeyError):
        return None
    if sum(m["role"] == "assistant" for m in messages) < 2:
        return None
    conversation = {"messages": messages} | ({"tools": tools} if tools else {})
    return {
        "dataset": dataset,
        "split": "train",
        "entry_name": key,
        "domain": domain,
        "language": "en",
        "user_prompt": next(m["content"] for m in messages if m["role"] == "user"),
        "conversation": json.dumps(conversation, ensure_ascii=False),
    }


def swe_rows(root: Path, repos: set[str]) -> Iterator[dict | None]:
    """SWE-rebench OpenHands trajectories (67k runs over 6.3k tasks), but none from a repository
    in SWE-bench Multilingual."""
    parquet = pq.ParquetFile(root / "trajectories.parquet")
    for group in range(parquet.num_row_groups):
        rows = parquet.read_row_group(
            group, columns=["trajectory_id", "repo", "trajectory", "tools"]
        )
        for row in rows.to_pylist():
            if row["repo"] not in repos:
                tools = _drop_none(row["tools"])
                yield _row(
                    "swe", "swe_rebench_openhands", row["trajectory_id"], row["trajectory"], tools
                )


def terminal_rows(root: Path, texts: list[str]) -> Iterator[dict | None]:
    """Nemotron-Terminal synthetic-task trials (Terminus-2); a trial whose task holds a
    Terminal-Bench instruction is left out."""
    for f in sorted(root.glob("synthetic_tasks/skill_based/*/*/data_filtered.parquet")):
        for row in pq.read_table(f, columns=["trial_name", "conversations"]).to_pylist():
            task = row["conversations"][0]["content"]
            if any(text in task for text in texts):
                print(f"  left out {row['trial_name']}: a Terminal-Bench task")
                continue
            yield _row(
                "terminal",
                "term_nemotron_trajectories",
                row["trial_name"],
                row["conversations"],
                None,
            )


def main() -> None:
    corpus, swe_root, term_root, exclude_path, out = map(Path, sys.argv[1:6])
    exclude = json.loads(exclude_path.read_text())
    agent: list[dict[str, Any]] = []
    for domain, rows in (
        ("swe", swe_rows(swe_root, set(exclude["repos"]))),
        ("terminal", terminal_rows(term_root, exclude["texts"])),
    ):
        rows = list(rows)
        kept = [r for r in rows if r]
        print(
            f"{domain}: {len(kept):,} trajectories, {len(rows) - len(kept):,} left out", flush=True
        )
        agent += kept
    schema = pa.schema([(c, pa.large_string()) for c in COLUMNS])
    old = sorted(corpus.glob("data/chungus-*.parquet"))
    total = len(old) + -(-len(agent) // AGENT_FILE_ROWS)
    (out / "data").mkdir(parents=True, exist_ok=True)

    def name(i: int) -> Path:
        return out / "data" / f"chungus-{i:05d}-of-{total:05d}.parquet"

    def rewrite(i: int) -> int:
        table = pq.read_table(old[i])
        # pyarrow.compute builds its functions at import time, so ty cannot see them
        table = table.filter(pc.invert(pc.is_in(table["domain"], pa.array(["swe", "terminal"]))))  # ty: ignore[unresolved-attribute]
        table = table.append_column("conversation", pa.nulls(len(table), pa.large_string()))
        pq.write_table(table.cast(schema), name(i), compression="zstd", row_group_size=10_000)
        return len(table)

    # pyarrow reads, compresses and writes outside the GIL
    with ThreadPoolExecutor(16) as pool:
        kept = sum(pool.map(rewrite, range(len(old))))
    for j, i in enumerate(range(len(old), total)):
        chunk = agent[j * AGENT_FILE_ROWS : (j + 1) * AGENT_FILE_ROWS]
        table = pa.Table.from_pylist(chunk, schema)
        pq.write_table(table, name(i), compression="zstd", row_group_size=500)
    print(f"{kept:,} rows kept, {len(agent):,} agent rows, {total} files in {out / 'data'}")


if __name__ == "__main__":
    main()
