"""Rebuild ingot_corpus's chungus split with the agent trajectories of swe and terminal: every other
domain's rows as they were and, in place of those two, whole trajectories in the conversation
column, rewritten to mini-swe-agent's format (common.agent_traces), which the eval's draw cuts
before one of their assistant turns (eval_service.dataset._cut).

    uv run --with pyarrow --with huggingface_hub python scripts/build_agent_corpus.py \\
        --open-swe <Open-SWE-Traces dir> --ukisai <parquet> --pivot <jsonl> --lhtb <LHTB dir> \\
        --out <out dir> [--corpus <current corpus dir>]

Sources (chosen by their teacher-forced KL against real benchmark runs, duels of 2026-10-27):
  swe       nvidia/Open-SWE-Traces, mini-swe-agent runs by Qwen3.8-27B (CC-BY-4.0); language = the
            repository's, so the draw spreads swe over languages as SWE-bench Multilingual does
  terminal  IntelligenceLab/LHTB-leaderboard, Terminus-2 runs on the 77 Long-Horizon Terminal-Bench
            2.0 tasks (Apache-2.0) as language "lhtb"; ukisai/Qwen3.8-27B-multi-turn-agent-sft
            (Apache-2.0) and nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1 (CC-BY-4.0) as "broad".
            Terminal-Bench-Science runs are not redistributable: eval_service.dataset.fetch adds
            them at draw time (chain.toml [dataset.extra]) as "tbsci".
Runs on a SWE-bench Multilingual repository or a Terminal-Bench 2.1 task are left out; the lists
come from the benchmarks' own repos. Without --corpus only the agent files are written.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from common.agent_traces import (
    BASH_TOOL,
    COLUMNS,
    corpus_row,
    from_mini_swe,
    from_terminus,
    strip_thinking,
)

AGENT_FILE_ROWS = 5_000
MULTILINGUAL = ("SWE-bench/SWE-bench_Multilingual", "846e647b9f33c0b51b739d005d13d85493c9af09")
TERMINAL_BENCH = ("harborframework/terminal-bench-2.1", "3e235dff6880252a587fa479c09bcb1e16edf2eb")


def benchmark_exclusions() -> tuple[set[str], list[str]]:
    """SWE-bench Multilingual's repositories and the first lines of Terminal-Bench 2.1's task
    instructions, from the benchmarks' pinned repos."""
    root = Path(
        snapshot_download(
            MULTILINGUAL[0],
            revision=MULTILINGUAL[1],
            repo_type="dataset",
            allow_patterns="data/*.parquet",
        )
    )
    repos = {
        r
        for f in root.glob("data/*.parquet")
        for r in pq.read_table(f, columns=["repo"])["repo"].to_pylist()
    }
    root = Path(
        snapshot_download(
            TERMINAL_BENCH[0],
            revision=TERMINAL_BENCH[1],
            repo_type="dataset",
            allow_patterns="tasks/*/instruction.md",
        )
    )
    texts = [f.read_text().strip()[:200] for f in sorted(root.glob("tasks/*/instruction.md"))]
    return repos, texts


def swe_rows(root: Path, repos: set[str]) -> Iterator[dict | None]:
    for f in sorted(
        glob.glob(str(root / "data" / "minisweagent" / "qwen38_27b" / "*" / "*.parquet"))
    ):
        parquet = pq.ParquetFile(f)
        for group in range(parquet.num_row_groups):
            columns = ["trajectory_id", "repo", "language", "messages", "tools"]
            for row in parquet.read_row_group(group, columns=columns).to_pylist():
                if row["repo"] in repos:
                    continue
                try:
                    messages, tools = from_mini_swe(row["messages"], row["tools"])
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    yield None
                    continue
                yield corpus_row(
                    "swe",
                    "open_swe_minisweagent",
                    row["trajectory_id"],
                    row["language"],
                    messages,
                    tools,
                )


def lhtb_rows(root: Path) -> Iterator[dict | None]:
    """Every LHTB 2.0 leaderboard entry's trials (Terminus-2 as ATIF)."""
    from common.agent_traces import from_atif

    pattern = "submissions/longer-horizon-terminal-bench-2.0/*/*/*/*/agent/trajectory.json"
    for f in sorted(root.glob(pattern)):
        entry, trial = f.parts[-5], f.parts[-3]
        try:
            messages = from_atif(json.loads(f.read_text()).get("steps") or [])
        except (StopIteration, KeyError, TypeError, ValueError):
            yield None
            continue
        yield corpus_row(
            "terminal", "lhtb_leaderboard", f"{entry}/{trial}", "lhtb", messages, BASH_TOOL
        )


def ukisai_rows(parquet: Path, texts: list[str]) -> Iterator[dict | None]:
    """Qwen3.8-27B's Terminus-2 runs on the OpenThoughts-Agent task base."""
    for row in pq.read_table(parquet, columns=["conversations", "trial_name"]).to_pylist():
        if _benchmark_task(row["conversations"][0]["content"], texts):
            continue
        messages = [
            {
                "role": m["role"],
                "content": strip_thinking(m["content"])
                if m["role"] == "assistant"
                else m["content"],
            }
            for m in row["conversations"]
        ]
        yield _terminus_row("ukisai_qwen38", row["trial_name"], messages)


def pivot_rows(path: Path, texts: list[str]) -> Iterator[dict | None]:
    """Each Pivot trajectory from its last decision point: that record's prompt plus its expected
    answer is the whole run (one record per decision point, the prompt growing with each)."""
    last: dict[str, tuple[int, list[dict[str, str]]]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            meta = r["metadata"]
            uid, turn = meta["source_trajectory_uid"], meta["pivot_agent_turn_index"]
            if uid not in last or turn > last[uid][0]:
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in r["responses_create_params"]["input"]
                ]
                last[uid] = (
                    turn,
                    messages + [{"role": "assistant", "content": r["expected_answer"]}],
                )
    for uid, (_, messages) in sorted(last.items()):
        if _benchmark_task(messages[0]["content"], texts):
            continue
        yield _terminus_row("nemotron_terminal_pivot", uid, messages)


def _terminus_row(dataset: str, key: str, messages: list[dict[str, str]]) -> dict | None:
    try:
        converted = from_terminus(messages)
    except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None
    return corpus_row("terminal", dataset, key, "broad", converted, BASH_TOOL)


def _benchmark_task(prompt: str, texts: list[str]) -> bool:
    return any(text in prompt for text in texts)


def write_corpus(agent: list[dict], out: Path, corpus: Path | None) -> None:
    """The agent rows in files of AGENT_FILE_ROWS after the corpus's other rows, or alone."""
    schema = pa.schema([(c, pa.large_string()) for c in COLUMNS])
    old = sorted(corpus.glob("data/chungus-*.parquet")) if corpus else []
    total = len(old) + -(-len(agent) // AGENT_FILE_ROWS)
    (out / "data").mkdir(parents=True, exist_ok=True)

    def name(i: int) -> Path:
        return out / "data" / f"chungus-{i:05d}-of-{total:05d}.parquet"

    def rewrite(i: int) -> int:
        table = pq.read_table(old[i])
        # pyarrow.compute builds its functions at import time, so ty cannot see them
        table = table.filter(pc.invert(pc.is_in(table["domain"], pa.array(["swe", "terminal"]))))  # ty: ignore[unresolved-attribute]
        if "conversation" not in table.column_names:
            table = table.append_column("conversation", pa.nulls(len(table), pa.large_string()))
        pq.write_table(table.cast(schema), name(i), compression="zstd", row_group_size=10_000)
        return len(table)

    # pyarrow reads, compresses and writes outside the GIL
    with ThreadPoolExecutor(16) as pool:
        kept = sum(pool.map(rewrite, range(len(old))))
    for j, i in enumerate(range(len(old), total)):
        chunk = agent[j * AGENT_FILE_ROWS : (j + 1) * AGENT_FILE_ROWS]
        pq.write_table(
            pa.Table.from_pylist(chunk, schema), name(i), compression="zstd", row_group_size=500
        )
    print(f"{kept:,} rows kept, {len(agent):,} agent rows, {total} files in {out / 'data'}")


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("open-swe", "ukisai", "pivot", "lhtb", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--corpus", type=Path)
    a = p.parse_args()
    repos, texts = benchmark_exclusions()
    agent: list[dict] = []
    for label, rows in (
        ("swe", swe_rows(a.open_swe, repos)),
        ("terminal/lhtb", lhtb_rows(a.lhtb)),
        ("terminal/ukisai", ukisai_rows(a.ukisai, texts)),
        ("terminal/pivot", pivot_rows(a.pivot, texts)),
    ):
        rows = list(rows)
        kept = [r for r in rows if r]
        languages = collections.Counter(r["language"] for r in kept)
        print(
            f"{label}: {len(kept):,} trajectories, {len(rows) - len(kept):,} left out; "
            f"{dict(languages)}",
            flush=True,
        )
        agent += kept
    write_corpus(agent, a.out, a.corpus)


if __name__ == "__main__":
    main()
