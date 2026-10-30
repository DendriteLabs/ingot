"""Agent trajectories from other harnesses as the eval reads them: mini-swe-agent's bash tool, the
format of the Terminal-Bench and SWE-bench runs the corpus is meant to match (duels of 2026-10-27:
the harness format alone moves a king's KL 1.7x, so every source is rewritten to it). Terminus-2
(JSON actions in text) and ATIF (Harbor's trajectory format) rewrite here; the eval's draw cuts a
row before an assistant turn and scores the teacher's answer to the prefix."""

from __future__ import annotations

import glob
import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

SYSTEM = {
    "role": "system",
    "content": "You are a helpful assistant that can interact with a computer.",
}
TASK_PREFIX = "Please solve this issue: "
# mini-swe-agent's default instance template, as the benchmark harness renders it
TASK_SUFFIX = Path(__file__).with_name("mini_swe_agent_task.txt").read_text()
BASH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                "required": ["command"],
            },
        },
    }
]
COLUMNS = ("dataset", "split", "entry_name", "domain", "language", "user_prompt", "conversation")
# Terminus-2 keystrokes that are tmux keys, not shell text
_TMUX_KEYS = ("C-", "M-", "Escape", "Enter", "Tab", "Up", "Down", "Left", "Right")
_SHELL_TOOLS = {"Bash", "bash", "run_terminal_command", "execute_command", "shell", "terminal"}


def strip_thinking(content: str) -> str:
    """An assistant turn without its reasoning: an agent never sends it back as context."""
    return content.split("</think>")[-1].lstrip() if "</think>" in content else content


def task_message(task: str) -> dict[str, str]:
    return {"role": "user", "content": TASK_PREFIX + task.strip() + TASK_SUFFIX}


def bash_call(command: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": "bash", "arguments": {"command": command}}}


def tool_result(output: str, returncode: int = 0) -> dict[str, str]:
    return {
        "role": "tool",
        "content": json.dumps({"returncode": returncode, "output": output}, indent=2),
    }


def from_mini_swe(
    messages: list[dict[str, Any]], tools: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """A run recorded by mini-swe-agent (Open-SWE-Traces): reasoning stripped, tool-call
    arguments and tool schemas as objects."""
    out = []
    for m in messages:
        if m.get("role") not in ("system", "user", "assistant", "tool"):
            continue
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        row: dict[str, Any] = {
            "role": m["role"],
            "content": strip_thinking(content) if m["role"] == "assistant" else content,
        }
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c["function"]["arguments"]
                args = json.loads(args or "{}") if isinstance(args, str) else args
                if not isinstance(args, dict):
                    raise ValueError("tool-call arguments that are not an object")
                calls.append(
                    {
                        "type": "function",
                        "function": {"name": c["function"]["name"], "arguments": args},
                    }
                )
            row["tool_calls"] = calls
        out.append(row)
    if isinstance(tools, str):
        tools = json.loads(tools)
    tools = [json.loads(t) if isinstance(t, str) else t for t in tools] if tools else None
    return out, tools


def from_terminus(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """A Terminus-2 run (the task in its prompt template, JSON actions as assistant text, screens
    as user turns) as mini-swe-agent: the task alone in our template, each action one bash call
    with its keystroke lines joined, each screen its tool result."""
    first = messages[0]["content"]
    task = first.split("Task Description:\n", 1)[1].split("\n\nCurrent terminal state:", 1)[0]
    out: list[dict[str, Any]] = [SYSTEM, task_message(task)]
    for m in messages[1:]:
        if m["role"] == "assistant":
            action = json.loads(strip_thinking(m["content"]))
            text = "\n\n".join(x for x in (action.get("analysis"), action.get("plan")) if x)
            out.append(
                {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [bash_call(_terminus_command(action))],
                }
            )
        else:
            out.append(tool_result(m["content"].removeprefix("New Terminal Output:").lstrip("\n")))
    return out


def _terminus_command(action: dict[str, Any]) -> str:
    lines, wait = [], 0.0
    for c in action.get("commands") or []:
        if not isinstance(c, dict):  # a model's malformed action, kept as the screen shows it
            continue
        keys = str(c.get("keystrokes") or "")
        wait += float(c.get("duration") or 0)
        if keys.strip() and not keys.strip().startswith(_TMUX_KEYS):
            lines.append(keys.rstrip("\n"))
    return "\n".join(lines) or f"sleep {max(1, round(wait))}"


def from_atif(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A Harbor ATIF trajectory (claude-code, Terminus-2 or another tool-calling agent) as
    mini-swe-agent: the first user step is the task, each agent step's tool calls become bash
    calls (file tools as the bash a person would have typed), its observation their result."""
    task = next(s["message"] for s in steps if s.get("source") == "user")
    # a run of this harness already carries the rendered task and tool results
    user = {"role": "user", "content": task} if task.startswith(TASK_PREFIX) else task_message(task)
    out: list[dict[str, Any]] = [SYSTEM, user]
    for s in steps:
        if s.get("source") != "agent":
            continue
        calls = s.get("tool_calls") or []
        text = s.get("message") or ""
        if not calls:
            if text:
                out.append({"role": "assistant", "content": text})
            continue
        out.append(
            {
                "role": "assistant",
                "content": text,
                "tool_calls": [bash_call(_atif_command(c)) for c in calls],
            }
        )
        parts = _observation_parts(s)
        if any(_is_tool_result(p) for p in parts):  # this harness: one result per call
            if len(parts) > len(calls):
                parts = parts[: len(calls) - 1] + ["\n".join(parts[len(calls) - 1 :])]
            parts += [""] * (len(calls) - len(parts))
            out += [_tool_message(p) for p in parts]
            continue
        observation = "\n".join(parts)
        for _ in calls:
            out.append(_tool_message(observation))
            observation = ""  # one observation per step; later calls of the step show nothing
    return out


def _atif_command(call: dict[str, Any]) -> str:
    name, args = call.get("function_name") or "", call.get("arguments") or {}
    if not isinstance(args, dict):
        args = {"input": str(args)}
    if name in _SHELL_TOOLS:
        return str(args.get("command") or args.get("cmd") or args.get("input") or "")
    if name == "bash_command":  # Terminus-2 in ATIF: keystrokes, a bare wait as a sleep
        keys = str(args.get("keystrokes") or "").rstrip("\n")
        return keys or f"sleep {max(1, round(float(args.get('duration') or 1)))}"
    path = args.get("path") or args.get("file_path") or args.get("filename") or ""
    if name in ("read_file", "Read", "view_file"):
        return f"cat {path}"
    if name in ("Write", "write_file", "create_file"):
        return f"cat > {path} <<'EOF'\n{args.get('content') or args.get('contents') or ''}\nEOF"
    if name in ("search_replace", "Edit", "edit_file", "str_replace_editor"):
        old = args.get("old_string") or args.get("old_str") or ""
        new = args.get("new_string") or args.get("new_str") or ""
        return (
            "python3 - <<'EOF'\nimport pathlib\n"
            f"p = pathlib.Path({path!r}); s = p.read_text(); assert {old!r} in s\n"
            f"p.write_text(s.replace({old!r}, {new!r}, 1))\nEOF"
        )
    return f"# {name} {json.dumps(args, ensure_ascii=False)[:400]}"


def _is_tool_result(text: str) -> bool:
    """A tool result as this harness writes it, with or without a notice appended after the JSON
    (the output-limit and step-limit reminders)."""
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except (ValueError, TypeError):
        return False
    return (
        isinstance(parsed, dict)
        and "returncode" in parsed
        and bool({"output", "output_head"} & set(parsed))
    )


def _tool_message(observation: str) -> dict[str, str]:
    """The observation as a tool message; one that is a tool result already stays as it is."""
    if _is_tool_result(observation):
        return {"role": "tool", "content": observation}
    return tool_result(observation)


def _observation_parts(step: dict[str, Any]) -> list[str]:
    obs = step.get("observation")
    parts = (
        (obs.get("results") or obs.get("outputs") or [obs])
        if isinstance(obs, dict)
        else (obs if isinstance(obs, list) else [obs])
    )
    return [str(p.get("content") if isinstance(p, dict) else p) for p in parts if p]


def corpus_row(
    domain: str, dataset: str, key: str, language: str, messages: list[dict[str, Any]], tools: Any
) -> dict[str, str] | None:
    """The trajectory as a corpus row; None without a second assistant turn to cut before."""
    if sum(m["role"] == "assistant" for m in messages) < 2:
        return None
    conversation = {"messages": messages} | ({"tools": tools} if tools else {})
    return {
        "dataset": dataset,
        "split": "train",
        "entry_name": key,
        "domain": domain,
        "language": language,
        "user_prompt": next(m["content"] for m in messages if m["role"] == "user"),
        "conversation": json.dumps(conversation, ensure_ascii=False),
    }


def terminal_bench_science(root: Path, models: Iterable[str]) -> Iterator[dict[str, str]]:
    """Corpus rows from the public Terminal-Bench-Science leaderboard runs (ATIF, one jsonl.gz per
    model and task), for the model directories named; successes and failures alike."""
    for model in sorted(models):
        for f in sorted(
            glob.glob(str(root / "trajectories" / "models" / model / "*" / "*.jsonl.gz"))
        ):
            with gzip.open(f, "rt") as fh:
                lines = list(fh)
            for line in lines:
                r = json.loads(line)
                steps = (r.get("trajectory") or {}).get("steps") or []
                try:
                    messages = from_atif(steps)
                except (StopIteration, KeyError, TypeError, ValueError):
                    continue
                key = f"{model}/{r.get('trial_id')}"
                row = corpus_row(
                    "terminal", "terminal_bench_science", key, "tbsci", messages, BASH_TOOL
                )
                if row:
                    yield row
