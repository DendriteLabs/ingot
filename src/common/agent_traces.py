"""Agent trajectories from other harnesses as the eval reads them: mini-swe-agent's bash tool, the
format every source is rewritten to. Terminus-2 (JSON actions in text) and ATIF (Harbor's
trajectory format) rewrite here; the eval's draw cuts a row before an assistant turn and scores the
teacher's answer to the prefix."""

from __future__ import annotations

import glob
import gzip
import json
import re
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
# the eval caches the rows it rewrites under this: bump it when the rewriting changes
FORMAT = 2
# the corpus's floor for terminal runs, applied to the rows joined at draw time too
TERMINAL_MIN_TURNS = 10
# Terminus-2 keystrokes that are tmux keys, not shell text
_TMUX_KEYS = ("C-", "M-", "Escape", "Enter", "Tab", "Up", "Down", "Left", "Right")
_SHELL_TOOLS = {"Bash", "bash", "run_terminal_command", "execute_command", "shell", "terminal"}
_CONTEXT_BLOCKS = re.compile(
    r"<(user_info|rules|environment_context|system-reminder|system_reminder)>.*?</\1>", re.S
)
_SCREEN_HEAD = re.compile(
    r"^(Previous response had (warnings|parsing errors):.*?"
    r"(?=New Terminal Output:|Current Terminal Screen:|$))?"
    r"(New Terminal Output:|Current Terminal Screen:)?\n*",
    re.S,
)
_WAIT = re.compile(r"^\s*sleep\s+[\d.]+\s*$")
_NOT_SHELL = re.compile(r"^# (?!mark_task_complete)\w+ [\[{\"]")
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def task_text(text: str) -> str:
    """The task alone, without any harness's template, context blocks or canary comments."""
    text = text.removeprefix(TASK_PREFIX).removesuffix(TASK_SUFFIX)
    # Terminus-2 puts the task inside its own instructions
    if "Task Description:\n" in text and "keystrokes" in text:
        text = text.split("Task Description:\n", 1)[1].split("\n\nCurrent terminal state:", 1)[0]
    query = re.search(r"<user_query>(.*?)</user_query>", text, re.S)
    text = query.group(1) if query else _CONTEXT_BLOCKS.sub("", text)
    return re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()


def normalize(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Another harness's run in bash calls as this harness records it; None under two steps."""
    task = task_text(next(m["content"] for m in messages if m["role"] == "user"))
    if not task:
        return None
    out: list[dict[str, Any]] = [SYSTEM, task_message(task)]
    i = next(k for k, m in enumerate(messages) if m["role"] == "user") + 1
    while i < len(messages):
        m = messages[i]
        i += 1
        if m["role"] != "assistant":
            continue
        results = []
        while i < len(messages) and messages[i]["role"] == "tool":
            results.append(messages[i])
            i += 1
        calls, kept_results, waited = [], [], []
        for k, call in enumerate(m.get("tool_calls") or []):
            command = str(call["function"]["arguments"].get("command", ""))
            command = "\n".join(
                line
                for line in command.split("\n")
                if not line.strip().startswith(_TMUX_KEYS) and line.strip() != "clear"
            ).strip("\n")
            output = _result_output(results[k]) if k < len(results) else None
            if command.startswith("# mark_task_complete"):
                command = SUBMIT
            if _NOT_SHELL.match(command):
                continue
            if not command or _WAIT.match(command):
                if output:
                    waited.append(output)
                continue
            calls.append(bash_call(command))
            if output is not None:
                kept_results.append(tool_result(output, _result_code(results[k])))
        if not calls:
            if waited and out[-1]["role"] == "tool":
                out[-1] = tool_result(
                    "\n".join([_result_output(out[-1]), *waited]), _result_code(out[-1])
                )
            continue
        if waited and kept_results:
            kept_results[-1] = tool_result(
                "\n".join([_result_output(kept_results[-1]), *waited]),
                _result_code(kept_results[-1]),
            )
        out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        out += kept_results
    return out if sum(m["role"] == "assistant" for m in out) >= 2 else None


def _turns(messages: list[dict[str, Any]]) -> int:
    return sum(m["role"] == "assistant" for m in messages)


def _result_output(message: dict[str, Any]) -> str:
    try:
        parsed = json.loads(message["content"])
        text = str(parsed.get("output", parsed.get("output_head", "")))
    except (ValueError, AttributeError):
        text = str(message["content"])
    return _SCREEN_HEAD.sub("", text, count=1)


def _result_code(message: dict[str, Any]) -> int:
    try:
        return int(json.loads(message["content"]).get("returncode", 0))
    except (ValueError, AttributeError, TypeError):
        return 0


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
    first = str(next(s["message"] for s in steps if s.get("source") == "user"))
    # a run of this harness has the rendered task; others open with context blocks, so the task
    # is the first user message that has one
    if first.startswith(TASK_PREFIX):
        user = {"role": "user", "content": first}
    else:
        texts = (task_text(str(s["message"])) for s in steps if s.get("source") == "user")
        user = task_message(next(t for t in texts if t))
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
    path = (
        args.get("path")
        or args.get("file_path")
        or args.get("filename")
        or args.get("target_file")
        or args.get("target_directory")
        or ""
    )
    # waiting on a background job has no shell form; a short sleep stands in for it
    if name in (
        "TaskOutput",
        "wait",
        "get_command_or_subagent_output",
    ):
        return "sleep 30"
    if name in ("read_file", "Read", "view_file"):
        return f"cat {path}"
    if name == "list_dir":
        return f"ls -la {path}"
    if name in ("grep", "Grep"):
        return f"grep -rnE {json.dumps(str(args.get('pattern') or ''))} {path or '.'}"
    if name in ("Write", "write", "write_file", "create_file"):
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
                    messages = normalize(from_atif(steps))
                except (StopIteration, KeyError, TypeError, ValueError):
                    continue
                if messages is None or _turns(messages) < TERMINAL_MIN_TURNS:
                    continue
                key = f"{model}/{r.get('trial_id')}"
                row = corpus_row("terminal", "term_science", key, "science", messages, BASH_TOOL)
                if row:
                    yield row
