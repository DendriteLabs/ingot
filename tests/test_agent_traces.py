"""Other harnesses' trajectories rewrite to mini-swe-agent's format exactly: the task alone in its
template, one bash call per action, the screen as its result, reasoning gone."""

from __future__ import annotations

import json

from common.agent_traces import (
    SYSTEM,
    TASK_PREFIX,
    TASK_SUFFIX,
    corpus_row,
    from_atif,
    from_mini_swe,
    from_terminus,
)

TERMINUS_PROMPT = (
    "You are an AI assistant...\n\nTask Description:\nFix the build.\n\n"
    "Current terminal state:\nroot@box:/app#"
)
ACTION = {
    "analysis": "Nothing built yet.",
    "plan": "Look around.",
    "commands": [
        {"keystrokes": "ls\n", "duration": 0.1},
        {"keystrokes": "C-c", "duration": 0.1},
        {"keystrokes": "make\n", "duration": 5.0},
    ],
    "task_complete": False,
}


def test_terminus_run_becomes_bash_calls_with_screens_as_results() -> None:
    messages = [
        {"role": "user", "content": TERMINUS_PROMPT},
        {"role": "assistant", "content": "<think>hmm</think>\n" + json.dumps(ACTION)},
        {"role": "user", "content": "New Terminal Output:\nroot@box:/app# ls\nMakefile\n"},
        {
            "role": "assistant",
            "content": json.dumps(
                {"analysis": "Done.", "plan": "", "commands": [], "task_complete": True}
            ),
        },
    ]
    out = from_terminus(messages)
    assert out[0] == SYSTEM
    assert out[1] == {"role": "user", "content": TASK_PREFIX + "Fix the build." + TASK_SUFFIX}
    assert out[2]["content"] == "Nothing built yet.\n\nLook around."
    assert out[2]["tool_calls"] == [
        {"type": "function", "function": {"name": "bash", "arguments": {"command": "ls\nmake"}}}
    ]
    assert json.loads(out[3]["content"]) == {
        "returncode": 0,
        "output": "root@box:/app# ls\nMakefile\n",
    }
    assert out[4]["tool_calls"][0]["function"]["arguments"] == {"command": "sleep 1"}, (
        "an empty action waits"
    )


def test_atif_file_tools_become_the_bash_a_person_would_type() -> None:
    steps = [
        {"source": "user", "message": "Fix the build."},
        {
            "source": "agent",
            "message": "Reading.",
            "tool_calls": [
                {"function_name": "Bash", "arguments": {"command": "ls"}},
                {"function_name": "read_file", "arguments": {"path": "/app/Makefile"}},
            ],
            "observation": {"results": [{"content": "Makefile\n"}]},
        },
        {
            "source": "agent",
            "message": "",
            "tool_calls": [
                {
                    "function_name": "bash_command",
                    "arguments": {"keystrokes": "make\n", "duration": 3},
                }
            ],
            "observation": {"results": [{"content": "ok"}]},
        },
    ]
    out = from_atif(steps)
    assert out[1]["content"].startswith(TASK_PREFIX + "Fix the build.")
    assert [c["function"]["arguments"]["command"] for c in out[2]["tool_calls"]] == [
        "ls",
        "cat /app/Makefile",
    ]
    assert (
        json.loads(out[3]["content"])["output"] == "Makefile\n"
        and json.loads(out[4]["content"])["output"] == ""
    )
    assert out[5]["tool_calls"][0]["function"]["arguments"] == {"command": "make"}


def test_mini_swe_rows_lose_reasoning_and_get_object_arguments() -> None:
    messages = [
        {"role": "user", "content": "Please solve this issue: x"},
        {
            "role": "assistant",
            "content": "<think>why</think>Looking.",
            "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}],
        },
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "Done."},
    ]
    out, tools = from_mini_swe(messages, ['{"type": "function", "function": {"name": "bash"}}'])
    assert out[1] == {
        "role": "assistant",
        "content": "Looking.",
        "tool_calls": [
            {"type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}
        ],
    }
    assert tools == [{"type": "function", "function": {"name": "bash"}}]
    row = corpus_row("swe", "d", "k", "python", out, tools)
    assert (
        row
        and row["user_prompt"] == "Please solve this issue: x"
        and "tools" in json.loads(row["conversation"])
    )
    assert corpus_row("swe", "d", "k", "python", out[:2], tools) is None, (
        "one assistant turn leaves nothing to cut"
    )
