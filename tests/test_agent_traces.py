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


def test_atif_of_this_harness_is_not_wrapped_twice() -> None:
    task = TASK_PREFIX + "fix it" + TASK_SUFFIX
    result = json.dumps({"returncode": 0, "output": "ok"}, indent=2)
    steps = [
        {"source": "system", "message": SYSTEM["content"]},
        {"source": "user", "message": task},
        {
            "source": "agent",
            "message": "looking",
            "tool_calls": [{"function_name": "bash", "arguments": {"command": "ls"}}],
            "observation": {"results": [{"content": result}]},
        },
        {
            "source": "agent",
            "message": "done",
            "tool_calls": [{"function_name": "bash", "arguments": {"command": "true"}}],
        },
    ]
    messages = from_atif(steps)
    assert messages[1]["content"] == task
    assert messages[3] == {"role": "tool", "content": result}


def test_atif_of_this_harness_keeps_one_result_per_call() -> None:
    task = TASK_PREFIX + "fix it" + TASK_SUFFIX
    a, b = (json.dumps({"returncode": 0, "output": o}, indent=2) for o in ("ok", "/app"))
    steps = [
        {"source": "user", "message": task},
        {
            "source": "agent",
            "message": "two at once",
            "tool_calls": [
                {"function_name": "bash", "arguments": {"command": "ls"}},
                {"function_name": "bash", "arguments": {"command": "pwd"}},
            ],
            "observation": {"results": [{"content": a}, {"content": b}]},
        },
        {
            "source": "agent",
            "message": "done",
            "tool_calls": [{"function_name": "bash", "arguments": {"command": "true"}}],
        },
    ]
    messages = from_atif(steps)
    assert [m["content"] for m in messages if m["role"] == "tool"][:2] == [a, b]


def test_a_tool_result_with_a_harness_notice_appended_stays_whole() -> None:
    result = json.dumps({"returncode": 0, "output": "ok"}, indent=2) + "\nRespond more concisely."
    steps = [
        {"source": "user", "message": TASK_PREFIX + "fix it" + TASK_SUFFIX},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [{"function_name": "bash", "arguments": {"command": "ls"}}],
            "observation": {"results": [{"content": result}]},
        },
        {
            "source": "agent",
            "message": "done",
            "tool_calls": [{"function_name": "bash", "arguments": {"command": "true"}}],
        },
    ]
    assert [m["content"] for m in from_atif(steps) if m["role"] == "tool"][0] == result


def test_normalize_gives_another_harness_run_this_harness_layout():
    from common.agent_traces import SUBMIT, SYSTEM, bash_call, normalize, task_message, tool_result

    wrapped = (
        "<user_info>\nOS Version: linux\n</user_info>\n<rules>be careful</rules>\n"
        "<user_query>\n<!-- canary GUID 1 -->\nFix the parser.\n</user_query>"
    )
    run = [
        SYSTEM,
        task_message(wrapped),
        {"role": "assistant", "content": "look", "tool_calls": [bash_call("C-c\nclear\nls -la\n")]},
        tool_result("New Terminal Output:\n\nfile.py"),
        {"role": "assistant", "content": "wait", "tool_calls": [bash_call("sleep 30")]},
        tool_result("Current Terminal Screen:\nbuild done"),
        {
            "role": "assistant",
            "content": "note",
            "tool_calls": [bash_call('# TaskCreate {"a": 1}')],
        },
        tool_result("task created"),
        {
            "role": "assistant",
            "content": "done",
            "tool_calls": [bash_call("# mark_task_complete {}")],
        },
    ]
    out = normalize(run)
    assert out is not None
    assert out[1] == task_message("Fix the parser.")
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant"]
    assert out[2]["tool_calls"] == [bash_call("ls -la")]
    assert json.loads(out[3]["content"])["output"] == "file.py\nbuild done"
    assert out[4]["tool_calls"] == [bash_call(SUBMIT)]
    assert normalize([SYSTEM, task_message("<environment_context>x</environment_context>")]) is None


def test_from_atif_takes_the_task_from_the_first_user_message_that_has_one():
    from common.agent_traces import from_atif, task_message

    steps = [
        {
            "source": "user",
            "message": "<environment_context><cwd>/root</cwd></environment_context>",
        },
        {"source": "user", "message": "Recover the spectrum."},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [
                {"function_name": "list_dir", "arguments": {"target_directory": "/root"}}
            ],
            "observation": "a",
        },
    ]
    out = from_atif(steps)
    assert out[1] == task_message("Recover the spectrum.")
    assert out[2]["tool_calls"][0]["function"]["arguments"]["command"] == "ls -la /root"
