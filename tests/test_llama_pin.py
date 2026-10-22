"""The llama.cpp fork the eval runs q2 models with is the submodule's commit."""

from __future__ import annotations

import subprocess

import pytest

from common.config import REPO_ROOT
from eval_service.config import LLAMA_CPP_COMMIT


def test_the_submodule_is_pinned_where_the_eval_says():
    tree = subprocess.run(
        ["git", "ls-files", "--stage", "llama-prism"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if not tree:
        pytest.skip("llama-prism is not checked in")
    assert tree[:2] == ["160000", LLAMA_CPP_COMMIT]
