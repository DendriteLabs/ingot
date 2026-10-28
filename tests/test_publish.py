"""A crowned king is published with its own quant's genesis files; a gguf king as one file named
after its repo, with a card that says how to run it."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from common import models
from eval_service.config import LLAMA_CPP_COMMIT
from promoter.main import Step, card


class FakeApi:
    commits: list[list[str]] = []

    def __init__(self, token=None) -> None:
        pass

    def create_repo(self, **_kw) -> None:
        pass

    def create_commit(self, *, operations, **_kw) -> None:
        FakeApi.commits.append(sorted(op.path_in_repo for op in operations))


def _fake_hub(monkeypatch, tmp_path: Path) -> list[tuple[str, str]]:
    fetched: list[tuple[str, str]] = []

    def hub_file(repo: str, name: str, revision: str | None = None) -> Path:
        fetched.append((repo, name))
        (tmp_path / name).write_text("x")
        return tmp_path / name

    hub = types.SimpleNamespace(
        HfApi=FakeApi,
        CommitOperationAdd=lambda path_in_repo, path_or_fileobj: types.SimpleNamespace(
            path_in_repo=path_in_repo
        ),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(models, "hub_file", hub_file)
    FakeApi.commits = []
    return fetched


def test_a_gguf_king_is_its_file_and_its_quants_license(tmp_path, monkeypatch, spec):
    fetched = _fake_hub(monkeypatch, tmp_path)
    model = tmp_path / "upload"
    model.mkdir()
    (model / "anything.gguf").write_bytes(b"GGUF")
    q2 = spec.pair("qwen38_27b", "q2")
    models.publish(model, "ns/Ingot-I-27B-Q2", "card", q2)
    assert FakeApi.commits == [["Ingot-I-27B-PQ2_0.gguf", "LICENSE", "NOTICE.txt", "README.md"]]
    assert {repo for repo, _ in fetched} == {q2.genesis_repo}, "q2's genesis, not the open one's"


def test_the_card(spec):
    """A gguf king's card pins the fork and links the vision projector, and shows its winning
    duel only: no gain over genesis chained along the reign."""
    row = {"hotkey": "hk", "uid": 1, "id": "sub"}
    text = card(row, "Ingot-I-27B-Q2", spec.pair("qwen38_27b", "q2"), [])
    assert LLAMA_CPP_COMMIT in text and "-m Ingot-I-27B-PQ2_0.gguf" in text
    assert "--mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf" in text
    assert (
        "hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
        in text
    )
    assert "Running it" not in card(row, "Ingot-I-27B-FP8", spec.pair("qwen38_27b", "fp8"), [])

    # king I beat genesis 0.08 → 0.072 (10%), king II beat king I 0.072 → 0.0648 (10% more)
    steps = [Step(0.072, 0.08, 0.45, 0.5), Step(0.0648, 0.072, 0.405, 0.45)]
    text = card(row, "Ingot-II-27B-Q2", spec.pair("qwen38_27b", "q2"), steps)
    assert "| this model, in its winning eval | 0.06480 | 0.4050 |" in text
    assert "| the king it beat, same eval | 0.07200 | 0.4500 |" in text
    assert "closer than genesis" not in text and "19.0%" not in text


def test_a_q2_card_credits_bonsai_2(spec):
    """Genesis copies Prism's Bonsai 2: the card lists it as the base, and says the codes are
    Bonsai 2's only when none changed."""
    row = {"hotkey": "hk", "uid": 1, "id": "sub"}
    q2 = spec.pair("qwen38_27b", "q2")
    kept = card(row, "Ingot-I-27B-Q2", q2, [], base_model="Qwen/Qwen3.8-27B", codes=(0, 1280))
    assert kept.startswith("---\nbase_model: prism-ml/Ternary-Bonsai-2-27B-gguf\n")
    assert "refined: the same ternary codes" in kept
    assert "Credit belongs to the miner, and to Prism ML for Bonsai 2." in kept
    changed = card(row, "Ingot-I-27B-Q2", q2, [], base_model="Qwen/Qwen3.8-27B", codes=(5, 1000))
    assert "the miner changed 0.50% of its ternary codes" in changed
    assert "same ternary codes" not in changed
