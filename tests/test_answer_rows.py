"""llama-states writes only the rows from `first`; load_states hands every role (vLLM files have
every row) the rows from `first`, and compare gives exactly the sums it gave on full files."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from eval_service import config, metrics  # noqa: E402
from eval_service.metrics import Sums  # noqa: E402

LAYERS, HIDDEN, VOCAB = len(config.LAYERS), 16, 50


class Head:
    def __init__(self, seed: int) -> None:
        self.w = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(seed))

    def logprobs(self, h: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(h.float() @ self.w.T, -1)


def reference(states, heads, answer, prompt_len, device):
    """compare as it was, on full [rows, layers, hidden] states."""
    first = prompt_len - 1
    out = {role: Sums(len(answer), len(answer) - 1) for role in states}
    teacher = states["teacher"]
    for c in range(0, len(answer), metrics.CHUNK):
        rows = slice(first + c, first + c + metrics.CHUNK)
        ref = heads["teacher"].logprobs(teacher[rows, -1].to(device))
        ref_p, ref_top = ref.exp(), ref.argmax(-1)
        for role, s in out.items():
            own = states[role][rows, -1].to(device)
            lp = ref if role == "teacher" else heads[role].logprobs(own)
            s.kl += (ref_p * (ref - lp)).sum().item()
            s.top1 += (lp.argmax(-1) == ref_top).sum().item()
    for c in range(0, len(answer) - 1, metrics.CHUNK):
        rows = slice(prompt_len + c, prompt_len + c + metrics.CHUNK)
        a = teacher[rows].to(device).float()
        a_len = a.norm(dim=-1)
        for role, s in out.items():
            b = a if role == "teacher" else states[role][rows].to(device).float()
            for acc, value in (
                (s.dist, (b - a).norm(dim=-1) / a_len),
                (s.cos_dist, 1 - (a * b).sum(-1) / (a_len * b.norm(dim=-1))),
            ):
                for j, v in enumerate(value.sum(0).tolist()):
                    acc[j] += v
    return out


def write(path: Path, states: torch.Tensor, read: list[int], first_row: int | None) -> str:
    tensors = {"token_ids": torch.tensor(read, dtype=torch.int64)}
    if first_row is None:  # vLLM: every row
        tensors["hidden_states"] = states.contiguous()
    else:  # llama-states: rows from first_row
        tensors["hidden_states"] = states[first_row:].contiguous()
        tensors["first_row"] = torch.tensor([first_row], dtype=torch.int64)
    save_file(tensors, path)
    path.with_name(path.name + ".lock").touch()
    return str(path)


@pytest.mark.parametrize("prompt_len,answer_len", [(7, 5), (40, 1100), (1, 2)])
def test_answer_rows_give_the_same_sums(tmp_path, monkeypatch, prompt_len, answer_len):
    monkeypatch.setattr(metrics, "CHUNK", 256)
    g = torch.Generator().manual_seed(prompt_len)
    read = list(range(prompt_len + answer_len - 1))
    answer = read[prompt_len:] + [0]
    full = {
        role: torch.randn(len(read), LAYERS, HIDDEN, generator=g)
        for role in ("teacher", "king", "challenger")
    }
    heads = {"teacher": Head(0), "king": Head(1), "challenger": Head(2)}
    first = prompt_len - 1
    loaded = {
        "teacher": metrics.load_states(
            write(tmp_path / "t.safetensors", full["teacher"], read, None), read, first
        ),
        **{
            role: metrics.load_states(
                write(tmp_path / f"{role}.safetensors", full[role], read, first), read, first
            )
            for role in ("king", "challenger")
        },
    }
    for role in full:
        assert torch.equal(loaded[role], full[role][first:]), "the same bytes, from row first"
    # the eval copies every state to its GPU before any math; fresh copies here, as CPU BLAS
    # picks its code path by the address
    fresh = {role: s.clone() for role, s in loaded.items()}
    assert metrics.compare(fresh, heads, answer, "cpu") == reference(  # ty: ignore[invalid-argument-type]
        full, heads, answer, prompt_len, "cpu"
    )


def test_a_file_from_past_first_is_rejected(tmp_path):
    read = list(range(10))
    states = torch.zeros(len(read), LAYERS, HIDDEN)
    path = write(tmp_path / "s.safetensors", states, read, 6)
    with pytest.raises(ValueError, match="from row 6"):
        metrics.load_states(path, read, 4)
