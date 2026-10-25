"""The eval draw is deterministic, equal per domain and broad across available languages."""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import pytest

from eval_service import config
from eval_service.dataset import Dataset, Sample, draw, load


def write(path: Path, n: int) -> None:
    messages = [{"role": "user", "content": "text"}]
    lines = [json.dumps({"id": f"s{i:03d}", "messages": messages}) for i in range(n)]
    path.write_text("\n".join(lines) + "\n")


def test_same_seed_same_samples_different_seed_different(tmp_path: Path):
    write(tmp_path / "eval.jsonl", 100)
    data = load(tmp_path / "eval.jsonl")
    a, b = data.draw("0xabc", 8), data.draw("0xabc", 8)
    assert (
        a == b and len(a) == 8 and [sample.id for sample in a] == sorted(sample.id for sample in a)
    )
    assert data.draw("0xdef", 8) != a
    assert len(data.draw("0xabc", 1000)) == 100, "never more than the dataset holds"


def test_dataset_hash_tracks_the_file(tmp_path: Path):
    write(tmp_path / "a.jsonl", 3)
    write(tmp_path / "b.jsonl", 4)
    assert load(tmp_path / "a.jsonl").sha256 != load(tmp_path / "b.jsonl").sha256
    assert load(tmp_path / "a.jsonl").sha256 == load(tmp_path / "a.jsonl").sha256


def test_prompts_with_unicode_line_separators_load_whole(tmp_path: Path):
    texts = ["a b", "c\u0085d", "e\x0cf", "g h"]
    lines = [
        json.dumps(
            {"id": f"s{i}", "messages": [{"role": "user", "content": text}]},
            ensure_ascii=False,
        )
        for i, text in enumerate(texts)
    ]
    (tmp_path / "eval.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    loaded = load(tmp_path / "eval.jsonl").samples
    assert [sample.messages[0]["content"] for sample in loaded] == texts


def corpus(domains: int, languages: int, rows: int) -> list[Sample]:
    return [
        Sample(
            id=f"d{domain}/l{language}/s{i}",
            messages=[{"role": "user", "content": f"prompt {i}"}],
            domain=f"d{domain}",
            language=f"l{language}",
        )
        for domain in range(domains)
        for language in range(languages)
        for i in range(rows)
    ]


def test_draws_the_same_count_per_domain_and_language():
    chosen = draw(corpus(domains=3, languages=4, rows=100), "0xabc", 60)
    assert collections.Counter(sample.domain for sample in chosen) == {
        "d0": 60,
        "d1": 60,
        "d2": 60,
    }
    per_language = collections.Counter((sample.domain, sample.language) for sample in chosen)
    assert set(per_language.values()) == {15}


def test_seed_randomizes_languages_when_there_are_more_than_the_quota():
    samples = corpus(domains=1, languages=100, rows=2)
    a, b = draw(samples, "0xabc", 60), draw(samples, "0xdef", 60)
    languages_a = {sample.language for sample in a}
    languages_b = {sample.language for sample in b}
    assert len(languages_a) == len(languages_b) == 60
    assert languages_a != languages_b


def test_small_languages_are_exhausted_and_their_slots_redistributed():
    samples = [Sample(f"small/{i}", [], domain="d", language="small") for i in range(2)] + [
        Sample(f"big/{i}", [], domain="d", language="big") for i in range(100)
    ]
    chosen = draw(samples, "seed", 10)
    assert collections.Counter(sample.language for sample in chosen) == {"small": 2, "big": 8}


def test_prompts_the_filter_rejects_are_passed_over():
    samples = corpus(domains=2, languages=2, rows=100)
    long = {sample.id for sample in samples if int(sample.id.rsplit("s", 1)[1]) % 3 == 0}
    chosen = draw(samples, "0xabc", 20, fits=lambda sample: sample.id not in long)
    assert collections.Counter(sample.domain for sample in chosen) == {"d0": 20, "d1": 20}
    assert not long & {sample.id for sample in chosen}
    assert draw(samples, "0xabc", 20, fits=lambda _sample: True) == draw(samples, "0xabc", 20)


def test_draw_does_not_depend_on_input_order():
    samples = corpus(domains=3, languages=5, rows=20)
    shuffled = samples[:]
    random.Random(1).shuffle(shuffled)
    assert draw(samples, "0xabc", 40) == draw(shuffled, "0xabc", 40)


def test_parquet_shards_are_sampled_and_mapped_to_chat_prompts(tmp_path: Path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = [
        {
            "dataset": "source",
            "split": "train",
            "entry_name": f"task-{i}",
            "domain": domain,
            "language": language,
            "user_prompt": f"prompt {i}",
        }
        for i, (domain, language) in enumerate(
            [("code", "en"), ("code", "pl"), ("math", "en"), ("math", "pl")] * 5
        )
    ]
    first, second = tmp_path / "chungus-0.parquet", tmp_path / "chungus-1.parquet"
    pq.write_table(pa.Table.from_pylist(rows[:10]), first)
    pq.write_table(pa.Table.from_pylist(rows[10:]), second)
    data = Dataset([], "hash", (first, second), len(rows))

    chosen = data.draw("block-hash", 4)
    reversed_shards = Dataset([], "hash", (second, first), len(rows))

    assert collections.Counter(sample.domain for sample in chosen) == {"code": 4, "math": 4}
    assert reversed_shards.draw("block-hash", 4) == chosen
    assert all(sample.source == "source" for sample in chosen)
    assert all(sample.messages[0]["role"] == "user" for sample in chosen)
    assert all(sample.messages[0]["content"].startswith("prompt ") for sample in chosen)

    filtered = data.draw("block-hash", 4, fits=lambda sample: "task-1" not in sample.id)
    assert collections.Counter(sample.domain for sample in filtered) == {"code": 4, "math": 4}
    assert all("task-1" not in sample.id for sample in filtered), "rejected rows are replaced"


def test_agent_rows_draw_their_own_count_as_conversations(tmp_path: Path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    monkeypatch.setattr(config, "DOMAIN_SAMPLES", {"swe": 6})
    call = {"type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}
    turn = {
        "messages": [
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": "src tests"},
        ],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }
    rows = [
        {
            "dataset": "source",
            "split": "train",
            "entry_name": f"task-{i}",
            "domain": domain,
            "language": "en",
            "user_prompt": f"prompt {i}",
            "conversation": json.dumps(turn) if domain == "swe" else None,
        }
        for i, domain in enumerate(["swe", "math"] * 10)
    ]
    path = tmp_path / "chungus-0.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    chosen = Dataset([], "hash", (path,), len(rows)).draw("block-hash", 2)

    assert collections.Counter(sample.domain for sample in chosen) == {"swe": 6, "math": 2}
    swe = [sample for sample in chosen if sample.domain == "swe"]
    assert all(s.messages == turn["messages"] and s.tools == turn["tools"] for s in swe)
    math = [sample for sample in chosen if sample.domain == "math"]
    assert all(s.tools is None and s.messages[0]["content"].startswith("prompt ") for s in math)
