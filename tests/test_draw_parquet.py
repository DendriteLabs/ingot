"""The parquet draw ranks on the id columns and reads only the kept rows' prompts; it must return
exactly what the full-row draw did, duplicates and row groups included."""

from __future__ import annotations

import heapq
import random
from collections import defaultdict
from pathlib import Path

import pytest

from eval_service import dataset
from eval_service.dataset import OVERDRAW, Sample, _choose, _rank

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def reference(files, seed, samples_per_domain, fits=None):
    """The draw as it was: every row through to_pylist, a Sample per kept row."""
    keep = samples_per_domain * (OVERDRAW if fits else 1)
    reservoirs = defaultdict(list)
    serial = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16_384):
            for row in batch.to_pylist():
                domain, language = row["domain"], row["language"]
                sample_id = "/".join((row["dataset"], row["split"], row["entry_name"], language))
                rank = _rank(seed, sample_id)
                reservoir = reservoirs[domain, language]
                if len(reservoir) < keep or rank < -reservoir[0][0]:
                    sample = Sample(
                        id=sample_id,
                        messages=[{"role": "user", "content": row["user_prompt"]}],
                        domain=domain,
                        source=row["dataset"],
                        language=language,
                    )
                    item = (-rank, sample_id, serial, sample)
                    if len(reservoir) < keep:
                        heapq.heappush(reservoir, item)
                    else:
                        heapq.heapreplace(reservoir, item)
                serial += 1
    by_domain = defaultdict(dict)
    for (domain, language), reservoir in reservoirs.items():
        by_domain[domain][language] = [item[3] for item in reservoir]
    return _choose(by_domain, seed, samples_per_domain, fits)


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, ...]:
    rng = random.Random(7)
    out = []
    for f in range(4):
        rows = []
        for i in range(rng.randint(150, 400)):
            # a few ids repeat with different prompts, within and across files
            entry = f"e{rng.randint(0, 60)}" if i % 9 == 0 else f"f{f}-e{i}"
            rows.append({
                "dataset": rng.choice(["org/a", "b"]),
                "split": "train",
                "entry_name": entry,
                "domain": rng.choice(["code", "math", "chat"]),
                "language": rng.choice(["en", "de", "pl"]),
                "user_prompt": f"prompt {f}/{i} " + "x" * rng.randint(0, 40),
            })  # fmt: skip
        path = tmp_path / f"train-{f:05d}.parquet"
        table = pa.Table.from_pylist(rows)
        if f % 2:  # the corpus's shards hold large_string
            table = table.cast(pa.schema([(n, pa.large_string()) for n in table.column_names]))
        pq.write_table(table, path, row_group_size=37)
        out.append(path)
    return tuple(out)


@pytest.mark.parametrize("seed", ["0xabc", "0x1", "block", "7"])
@pytest.mark.parametrize("per_domain", [1, 5, 40])
def test_the_draw_is_the_full_row_draw(files, seed, per_domain):
    def fits(s: Sample) -> bool:
        return len(s.messages[0]["content"]) % 3 != 0

    assert dataset._draw_parquet(files, seed, per_domain) == reference(files, seed, per_domain)
    assert dataset._draw_parquet(files, seed, per_domain, fits) == reference(
        files, seed, per_domain, fits
    )
