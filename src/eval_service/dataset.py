"""Load evaluation prompts and draw the same domain-balanced sample on every validator."""

from __future__ import annotations

import bisect
import hashlib
import heapq
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import models
from common.config import ChainSpec

_PARQUET_COLUMNS = (
    "dataset",
    "split",
    "entry_name",
    "domain",
    "language",
    "user_prompt",
)
_ID_COLUMNS = ("dataset", "split", "entry_name", "domain", "language")
Fits = Callable[["Sample"], bool]
# candidates kept per language, as a multiple of the draw, for the ones `fits` passes over
OVERDRAW = 4


@dataclass(frozen=True)
class Sample:
    id: str
    messages: list[dict[str, Any]]
    domain: str = ""
    source: str = ""
    language: str = ""


@dataclass(frozen=True)
class Dataset:
    samples: list[Sample]
    sha256: str
    parquet_files: tuple[Path, ...] = ()
    parquet_rows: int = 0

    @property
    def row_count(self) -> int:
        return self.parquet_rows or len(self.samples)

    def draw(self, seed: str, samples_per_domain: int, fits: Fits | None = None) -> list[Sample]:
        if self.parquet_files:
            return _draw_parquet(self.parquet_files, seed, samples_per_domain, fits)
        return draw(self.samples, seed, samples_per_domain, fits)


def load(path: Path) -> Dataset:
    """Load local JSONL used for development and smoke tests."""
    raw = path.read_bytes()
    # split on "\n" only: str.splitlines() also breaks at Unicode separators left in prompts
    rows = [json.loads(line) for line in raw.decode().split("\n") if line.strip()]
    samples = [
        Sample(
            id=row["id"],
            messages=row["messages"],
            domain=row.get("domain", ""),
            source=row.get("source", ""),
            language=row.get("language", ""),
        )
        for row in rows
    ]
    if not samples:
        raise ValueError(f"no samples in {path}")
    return Dataset(samples=samples, sha256=hashlib.sha256(raw).hexdigest())


def fetch(spec: ChainSpec, local: Path) -> Dataset:
    """Download the configured Parquet split, or load local JSONL when no repo is configured."""
    if not spec.dataset_repo:
        return load(local)
    pattern = f"data/{spec.dataset_split}-*.parquet"
    root = models.snapshot(
        spec.dataset_repo,
        spec.dataset_revision,
        repo_type="dataset",
        allow_patterns=pattern,
    )
    files = tuple(sorted(root.glob(pattern)))
    if not files:
        raise ValueError(f"{spec.dataset_repo}@{spec.dataset_revision} has no {pattern}")
    rows = _parquet_row_count(files)
    digest = hashlib.sha256(
        f"{spec.dataset_repo}@{root.name}:{spec.dataset_split}".encode()
    ).hexdigest()
    return Dataset(samples=[], sha256=digest, parquet_files=files, parquet_rows=rows)


def draw(
    samples: Sequence[Sample], seed: str, samples_per_domain: int, fits: Fits | None = None
) -> list[Sample]:
    """Draw an equal count per domain and spread each count evenly over its languages."""
    tree: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        tree[sample.domain][sample.language].append(sample)
    return _choose(tree, seed, samples_per_domain, fits)


def _draw_parquet(
    files: tuple[Path, ...], seed: str, samples_per_domain: int, fits: Fits | None = None
) -> list[Sample]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    # Keeping the lowest seeded hashes makes the result independent of shard and row order.
    # The ranking pass reads the id columns only; the prompts of the kept rows are read after.
    keep = samples_per_domain * (OVERDRAW if fits else 1)
    reservoirs: dict[tuple[str, str], list[tuple[int, str, int, str]]] = defaultdict(list)
    starts = []  # the serial of each file's first row
    serial = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        missing = set(_PARQUET_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing Parquet columns: {sorted(missing)}")
        starts.append(serial)
        for batch in parquet.iter_batches(columns=_ID_COLUMNS, batch_size=16_384):
            # one string type for the join: shards hold string or large_string
            parts = [
                batch[c].cast(pa.large_string())
                for c in ("dataset", "split", "entry_name", "language")
            ]
            if any(part.null_count for part in parts):
                raise ValueError(f"{path} has a row without dataset, split, entry_name or language")
            sep = pa.scalar("/", pa.large_string())
            ids = pc.binary_join_element_wise(*parts, sep).to_pylist()  # ty: ignore[unresolved-attribute]
            columns = (batch["dataset"].to_pylist(), batch["domain"].to_pylist())
            for sample_id, source, domain, language in zip(
                ids, *columns, batch["language"].to_pylist(), strict=True
            ):
                rank = _rank(seed, sample_id)
                reservoir = reservoirs[domain, language]
                if len(reservoir) < keep:
                    heapq.heappush(reservoir, (-rank, sample_id, serial, source))
                elif rank < -reservoir[0][0]:
                    heapq.heapreplace(reservoir, (-rank, sample_id, serial, source))
                serial += 1

    prompts = _prompts(files, starts, {item[2] for r in reservoirs.values() for item in r})
    by_domain: dict[str, dict[str, list[Sample]]] = defaultdict(dict)
    for (domain, language), reservoir in reservoirs.items():
        by_domain[domain][language] = [
            Sample(
                id=sample_id,
                messages=[{"role": "user", "content": prompts[row]}],
                domain=domain,
                source=source,
                language=language,
            )
            for _, sample_id, row, source in reservoir
        ]
    return _choose(by_domain, seed, samples_per_domain, fits)


def _prompts(files: tuple[Path, ...], starts: list[int], serials: set[int]) -> dict[int, str]:
    """user_prompt of the given rows (serials over all files in order), reading only the row
    groups that hold them."""
    import pyarrow.parquet as pq

    wanted: dict[int, list[int]] = defaultdict(list)
    for serial in serials:
        wanted[bisect.bisect_right(starts, serial) - 1].append(serial)
    out: dict[int, str] = {}
    for f, rows in wanted.items():
        parquet = pq.ParquetFile(files[f])
        groups = [0]
        for g in range(parquet.metadata.num_row_groups):
            groups.append(groups[-1] + parquet.metadata.row_group(g).num_rows)
        by_group: dict[int, list[int]] = defaultdict(list)
        for serial in rows:
            by_group[bisect.bisect_right(groups, serial - starts[f]) - 1].append(serial)
        for g, serials_in in by_group.items():
            column = parquet.read_row_group(g, columns=["user_prompt"])["user_prompt"]
            for serial in serials_in:
                out[serial] = column[serial - starts[f] - groups[g]].as_py()
    return out


def _choose(
    tree: dict[str, dict[str, list[Sample]]],
    seed: str,
    samples_per_domain: int,
    fits: Fits | None,
) -> list[Sample]:
    """An equal count per domain, spread evenly over its languages, each language's samples in
    seeded rank order; what `fits` rejects is passed over, within OVERDRAW times the draw."""
    keep = samples_per_domain * OVERDRAW
    chosen: list[Sample] = []
    for domain, languages in tree.items():
        ranked: dict[str, list[Sample]] = {}
        for language, items in languages.items():
            ordered = sorted(items, key=lambda sample: (_rank(seed, sample.id), sample.id))
            ranked[language] = [s for s in ordered[:keep] if fits is None or fits(s)]
        counts = _equal_shares(
            {language: len(items) for language, items in ranked.items()},
            samples_per_domain,
            f"{seed}/{domain}",
        )
        for language, count in counts.items():
            chosen.extend(ranked[language][:count])
    return sorted(chosen, key=lambda sample: sample.id)


def _equal_shares(sizes: dict[str, int], n: int, seed: str) -> dict[str, int]:
    """Split n evenly, using the seed for remainders and redistributing exhausted strata."""
    counts = dict.fromkeys(sizes, 0)
    order = sorted(
        (name for name, size in sizes.items() if size), key=lambda name: _rank(seed, name)
    )
    left = min(n, sum(sizes.values()))
    while left:
        for name in order:
            if left and counts[name] < sizes[name]:
                counts[name] += 1
                left -= 1
    return counts


def _parquet_row_count(files: tuple[Path, ...]) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def _rank(seed: str, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\x00{name}".encode()).digest(), "big")
