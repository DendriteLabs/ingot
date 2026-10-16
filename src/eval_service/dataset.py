"""Load evaluation prompts and draw the same domain-balanced sample on every validator."""

from __future__ import annotations

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
    import pyarrow.parquet as pq

    # Keeping the lowest seeded hashes makes the result independent of shard and row order.
    keep = samples_per_domain * (OVERDRAW if fits else 1)
    reservoirs: dict[tuple[str, str], list[tuple[int, str, int, Sample]]] = defaultdict(list)
    serial = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        missing = set(_PARQUET_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing Parquet columns: {sorted(missing)}")
        for batch in parquet.iter_batches(columns=_PARQUET_COLUMNS, batch_size=16_384):
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

    by_domain: dict[str, dict[str, list[Sample]]] = defaultdict(dict)
    for (domain, language), reservoir in reservoirs.items():
        by_domain[domain][language] = [item[3] for item in reservoir]
    return _choose(by_domain, seed, samples_per_domain, fits)


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
