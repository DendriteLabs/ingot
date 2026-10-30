"""Load evaluation prompts and draw the same domain-balanced sample on every validator."""

from __future__ import annotations

import bisect
import hashlib
import heapq
import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from common import models
from common.config import ChainSpec
from eval_service import config

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
    tools: list[dict[str, Any]] | None = None  # an agent conversation's tool schemas


@dataclass(frozen=True)
class ParquetIndex:
    rows: dict[tuple[str, str], tuple[list[str], list[int], list[str]]]
    starts: list[int]  # the serial of each file's first row
    texts: list[Any]  # each file's user_prompt and conversation columns, a pyarrow Table

    def chat(self, serial: int) -> dict[str, Any]:
        """The row's Sample messages, and tools: its conversation where it has one."""
        f = bisect.bisect_right(self.starts, serial) - 1
        row = self.texts[f].slice(serial - self.starts[f], 1).to_pylist()[0]
        if row.get("conversation"):
            return json.loads(row["conversation"])
        return {"messages": [{"role": "user", "content": row["user_prompt"]}]}


@dataclass(frozen=True)
class Dataset:
    samples: list[Sample]
    sha256: str
    parquet_files: tuple[Path, ...] = ()
    parquet_rows: int = 0
    index: ParquetIndex | None = None

    @property
    def row_count(self) -> int:
        return self.parquet_rows or len(self.samples)

    def draw(self, seed: str, samples_per_domain: int, fits: Fits | None = None) -> list[Sample]:
        if self.index:
            return _draw_index(self.index, seed, samples_per_domain, fits)
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
            tools=row.get("tools"),
        )
        for row in rows
    ]
    if not samples:
        raise ValueError(f"no samples in {path}")
    return Dataset(samples=samples, sha256=hashlib.sha256(raw).hexdigest())


def fetch(spec: ChainSpec, local: Path) -> Dataset:
    """Download the configured Parquet split, plus the extra source's rows built here, or load
    local JSONL when no repo is configured."""
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
    identity = f"{spec.dataset_repo}@{root.name}:{spec.dataset_split}"
    if spec.dataset_extra_repo:
        extra, extra_identity = _extra(spec)
        files += (extra,)
        identity += f"+{extra_identity}"
    rows = _parquet_row_count(files)
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return Dataset(
        samples=[], sha256=digest, parquet_files=files, parquet_rows=rows, index=load_index(files)
    )


def _extra(spec: ChainSpec) -> tuple[Path, str]:
    """The extra source (chain.toml [dataset.extra]: runs that may be used but not redistributed)
    as one Parquet file of corpus rows, built once per revision in the HF cache, and its
    repo@revision for the dataset's identity."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from common.agent_traces import COLUMNS, terminal_bench_science
    from common.config import settings

    root = models.snapshot(
        spec.dataset_extra_repo,
        spec.dataset_extra_revision,
        repo_type="dataset",
        allow_patterns=[f"trajectories/models/{m}/*/*.jsonl.gz" for m in spec.dataset_extra_models],
    )
    out = (
        settings().eval_work_dir
        / "corpus-extra"
        / root.name
        / f"{spec.dataset_split}-extra.parquet"
    )
    if not out.exists():
        rows = list(terminal_bench_science(root, spec.dataset_extra_models))
        out.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([(c, pa.large_string()) for c in COLUMNS])
        partial = out.with_suffix(".tmp")
        pq.write_table(
            pa.Table.from_pylist(rows, schema), partial, compression="zstd", row_group_size=500
        )
        partial.replace(out)  # a crash mid-write must not leave a file that passes the exists check
    return out, f"{spec.dataset_extra_repo}@{root.name}"


def draw(
    samples: Sequence[Sample], seed: str, samples_per_domain: int, fits: Fits | None = None
) -> list[Sample]:
    """Draw each domain's count and spread it evenly over its languages."""
    tree: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        tree[sample.domain][sample.language].append(sample)
    return _choose(tree, seed, samples_per_domain, fits)


def _draw_parquet(
    files: tuple[Path, ...], seed: str, samples_per_domain: int, fits: Fits | None = None
) -> list[Sample]:
    return _draw_index(load_index(files), seed, samples_per_domain, fits)


def load_index(files: tuple[Path, ...]) -> ParquetIndex:
    """Reads the id columns and the prompts and conversations of every file into a ParquetIndex."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    rows: dict[tuple[str, str], tuple[list[str], list[int], list[str]]] = defaultdict(
        lambda: ([], [], [])
    )
    starts = []
    texts = []
    serial = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        missing = set(_PARQUET_COLUMNS) - set(names)
        if missing:
            raise ValueError(f"{path} is missing Parquet columns: {sorted(missing)}")
        starts.append(serial)
        # an agent row's conversation: JSON {"messages", "tools"}, a whole trajectory the draw cuts
        texts.append([c for c in ("user_prompt", "conversation") if c in names])
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
                key_ids, key_serials, key_sources = rows[domain, language]
                key_ids.append(sample_id)
                key_serials.append(serial)
                key_sources.append(source)
                serial += 1
    # the prompt columns: tens of GB, read in parallel (pyarrow lets the threads run)
    with ThreadPoolExecutor(config.DRAW_THREADS) as pool:
        tables = list(pool.map(lambda p, c: pq.read_table(p, columns=c), files, texts))
    return ParquetIndex(dict(rows), starts, tables)


def _draw_index(
    index: ParquetIndex, seed: str, samples_per_domain: int, fits: Fits | None
) -> list[Sample]:
    # Keeping the lowest seeded hashes makes the result independent of shard and row order.
    # Each key's rows are ranked in file order, as reading the files ranked them.
    by_domain: dict[str, dict[str, list[Sample]]] = defaultdict(dict)
    for (domain, language), (ids, serials, sources) in index.rows.items():
        keep = config.DOMAIN_SAMPLES.get(domain, samples_per_domain) * (OVERDRAW if fits else 1)
        reservoir: list[tuple[int, str, int, str]] = []
        for sample_id, serial, source in zip(ids, serials, sources, strict=True):
            rank = _rank(seed, sample_id)
            if len(reservoir) < keep:
                heapq.heappush(reservoir, (-rank, sample_id, serial, source))
            elif rank < -reservoir[0][0]:
                heapq.heapreplace(reservoir, (-rank, sample_id, serial, source))
        by_domain[domain][language] = [
            Sample(
                id=sample_id, domain=domain, source=source, language=language, **index.chat(serial)
            )
            for _, sample_id, serial, source in reservoir
        ]
    return _choose(by_domain, seed, samples_per_domain, fits)


def _choose(
    tree: dict[str, dict[str, list[Sample]]],
    seed: str,
    samples_per_domain: int,
    fits: Fits | None,
) -> list[Sample]:
    """`samples_per_domain` from each domain (DOMAIN_SAMPLES from those it names), spread evenly
    over its languages, each language's samples in seeded rank order and trajectories cut (_cut);
    what `fits` rejects is passed over, within OVERDRAW times the draw."""
    chosen: list[Sample] = []
    # fits renders the long candidates through the tokenizer, which lets other threads run
    pool = ThreadPoolExecutor(config.DRAW_THREADS)
    for domain, languages in tree.items():
        quota = config.DOMAIN_SAMPLES.get(domain, samples_per_domain)
        keep = quota * OVERDRAW
        ranked: dict[str, list[Sample]] = {}
        for language, items in languages.items():
            ordered = sorted(items, key=lambda sample: (_rank(seed, sample.id), sample.id))[:keep]
            cuts = pool.map(lambda sample: _cut(sample, seed, fits), ordered)
            ranked[language] = [sample for sample in cuts if sample is not None]
        counts = _equal_shares(
            {language: len(items) for language, items in ranked.items()},
            quota,
            f"{seed}/{domain}",
        )
        for language, count in counts.items():
            chosen.extend(ranked[language][:count])
    pool.shutdown()
    return sorted(chosen, key=lambda sample: sample.id)


def _cut(sample: Sample, seed: str, fits: Fits | None) -> Sample | None:
    """A trajectory (a sample with assistant turns) cut before a seeded random assistant turn after
    the first, uniform over those whose prompt fits (the later of two picks for LATE_CUT_SOURCES);
    any other sample as it is, if it fits."""
    turns = [i for i, m in enumerate(sample.messages) if m["role"] == "assistant"][1:]
    if not turns:
        return sample if fits is None or fits(sample) else None
    rng = random.Random(f"{seed}/{sample.id}")
    late = sample.source in config.LATE_CUT_SOURCES
    while turns:
        # the prompt only grows turn by turn, so a turn that does not fit rules out every later
        # one; picking again among the earlier ones keeps the pick uniform over the ones that fit
        turn = max(rng.choice(turns), rng.choice(turns)) if late else rng.choice(turns)
        cut = replace(sample, messages=sample.messages[:turn])
        if fits is None or fits(cut):
            return cut
        turns = turns[: turns.index(turn)]
    return None


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
