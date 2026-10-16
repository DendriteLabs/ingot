"""Deletes downloaded private models a box no longer needs, judged by their submission state."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger as log

from common import db


@dataclass(frozen=True)
class Policy:
    newest: tuple[tuple[frozenset[str], int], ...] = ()
    after: dict[str, timedelta] = field(default_factory=dict)


def doomed(
    present: Iterable[str], rows: Iterable[db.Row], policy: Policy, keep: set[str], now: datetime
) -> set[str]:
    on_disk = set(present) - keep
    rows = [r for r in rows if r["model_hash"] in on_disk]
    out = set()
    for group, n in policy.newest:
        hit = sorted(
            (r for r in rows if r["state"] in group), key=lambda r: r["updated_at"], reverse=True
        )
        out |= {r["model_hash"] for r in hit[n:]}
    for r in rows:
        grace = policy.after.get(r["state"])
        if grace is not None and now - r["updated_at"] >= grace:
            out.add(r["model_hash"])
    return out


def sweep(conn: db.Conn, cache_dir: Path, policy: Policy, keep_kings: int = 0) -> None:
    present = [p.name.removesuffix(".verified") for p in cache_dir.glob("*.verified")]
    if not present:
        return
    rows = conn.execute(
        "SELECT model_hash, state, updated_at, now() AS now FROM submissions"
        " WHERE model_hash = ANY(%s)",
        (present,),
    ).fetchall()
    kings = conn.execute(
        "SELECT s.model_hash FROM kings k JOIN submissions s ON s.id = k.submission_id"
        " ORDER BY k.id DESC LIMIT %s",
        (keep_kings,),
    ).fetchall()
    keep = {k["model_hash"] for k in kings}
    now = rows[0]["now"] if rows else datetime.now().astimezone()
    for digest in doomed(present, rows, policy, keep, now):
        shutil.rmtree(cache_dir / digest, ignore_errors=True)
        (cache_dir / f"{digest}.verified").unlink(missing_ok=True)
        log.info("deleted cached model {}", digest)
