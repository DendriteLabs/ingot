from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from loguru import logger as log

from common import cache, db, guard, models, states
from common.config import ChainSpec, Settings, chain_spec, settings
from common.loop import forever
from model_validation import config, pipeline
from model_validation.checks import check_files
from model_validation.families import Family

Fetch = Callable[[str, Path], Path]
ListFiles = Callable[[str], list[str]]
Check = Callable[..., str | None]
Load = Callable[[str], Family]
Upload = Callable[[str, bytes], str | None]


def run_once(
    conn: db.Conn,
    *,
    worker: str,
    s: Settings,
    spec: ChainSpec,
    fetch: Fetch = models.fetch,
    list_files: ListFiles = models.list_files,
    check: Check = pipeline.validate,
    load: Load = pipeline.load,
    upload: Upload = models.upload_artifact,
) -> bool:
    db.heartbeat(conn, worker, queued=states.SUBMITTED)
    cache.sweep(conn, s.model_cache_dir)
    row = db.claim(
        conn,
        worker=worker,
        queued=states.SUBMITTED,
        running=states.VALIDATING,
        lease_s=config.LEASE_S,
    )
    if row is None:
        return False
    try:
        family = load(row["family"])
        pair = spec.pair(row["family"], row["quant"])
        reason = check_files(list_files(row["model_uri"]), spec, pair.format) or check(
            fetch(row["model_uri"], s.model_cache_dir), spec=spec, family=family, pair=pair
        )
    except Exception as exc:
        log.exception("validation of {} crashed", row["id"])
        error = f"validation crashed: {exc}"
        state = db.fail(
            conn, row, back_to=states.SUBMITTED, error=error, max_attempts=config.MAX_ATTEMPTS
        )
        if state == states.FAILED:
            _upload_fails(upload, row, state, error, guard.is_used(conn, row["hotkey"]))
        return True
    db.finish(conn, row["id"], states.INVALID if reason else states.VALID, reason)
    if reason:
        barred = _bar_after_max_invalid(conn, row["hotkey"])
        _upload_fails(upload, row, states.INVALID, reason, barred)
    log.info("submission {} {}", row["id"], reason or "valid")
    return True


def _bar_after_max_invalid(conn: db.Conn, hotkey: str) -> bool:
    invalid = db.one(
        conn,
        "SELECT count(*) AS n FROM submissions WHERE hotkey = %s AND state = %s",
        (hotkey, states.INVALID),
    )["n"]
    if invalid >= config.MAX_INVALID:
        guard.mark_used(conn, hotkey, "invalid", {"invalid": invalid})
        log.info("hotkey {} barred after {} invalid submissions", hotkey, invalid)
    return guard.is_used(conn, hotkey)


def _upload_fails(upload: Upload, row: db.Row, state: str, reason: str, barred: bool) -> None:
    doc = {
        "submission_id": str(row["id"]),
        "hotkey": row["hotkey"],
        "uid": row["uid"],
        "family": row["family"],
        "quant": row["quant"],
        "model_hash": row["model_hash"],
        "state": state,
        "reason": reason,
        "barred": barred,
    }
    upload(f"validation/{row['hotkey']}/{row['id']}/fails.json", json.dumps(doc, indent=2).encode())


def main() -> None:
    s, spec = settings(), chain_spec()
    link = db.Link()
    for family, _quant in spec.quants:
        pipeline.load(family)
    worker = db.register_worker(link.conn(), "model_validation")
    log.info("model validation started — pairs {}", sorted(spec.quants))

    def tick() -> bool:
        db.release_closed(link.conn(), spec)
        return run_once(link.conn(), worker=worker, s=s, spec=spec)

    forever(
        tick,
        idle_s=config.IDLE_S,
        on_stop=lambda: db.offline_quietly(link, worker),
    )
