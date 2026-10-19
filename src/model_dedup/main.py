"""Runs on the GPU box next to the evaluator: fingerprints every valid submission and keeps
copies of other miners' models, genesis included, away from the evaluation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Protocol

from loguru import logger as log

from common import cache, db, guard, models, states
from common.config import Settings, chain_spec, settings
from common.loop import forever
from model_dedup import config
from model_dedup.bank import AUDIT, BANK, Bank
from model_dedup.verdict import Verdict, decide

Fetch = Callable[[str, Path], Path]
Fingerprint = Callable[[Path], dict]
Upload = Callable[[str, bytes], str | None]
_PASSED = (
    states.UNIQUE,
    states.EVALUATING,
    states.WON,
    states.LOST,
    states.CROWNING,
    states.CROWNED,
)
POLICY = cache.Policy(
    newest=((frozenset({states.DUPLICATE}), config.KEEP_DUPLICATES),),
    after=dict.fromkeys(_PASSED, timedelta(seconds=config.KEEP_UNIQUE_S)),
)


class FingerprintBank(Protocol):
    def shortlist(self, fp: dict, *, hotkey: str, coldkey: str | None) -> dict[str, dict]: ...

    def put(
        self,
        bank_id: str,
        fp: dict,
        *,
        status: str,
        model_uri: str,
        hotkey: str | None = None,
        coldkey: str | None = None,
        verdict: Verdict | None = None,
    ) -> None: ...


def run_once(
    conn: db.Conn,
    *,
    worker: str,
    s: Settings,
    bank: FingerprintBank,
    fingerprint: Fingerprint,
    fetch: Fetch = models.fetch,
    upload: Upload = models.upload_artifact,
) -> bool:
    db.heartbeat(conn, worker, queued=states.VALID)
    cache.sweep(conn, s.model_cache_dir, POLICY)
    row = db.claim(
        conn, worker=worker, queued=states.VALID, running=states.DEDUPING, lease_s=config.LEASE_S
    )
    if row is None:
        return False
    dedup_id = db.one(
        conn, "INSERT INTO dedups (submission_id) VALUES (%s) RETURNING id", (row["id"],)
    )["id"]
    t0 = time.perf_counter()
    try:
        fp = fingerprint(fetch(row["model_uri"], s.model_cache_dir))
        shortlist = bank.shortlist(fp, hotkey=row["hotkey"], coldkey=row["coldkey"])
        verdict = decide(fp, shortlist)
        bank.put(
            str(row["id"]),
            fp,
            status=AUDIT if verdict.duplicate else BANK,
            model_uri=row["model_uri"],
            hotkey=row["hotkey"],
            coldkey=row["coldkey"],
            verdict=verdict,
        )
    except Exception as exc:
        log.exception("dedup {} crashed", dedup_id)
        conn.execute(
            "UPDATE dedups SET error = %s, finished_at = now() WHERE id = %s", (str(exc), dedup_id)
        )
        error = f"dedup crashed: {exc}"
        state = db.fail(
            conn, row, back_to=states.VALID, error=error, max_attempts=config.MAX_ATTEMPTS
        )
        if state == states.FAILED:
            doc = {
                "submission_id": str(row["id"]),
                "hotkey": row["hotkey"],
                "uid": row["uid"],
                "state": state,
                "reason": error,
                "barred": guard.is_used(conn, row["hotkey"]),
            }
            _put_dedup_fail(upload, row, doc)
        return True
    with conn.transaction():
        conn.execute(
            "UPDATE dedups SET duplicate = %s, reason = %s, duplicate_of = %s, corr = %s,"
            " copied_frac = %s, rel_to_teacher = %s, compared = %s, notes = %s, seconds = %s,"
            " finished_at = now() WHERE id = %s",
            (
                verdict.duplicate,
                verdict.reason,
                verdict.of,
                verdict.corr,
                verdict.copied_frac,
                fp["rel_to_teacher"],
                len(shortlist),
                "; ".join(verdict.notes) or None,
                time.perf_counter() - t0,
                dedup_id,
            ),
        )
        if verdict.duplicate:
            detail = {"submission_id": str(row["id"]), "duplicate_of": verdict.of}
            guard.mark_used(conn, row["hotkey"], "dedup", detail)
            db.finish(conn, row["id"], states.DUPLICATE, f"duplicate: {verdict.message}")
        else:
            db.finish(conn, row["id"], states.UNIQUE)
    if verdict.duplicate:
        _upload_dedup_fail(conn, upload, row, verdict)
    log.info(
        "dedup {}: {} (corr {:.3f}, copied {:.0%}, {} compared)",
        dedup_id,
        f"{verdict.reason} of {verdict.of}" if verdict.duplicate else "unique",
        verdict.corr,
        verdict.copied_frac,
        len(shortlist),
    )
    return True


def _upload_dedup_fail(conn: db.Conn, upload: Upload, row: db.Row, verdict: Verdict) -> None:
    source = conn.execute(
        "SELECT hotkey, uid FROM submissions WHERE id::text = %s", (verdict.of,)
    ).fetchone()
    doc = {
        "submission_id": str(row["id"]),
        "hotkey": row["hotkey"],
        "uid": row["uid"],
        "state": states.DUPLICATE,
        "reason": verdict.reason,
        "corr": verdict.corr,
        "copied_frac": verdict.copied_frac,
        "duplicate_of": {"submission_id": verdict.of, **source} if source else verdict.of,
        "barred": True,
    }
    _put_dedup_fail(upload, row, doc)


def _put_dedup_fail(upload: Upload, row: db.Row, doc: dict) -> None:
    upload(f"dedup/{row['hotkey']}/{row['id']}/dedup_fail.json", json.dumps(doc, indent=2).encode())


def main() -> None:
    from model_dedup.fingerprint import (  # torch: the dedup extra
        VERSION,
        fingerprint_dir,
        key_id,
        load_secret,
    )

    s, spec = settings(), chain_spec()
    secret = load_secret(s)
    bank = Bank.connect(s)
    if bank.genesis_scope() != (VERSION, key_id(secret)):
        raise SystemExit(
            "the dedup bank has no genesis fingerprint under this secret and version;"
            " run scripts/seed_dedup_bank.py"
        )
    teacher_dir = models.snapshot(spec.teacher_repo, spec.teacher_revision)
    fingerprint = partial(
        fingerprint_dir, teacher_dir=teacher_dir, secret=secret, device=s.dedup_device
    )
    link = db.Link()
    worker = db.register_worker(link.conn(), "model_dedup")
    log.info("model dedup started — bank {} on {}", s.opensearch_index, s.dedup_device)
    forever(
        lambda: run_once(link.conn(), worker=worker, s=s, bank=bank, fingerprint=fingerprint),
        idle_s=config.IDLE_S,
        on_stop=lambda: db.offline_quietly(link, worker),
    )
