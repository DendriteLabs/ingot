"""Runs on the GPU box against the shared Postgres. One eval at a time, none while a win is
being crowned, so every eval faces the newest king."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

from loguru import logger as log

from common import cache, db, guard, models, states
from common.config import Settings, chain_spec, settings
from common.loop import forever
from eval_service import config, dataset, run, servers
from eval_service.dataset import Dataset

Fetch = Callable[[str, Path], Path]
Upload = Callable[[str, bytes], str | None]
_ONE_AT_A_TIME = (
    "EXISTS (SELECT 1 FROM submissions WHERE state IN ('WON', 'CROWNING')"
    " OR (state = 'EVALUATING' AND lease_until > now()))"
)

# CROWNED goes at once too: the newest KEEP_KINGS kings are kept by sweep.
POLICY = cache.Policy(
    newest=((frozenset({states.FAILED}), config.KEEP_FAILED),),
    after=dict.fromkeys((states.LOST, states.CROWNED), timedelta(0)),
)


def run_once(
    conn: db.Conn,
    *,
    worker: str,
    teacher: str,
    teacher_dir: Path,
    data: Dataset,
    s: Settings,
    fetch: Fetch = models.fetch,
    scorer: run.Scorer = run.duel,
    upload: Upload = models.upload_artifact,
    fits: dataset.Fits | None = None,
) -> bool:
    db.heartbeat(conn, worker, queued=states.UNIQUE)
    cache.sweep(conn, s.model_cache_dir, POLICY, keep_kings=config.KEEP_KINGS)
    row = db.claim(
        conn,
        worker=worker,
        queued=states.UNIQUE,
        running=states.EVALUATING,
        lease_s=config.LEASE_S,
        unless=_ONE_AT_A_TIME,
    )
    if row is None:
        return False
    king = db.current_king(conn)
    seed = row["block_hash"] or str(row["id"])
    started = time.monotonic()
    drawn = data.draw(seed, config.SAMPLES_PER_DOMAIN, fits)
    drew_s = time.monotonic() - started
    eval_id = db.one(
        conn,
        "INSERT INTO evals (submission_id, king_id, teacher, dataset_hash, seed, sample_count,"
        " margin, dist_margin) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (
            row["id"],
            king["id"],
            teacher,
            data.sha256,
            seed,
            len(drawn),
            config.KL_MARGIN,
            config.DIST_MARGIN,
        ),
    )["id"]
    log.info(
        "eval {} of {}: {} samples drawn in {:.0f}s", eval_id, row["hotkey"], len(drawn), drew_s
    )
    try:
        started = time.monotonic()
        challenger_dir = fetch(row["model_uri"], s.model_cache_dir)
        between = time.monotonic()
        king_dir = fetch(king["model_uri"], s.model_cache_dir)
        fetched = time.monotonic()
        log.info(
            "eval {}: challenger fetched in {:.0f}s, king in {:.0f}s",
            eval_id,
            between - started,
            fetched - between,
        )
        duel = scorer(teacher_dir, challenger_dir, king_dir, drawn)
        log.info("eval {}: duel took {:.0f}s", eval_id, time.monotonic() - fetched)
    except Exception as exc:
        log.exception("eval {} crashed", eval_id)
        conn.execute(
            "UPDATE evals SET error = %s, finished_at = now() WHERE id = %s", (str(exc), eval_id)
        )
        _upload_summary(conn, upload, row["hotkey"], eval_id)
        db.fail(
            conn,
            row,
            back_to=states.UNIQUE,
            error=f"eval crashed: {exc}",
            max_attempts=config.MAX_ATTEMPTS,
        )
        return True
    scored = time.monotonic()
    results = s.eval_work_dir / "results" / f"{eval_id}.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    samples = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in duel.samples)
    results.write_text(samples)
    upload(f"evals/{row['hotkey']}/{eval_id}/samples.jsonl", samples.encode())
    with conn.transaction():
        conn.execute(
            "UPDATE evals SET score_challenger = %s, score_king = %s, dist_challenger = %s,"
            " dist_king = %s, noise_kl = %s, noise_dist = %s, won = %s, sample_count = %s,"
            " finished_at = now() WHERE id = %s",
            (
                duel.challenger.kl,
                duel.king.kl,
                duel.challenger.dist,
                duel.king.dist,
                duel.noise.kl,
                duel.noise.dist,
                duel.won,
                len(duel.samples),  # the ones scored: a failed sample is left out
                eval_id,
            ),
        )
        guard.mark_used(conn, row["hotkey"], "eval", {"submission_id": str(row["id"])})
        db.finish(conn, row["id"], states.WON if duel.won else states.LOST)
    _upload_summary(conn, upload, row["hotkey"], eval_id)
    log.info("eval {}: results stored and uploaded in {:.0f}s", eval_id, time.monotonic() - scored)
    log.info(
        "eval {} ({} of {} samples): KL challenger {:.5f} vs king {:.5f}, distance {:.4f} vs"
        " {:.4f} (noise floor {:.5f} / {:.4f}) → {}",
        eval_id,
        len(duel.samples),
        len(drawn),
        duel.challenger.kl,
        duel.king.kl,
        duel.challenger.dist,
        duel.king.dist,
        duel.noise.kl,
        duel.noise.dist,
        "WON" if duel.won else "LOST",
    )
    return True


def _upload_summary(conn: db.Conn, upload: Upload, hotkey: str, eval_id: int) -> None:
    row = db.one(conn, "SELECT * FROM evals WHERE id = %s", (eval_id,))
    doc = json.dumps(row, indent=2, default=str)
    upload(f"evals/{hotkey}/{eval_id}/summary.json", doc.encode())


def main() -> None:
    s, spec = settings(), chain_spec()
    conn = db.connect()
    teacher = f"{spec.teacher_repo}@{spec.teacher_revision}"
    teacher_dir = models.snapshot(spec.teacher_repo, spec.teacher_revision)
    data = dataset.fetch(spec, s.dataset_path)
    fits = run.fits(run.load_tokenizer(teacher_dir))  # the draw sizes prompts with it
    worker = db.register_worker(conn, "eval_service")
    log.info(
        "eval service started — teacher {} dataset {} ({} samples); {} orphaned state files swept",
        teacher,
        data.sha256[:12],
        data.row_count,
        servers.sweep(),
    )
    forever(
        lambda: run_once(
            conn,
            worker=worker,
            teacher=teacher,
            teacher_dir=teacher_dir,
            data=data,
            s=s,
            fits=fits,
        ),
        idle_s=config.IDLE_S,
        on_stop=lambda: stop(conn, worker),
    )


def stop(conn: db.Conn, worker: str) -> None:
    # the vLLM servers run in their own sessions, so pm2 stopping this process does not stop them
    servers.stop_all()
    db.offline(conn, worker)
