"""Runs on the GPU box against the shared Postgres. One eval at a time, none while a win is
being crowned, so every eval faces the newest king. The next eval is claimed, drawn and partly
answered by the teacher while the current one drains (run.Prefetch): the teacher's servers would
otherwise idle through the current eval's last long answers."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger as log

from common import cache, db, guard, models, states
from common.config import Settings, chain_spec, settings
from common.loop import forever
from eval_service import config, dataset, run, servers
from eval_service.dataset import Dataset, Sample

Fetch = Callable[[str, Path], Path]
Upload = Callable[[str, bytes], str | None]
Prefetcher = Callable[[Path, list[Sample]], Any]  # run.Prefetch: result() is a run.Prefetched
_CROWNING = "EXISTS (SELECT 1 FROM submissions WHERE state IN ('WON', 'CROWNING'))"
_ONE_AT_A_TIME = (
    f"{_CROWNING} OR EXISTS"
    " (SELECT 1 FROM submissions WHERE state = 'EVALUATING' AND lease_until > now())"
)


@dataclass
class _Pending:
    """The next eval, claimed while the current one drained: drawn, and the teacher answering its
    first PREFETCH_SAMPLES samples."""

    row: db.Row
    drawn: list[Sample]
    prefetch: Any


# one eval at a time: the one claimed ahead waits here for run_once's next call
_pending: _Pending | None = None
_claiming: threading.Thread | None = None


def run_once(
    link: db.Link,
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
    prefetcher: Prefetcher = run.Prefetch,
) -> bool:
    global _pending, _claiming
    conn = link.conn()
    db.heartbeat(conn, worker, queued=states.UNIQUE)
    cache.sweep(conn, s.model_cache_dir)
    if _claiming is not None:
        _claiming.join()
        _claiming = None
    prefetched: run.Prefetched = {}
    if _pending is None:
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
        started = time.monotonic()
        drawn = data.draw(_seed(row), config.SAMPLES_PER_DOMAIN, fits)
        drew = f"drawn in {time.monotonic() - started:.0f}s"
    else:
        pending, _pending = _pending, None
        row, drawn = pending.row, pending.drawn
        # the eval before this one may have won: its coronation makes the king this one faces
        _wait_for_crowning(conn)
        prefetched = pending.prefetch.result()
        drew = f"drawn ahead, {len(prefetched)} answered ahead"
    king = db.current_king(conn, row["family"], row["quant"])
    eval_id = db.one(
        conn,
        "INSERT INTO evals (submission_id, king_id, teacher, dataset_hash, seed, sample_count)"
        " VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (row["id"], king["id"], teacher, data.sha256, _seed(row), len(drawn)),
    )["id"]
    log.info("eval {} of {}: {} samples {}", eval_id, row["hotkey"], len(drawn), drew)
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
        duel = scorer(
            teacher_dir,
            challenger_dir,
            king_dir,
            drawn,
            on_progress=_progress(link, eval_id),
            prefetched=prefetched,
            on_teacher_drained=lambda: _claim_ahead(worker, data, fits, teacher_dir, s, prefetcher),
        )
        log.info("eval {}: duel took {:.0f}s", eval_id, time.monotonic() - fetched)
    except Exception as exc:
        # answers taken ahead that no sample used (the duel pops the ones it did)
        for _, path in prefetched.values():
            servers.discard(path)
        # the duel ran for hours: the connection it started with may be gone by now
        conn = link.conn(check=True)
        if isinstance(exc, servers.ServerExited) and exc.gpu in s.gpus("challenger"):
            _does_not_load(conn, upload, row, eval_id, f"the model does not run: {exc.reason}")
            return True
        log.exception("eval {} crashed", eval_id)
        if isinstance(exc, servers.ServerExited) and exc.gpu in s.gpus("king"):
            # the king's model failed, not this challenger: back in line, no attempt spent
            conn.execute(
                "UPDATE evals SET error = %s, finished_at = now() WHERE id = %s",
                (f"the king's server failed: {exc.reason}", eval_id),
            )
            db.finish(conn, row["id"], states.UNIQUE, "the king's server failed; queued again")
            return True
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
    finally:
        # the teacher stays loaded; the king and challenger free their GPUs until the next eval
        servers.release("king", "challenger")
    scored = time.monotonic()
    conn = link.conn(check=True)
    results = s.eval_work_dir / "results" / f"{eval_id}.jsonl"
    results.parent.mkdir(parents=True, exist_ok=True)
    samples = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in duel.samples)
    results.write_text(samples)
    upload(f"evals/{row['hotkey']}/{eval_id}/samples.jsonl", samples.encode())
    with conn.transaction():
        conn.execute(
            "UPDATE evals SET score_challenger = %s, score_king = %s, dist_challenger = %s,"
            " dist_king = %s, noise_kl = %s, noise_dist = %s, won = %s, sample_count = %s,"
            " margin = %s, dist_margin = %s, finished_at = now() WHERE id = %s",
            (
                duel.challenger.kl,
                duel.king.kl,
                duel.challenger.dist,
                duel.king.dist,
                duel.noise.kl,
                duel.noise.dist,
                duel.won,
                len(duel.samples),  # the ones scored: a failed sample is left out
                # the leads this eval needed, a share of this king's scores
                run.needed(duel.king).kl,
                run.needed(duel.king).dist,
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


def _seed(row: db.Row) -> str:
    return row["block_hash"] or str(row["id"])


def _wait_for_crowning(conn: db.Conn) -> None:
    """The claim of an eval not taken ahead waits for a coronation (_ONE_AT_A_TIME); one taken
    ahead waits here, so it too faces the king being crowned."""
    while db.one(conn, f"SELECT {_CROWNING} AS crowning")["crowning"]:
        time.sleep(config.IDLE_S)


def _claim_ahead(
    worker: str,
    data: Dataset,
    fits: dataset.Fits | None,
    teacher_dir: Path,
    s: Settings,
    prefetcher: Prefetcher,
) -> None:
    """Claims the next eval and starts its draw and prefetch, on a thread: the duel's own threads
    call this, and a draw takes minutes. What it claimed waits in _pending for run_once."""
    global _claiming

    def work() -> None:
        global _pending
        # its own connection: the duel's progress reports hold the link's
        conn = db.connect(s.database_url)
        row = None
        try:
            row = db.claim(
                conn,
                worker=worker,
                queued=states.UNIQUE,
                running=states.EVALUATING,
                lease_s=config.LEASE_S,
                unless=_CROWNING,
            )
            if row is None:
                return
            drawn = data.draw(_seed(row), config.SAMPLES_PER_DOMAIN, fits)
            _pending = _Pending(row, drawn, prefetcher(teacher_dir, drawn))
            log.info(
                "eval of {} claimed ahead: {} samples drawn, the teacher answering {} of them",
                row["hotkey"],
                len(drawn),
                min(len(drawn), config.PREFETCH_SAMPLES),
            )
        except Exception:
            log.exception("claiming the next eval ahead failed")
            if row is not None:
                db.finish(conn, row["id"], states.UNIQUE, "claimed ahead and dropped; queued again")
        finally:
            conn.close()

    _claiming = threading.Thread(target=work, daemon=True)
    _claiming.start()


def _progress(link: db.Link, eval_id: int) -> run.Progress:
    """Writes samples_done for the dashboard; the scorer calls it from its worker threads."""
    lock = threading.Lock()

    def report(done: int) -> None:
        with lock:
            link.conn().execute("UPDATE evals SET samples_done = %s WHERE id = %s", (done, eval_id))

    return report


def _does_not_load(conn: db.Conn, upload: Upload, row: db.Row, eval_id: int, reason: str) -> None:
    """The challenger's servers exit while loading it: the model's fault, so it is rejected like
    an invalid one (its upload attempt goes back to the miner) instead of retried as a crash."""
    log.warning("eval {}: {}", eval_id, reason)
    conn.execute(
        "UPDATE evals SET error = %s, finished_at = now() WHERE id = %s", (reason, eval_id)
    )
    _upload_summary(conn, upload, row["hotkey"], eval_id)
    db.finish(conn, row["id"], states.INVALID, reason)


def _upload_summary(conn: db.Conn, upload: Upload, hotkey: str, eval_id: int) -> None:
    row = db.one(conn, "SELECT * FROM evals WHERE id = %s", (eval_id,))
    doc = json.dumps(row, indent=2, default=str)
    upload(f"evals/{hotkey}/{eval_id}/summary.json", doc.encode())


def main() -> None:
    s, spec = settings(), chain_spec()
    link = db.Link(db.connect())  # first, so a database that is down fails before the slow setup
    teacher = f"{spec.teacher_repo}@{spec.teacher_revision}"
    teacher_dir = models.snapshot(spec.teacher_repo, spec.teacher_revision)
    started = time.monotonic()
    data = dataset.fetch(spec, s.dataset_path)  # the whole corpus into memory, once
    log.info("dataset {} read in {:.0f}s", data.sha256[:12], time.monotonic() - started)
    fits = run.fits(run.load_tokenizer(teacher_dir))  # the draw sizes prompts with it
    if not s.teacher_url:
        # a teacher on this box's GPUs loads now, not with the first eval; every eval keeps it
        servers.serve({"teacher": teacher_dir})
    worker = db.register_worker(link.conn(), "eval_service")
    log.info(
        "eval service started — teacher {} dataset {} ({} samples); {} orphaned state files swept",
        teacher,
        data.sha256[:12],
        data.row_count,
        servers.sweep(),
    )

    def tick() -> bool:
        db.release_closed(link.conn(), spec)
        return run_once(
            link,
            worker=worker,
            teacher=teacher,
            teacher_dir=teacher_dir,
            data=data,
            s=s,
            fits=fits,
        )

    forever(
        tick,
        idle_s=config.IDLE_S,
        on_stop=lambda: stop(link, worker),
    )


def stop(link: db.Link, worker: str) -> None:
    # the vLLM servers run in their own sessions, so pm2 stopping this process does not stop them
    servers.stop_all()
    db.offline_quietly(link, worker)
