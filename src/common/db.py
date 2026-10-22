"""Postgres: one connection helper and the claim/finish primitives every service uses.

Raw SQL over psycopg in autocommit mode. A step that needs several statements to land
together wraps them in `with conn.transaction():`.
"""

from __future__ import annotations

import socket
from typing import Any, LiteralString

import psycopg
from loguru import logger as log
from psycopg.abc import Params
from psycopg.rows import dict_row

from common import states
from common.config import ChainSpec, settings

Row = dict[str, Any]
Conn = psycopg.Connection[Row]
# a worker silent this long is OFFLINE: one killed without its exit write (its box gone, or its
# tunnel cut first) would otherwise show ACTIVE for good
WORKER_STALE_S = 30 * 60


def connect(url: str | None = None) -> Conn:
    conn = psycopg.Connection[Row].connect(url or settings().database_url, autocommit=True)
    conn.row_factory = dict_row
    return conn


class Link:
    """A service's connection, replaced with a fresh one once Postgres has dropped it: a step
    that hits the drop fails, and the next one starts on a new connection."""

    def __init__(self, conn: Conn | None = None, url: str | None = None) -> None:
        self.url, self._conn = url, conn

    def conn(self, *, check: bool = False) -> Conn:
        """`check` pings the connection first, for a write after hours of work: a drop is only
        noticed on the next query, so an idle connection can look open when it is not."""
        if self._conn is not None and not self._conn.closed and check:
            try:
                self._conn.execute("SELECT 1")
            except psycopg.OperationalError:
                self._conn.close()
        if self._conn is None or self._conn.closed:
            if self._conn is not None:
                log.warning("database connection lost; reconnecting")
            self._conn = connect(self.url)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()


def offline_quietly(link: Link, worker: str) -> None:
    """Mark the worker OFFLINE on the way out; the database being down must not stop the exit."""
    try:
        offline(link.conn(), worker)
    except psycopg.Error as exc:
        log.warning("could not mark {} offline: {}", worker, exc)


def one(conn: Conn, query: LiteralString, params: Params | None = None) -> Row:
    """The row a query must return (RETURNING, aggregates); raises when it returns none."""
    row = conn.execute(query, params).fetchone()
    if row is None:
        raise RuntimeError(f"query returned no row: {query!r}")
    return row


def register_worker(conn: Conn, service: str) -> str:
    """Upsert the workers row at startup; OFFLINE comes back ACTIVE, DRAINING stays."""
    name = f"{service}@{socket.gethostname()}"
    conn.execute(
        """
        INSERT INTO workers (name, service) VALUES (%s, %s)
        ON CONFLICT (name) DO UPDATE
        SET state = CASE WHEN workers.state = 'OFFLINE' THEN 'ACTIVE' ELSE workers.state END,
            busy_with = NULL, started_at = now(), last_heartbeat_at = now()
        """,
        (name, service),
    )
    return name


def heartbeat(conn: Conn, worker: str, *, queued: str | None = None) -> None:
    """Touch the worker row; `queued` is the submission state this service drains. A row marked
    OFFLINE while it was silent (a long eval) comes back ACTIVE. Every other ACTIVE row silent for
    WORKER_STALE_S goes OFFLINE; a DRAINING one stays, so no heartbeat undoes a drain."""
    conn.execute(
        "UPDATE workers SET last_heartbeat_at = now(),"
        " state = CASE WHEN state = 'OFFLINE' THEN 'ACTIVE' ELSE state END,"
        " queued = (SELECT count(*) FROM submissions WHERE state = %s) WHERE name = %s",
        (queued, worker),
    )
    conn.execute(
        "UPDATE workers SET state = 'OFFLINE' WHERE state = 'ACTIVE'"
        " AND last_heartbeat_at < now() - make_interval(secs => %s)",
        (WORKER_STALE_S,),
    )


def offline(conn: Conn, worker: str) -> None:
    conn.execute(
        "UPDATE workers SET state = 'OFFLINE', busy_with = NULL WHERE name = %s", (worker,)
    )


def claim(
    conn: Conn,
    *,
    worker: str,
    queued: str,
    running: str,
    lease_s: int,
    unless: LiteralString = "FALSE",
) -> Row | None:
    """Move the oldest `queued` row, or a `running` row whose lease expired, to `running`.

    Only an ACTIVE worker claims. `unless` is a SQL condition that blocks claiming while true
    (the evaluator uses it to wait for the promoter). One statement, so two workers never
    claim the same row.
    """
    with conn.transaction():
        row = conn.execute(
            f"""
            UPDATE submissions
            SET state = %(running)s, attempts = attempts + 1, updated_at = now(),
                lease_until = now() + make_interval(secs => %(lease)s)
            WHERE id = (
                SELECT id FROM submissions
                WHERE (state = %(queued)s OR (state = %(running)s AND lease_until < now()))
                  AND NOT ({unless})
                  AND EXISTS (SELECT 1 FROM workers WHERE name = %(worker)s AND state = 'ACTIVE')
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED)
            RETURNING *
            """,
            {"running": running, "queued": queued, "lease": lease_s, "worker": worker},
        ).fetchone()
        if row is not None:
            conn.execute("UPDATE workers SET busy_with = %s WHERE name = %s", (row["id"], worker))
            event(conn, row["id"], running, worker=worker)
    return row


def finish(
    conn: Conn, submission_id: Any, state: str, error: str | None = None, *, retry: bool = False
) -> None:
    """Leave the running state. A stage that completes resets `attempts`: each stage gets its
    own max_attempts. A `retry` keeps the count."""
    with conn.transaction():
        conn.execute(
            "UPDATE submissions SET state = %s, error = %s, lease_until = NULL, updated_at = now(),"
            " attempts = CASE WHEN %s THEN attempts ELSE 0 END WHERE id = %s",
            (state, error, retry, submission_id),
        )
        conn.execute("UPDATE workers SET busy_with = NULL WHERE busy_with = %s", (submission_id,))
        event(conn, submission_id, state, error=error)


def event(
    conn: Conn,
    submission_id: Any,
    state: str,
    *,
    worker: str | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO events (submission_id, state, worker, error) VALUES (%s, %s, %s, %s)",
        (submission_id, state, worker, error),
    )


def fail(conn: Conn, row: Row, *, back_to: str, error: str, max_attempts: int) -> str:
    """Return a crashed row to its queue, or mark it FAILED once max_attempts is spent."""
    state = back_to if row["attempts"] < max_attempts else states.FAILED
    finish(conn, row["id"], state, error, retry=True)
    return state


def current_king(conn: Conn, family: str, quant: str) -> Row:
    """The newest king of a pair; each starts from its own genesis."""
    row = conn.execute(
        "SELECT * FROM kings WHERE family = %s AND quant = %s ORDER BY id DESC LIMIT 1",
        (family, quant),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"no {family}.{quant} king; run scripts/create_genesis.py --quant {quant}"
        )
    return row


# the queued states of the stages a closed pair may not go through, and their running states,
# released only once their lease has run out: a live worker still owns its row and finishes it
# (its next queued state is released then), else its finish would overwrite FAILED "closed"
_QUEUED = (states.SUBMITTED, states.VALID, states.UNIQUE)
_RUNNING = (states.VALIDATING, states.DEDUPING, states.EVALUATING)


def release_closed(conn: Conn, spec: ChainSpec) -> int:
    """Fail the waiting rows of a closed pair with error "closed" and give their registration its
    attempt back (FAILED, so the miner may activate the open pair): the miner did nothing wrong.
    Returns how many."""
    with conn.transaction():
        rows = conn.execute(
            "UPDATE submissions SET state = %s, error = 'closed', lease_until = NULL,"
            " updated_at = now(), attempts = 0"
            " WHERE (state = ANY(%s) OR (state = ANY(%s) AND lease_until < now()))"
            " AND (family, quant) <> (%s, %s) RETURNING id",
            (states.FAILED, list(_QUEUED), list(_RUNNING), spec.family, spec.quant),
        ).fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            conn.execute(
                "UPDATE private_registrations SET extra_attempts = extra_attempts + 1,"
                " state = 'FAILED', fault_message = 'closed', updated_at = now()"
                " WHERE submission_id = ANY(%s)",
                (ids,),
            )
            conn.execute("UPDATE workers SET busy_with = NULL WHERE busy_with = ANY(%s)", (ids,))
            for sid in ids:
                event(conn, sid, states.FAILED, error="closed")
    if ids:
        log.info("released {} submissions of closed quants", len(ids))
    return len(ids)
