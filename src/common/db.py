"""Postgres: one connection helper and the claim/finish primitives every service uses.

Raw SQL over psycopg in autocommit mode. A step that needs several statements to land
together wraps them in `with conn.transaction():`.
"""

from __future__ import annotations

import socket
from typing import Any, LiteralString

import psycopg
from psycopg.abc import Params
from psycopg.rows import dict_row

from common import states
from common.config import settings

Row = dict[str, Any]
Conn = psycopg.Connection[Row]


def connect(url: str | None = None) -> Conn:
    conn = psycopg.Connection[Row].connect(url or settings().database_url, autocommit=True)
    conn.row_factory = dict_row
    return conn


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
    """Touch the worker row; `queued` is the submission state this service drains."""
    conn.execute(
        "UPDATE workers SET last_heartbeat_at = now(),"
        " queued = (SELECT count(*) FROM submissions WHERE state = %s) WHERE name = %s",
        (queued, worker),
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


def current_king(conn: Conn) -> Row:
    row = conn.execute("SELECT * FROM kings ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        raise RuntimeError("no king in the database; run scripts/create_genesis.py")
    return row
