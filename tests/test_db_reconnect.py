"""A service outlives Postgres dropping its connection: the step that hits the drop fails, and the
next one gets a fresh connection instead of failing forever on the dead one."""

from __future__ import annotations

import psycopg
import pytest

from common import db


def drop_others(conn: db.Conn) -> None:
    """What a Postgres restart or a dropped tunnel does to every other connection."""
    conn.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity"
        " WHERE datname = current_database() AND pid <> pg_backend_pid()"
    )


def test_the_step_after_a_drop_gets_a_new_connection(conn, s):
    link = db.Link(url=s.database_url)
    first = link.conn()
    drop_others(conn)
    with pytest.raises(psycopg.OperationalError):
        first.execute("SELECT 1")  # the step that hits the drop fails
    second = link.conn()
    assert second is not first
    assert db.one(second, "SELECT 1 AS ok")["ok"] == 1


def test_check_finds_a_drop_no_query_has_hit_yet(conn, s):
    link = db.Link(url=s.database_url)
    first = link.conn()
    drop_others(conn)
    assert not first.closed, "nothing has noticed the drop yet"
    second = link.conn(check=True)
    assert second is not first
    assert db.one(second, "SELECT 1 AS ok")["ok"] == 1


def test_a_live_connection_is_kept(s):
    link = db.Link(url=s.database_url)
    assert link.conn() is link.conn() is link.conn(check=True)


def test_going_offline_with_the_database_down_does_not_raise(s):
    link = db.Link(url=s.database_url.replace(f":{s.postgres_port}/", ":1/"))
    db.offline_quietly(link, "nobody")
