"""Runs on the box that hosts the teacher for an eval box without teacher GPUs: the teacher's vLLM
servers on INGOT_TEACHER_GPUS behind one localhost endpoint, which the eval box reaches through its
tunnel (INGOT_TEACHER_URL there). A request's hidden states wait in /dev/shm until the eval box
downloads them once; the download deletes them. It shows in the workers table as teacher_server,
through the central Postgres (ingot-teacher-db-tunnel)."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

import psycopg
from loguru import logger as log

from common import db, models
from common.config import chain_spec, settings
from eval_service import config, servers


class Teacher(Protocol):
    def complete(
        self,
        prompt: list[int],
        max_tokens: int,
        seed: int,
        force: list[int] | None = None,
        book: int | None = None,
        prefetch: bool = False,
    ) -> tuple[list[int], str]: ...

    def preemptions(self) -> float: ...

    def promote(self) -> None: ...


class Handler(BaseHTTPRequestHandler):
    teacher: Teacher
    states_dir: Path = servers.HIDDEN_STATES_DIR

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"ok": True})
        elif self.path == "/preemptions":
            self._json({"preemptions": self.teacher.preemptions()})
        elif self.path.startswith("/states/"):
            self._states(self.path.removeprefix("/states/"))
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/promote":
            self.teacher.promote()
            self._json({"ok": True})
            return
        if self.path != "/complete":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tokens, path = self.teacher.complete(
            body["prompt"],
            body["max_tokens"],
            body["seed"],
            body.get("force"),
            body.get("book"),
            bool(body.get("prefetch")),
        )
        keep_answer_rows(Path(path), len(body["prompt"]) - 1)
        self._json({"token_ids": tokens, "states": str(Path(path).relative_to(self.states_dir))})

    def _states(self, name: str) -> None:
        path = (self.states_dir / name).resolve()
        if not path.is_relative_to(self.states_dir.resolve()) or path.suffix != ".safetensors":
            self.send_error(404)
            return
        try:
            with open(f"{path}.lock") as lock:
                # the vLLM server holds this lock exclusively until the file is complete
                fcntl.flock(lock, fcntl.LOCK_SH)
            size = path.stat().st_size
        except FileNotFoundError:
            self.send_error(404)
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile, 1 << 20)
        finally:
            servers.discard(str(path))

    def _json(self, doc: dict[str, Any]) -> None:
        data = json.dumps(doc).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def keep_answer_rows(path: Path, first: int) -> None:
    """Rewrites a vLLM states file with only the hidden_states rows from `first`, in llama-states'
    layout (a `first_row` tensor), which the eval reads as is: the prompt rows are ~90% of the file
    and the comparison never reads them, and the file crosses the link to the eval box."""
    with open(f"{path}.lock") as lock:
        # the vLLM server holds this lock exclusively until the file is complete
        fcntl.flock(lock, fcntl.LOCK_SH)
    with path.open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
        data = 8 + size
        states, ids = header["hidden_states"], header["token_ids"]
        rows = states["shape"][0]
        row_bytes = (states["data_offsets"][1] - states["data_offsets"][0]) // rows
        kept = (rows - first) * row_bytes
        id_bytes = ids["data_offsets"][1] - ids["data_offsets"][0]
        out = {
            "hidden_states": {
                **states,
                "shape": [rows - first, *states["shape"][1:]],
                "data_offsets": [0, kept],
            },
            "token_ids": {**ids, "data_offsets": [kept, kept + id_bytes]},
            "first_row": {
                "dtype": "I64",
                "shape": [1],
                "data_offsets": [kept + id_bytes, kept + id_bytes + 8],
            },
        }
        head = json.dumps(out).encode()
        head += b" " * (-len(head) % 8)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as g:
            g.write(struct.pack("<Q", len(head)) + head)
            f.seek(data + states["data_offsets"][0] + first * row_bytes)
            _copy(f, g, kept)
            f.seek(data + ids["data_offsets"][0])
            _copy(f, g, id_bytes)
            g.write(struct.pack("<q", first))
    os.replace(tmp, path)


def _copy(src: Any, dst: Any, n: int) -> None:
    while n > 0:
        chunk = src.read(min(n, 1 << 20))
        if not chunk:
            raise EOFError("states file shorter than its header says")
        dst.write(chunk)
        n -= len(chunk)


def _sweep_abandoned() -> None:
    # an eval fetches a request's states right after its answer; one that died leaves them
    while True:
        time.sleep(600)
        if swept := servers.sweep(config.STATES_TTL_S):
            log.info("{} state files no eval fetched swept", swept)


def _heartbeat(link: db.Link, worker: str) -> None:
    # the database being down must not stop the teacher serving the eval box
    while True:
        time.sleep(config.TEACHER_HEARTBEAT_S)
        try:
            db.heartbeat(link.conn(), worker)
        except psycopg.Error as exc:
            log.warning("teacher heartbeat failed: {}", exc)


def main() -> None:
    s, spec = settings(), chain_spec()
    link = db.Link(db.connect())  # first, so a database that is down fails before the slow setup
    gpus = s.gpus("teacher")
    if not gpus:
        raise SystemExit("INGOT_TEACHER_GPUS is empty: the GPUs this box serves the teacher on")
    # registered before the servers load, as every service is: a start takes minutes
    worker = db.register_worker(link.conn(), "teacher_server")
    threading.Thread(target=_heartbeat, args=(link, worker), daemon=True).start()
    group = None
    try:
        log.info("{} orphaned state files swept", servers.sweep())
        teacher_dir = models.snapshot(spec.teacher_repo, spec.teacher_revision)
        group = servers.Group(teacher_dir, gpus, config.MEMORY_SHARE["teacher"])
        group.wait_ready(time.monotonic() + config.SERVER_START_S)
        Handler.teacher = group
        ThreadingHTTPServer.request_queue_size = 1024
        server = ThreadingHTTPServer(("127.0.0.1", s.teacher_port), Handler)
        log.info(
            "teacher {}@{} on GPUs {} at http://127.0.0.1:{} ({} free in /dev/shm)",
            spec.teacher_repo,
            spec.teacher_revision,
            gpus,
            s.teacher_port,
            shutil.disk_usage("/dev/shm").free,
        )
        threading.Thread(target=_sweep_abandoned, daemon=True).start()
        server.serve_forever()
    finally:
        if group:
            group.stop()
        db.offline_quietly(link, worker)
