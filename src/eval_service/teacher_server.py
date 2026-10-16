"""Runs on the box that hosts the teacher for an eval box without teacher GPUs: the teacher's vLLM
servers on INGOT_TEACHER_GPUS behind one localhost endpoint, which the eval box reaches through its
tunnel (INGOT_TEACHER_URL there). A request's hidden states wait in /dev/shm until the eval box
downloads them once; the download deletes them."""

from __future__ import annotations

import fcntl
import json
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from loguru import logger as log

from common import models
from common.config import chain_spec, settings
from eval_service import config, servers


class Teacher(Protocol):
    def complete(
        self, prompt: list[int], max_tokens: int, seed: int, force: list[int] | None = None
    ) -> tuple[list[int], str]: ...

    def preemptions(self) -> float: ...


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
        if self.path != "/complete":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tokens, path = self.teacher.complete(
            body["prompt"], body["max_tokens"], body["seed"], body.get("force")
        )
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


def _sweep_abandoned() -> None:
    # an eval fetches a request's states right after its answer; one that died leaves them
    while True:
        time.sleep(600)
        if swept := servers.sweep(config.STATES_TTL_S):
            log.info("{} state files no eval fetched swept", swept)


def main() -> None:
    s, spec = settings(), chain_spec()
    gpus = s.gpus("teacher")
    if not gpus:
        raise SystemExit("INGOT_TEACHER_GPUS is empty: the GPUs this box serves the teacher on")
    log.info("{} orphaned state files swept", servers.sweep())
    teacher_dir = models.snapshot(spec.teacher_repo, spec.teacher_revision)
    group = servers.Group(teacher_dir, gpus, config.MEMORY_SHARE["teacher"])
    group.wait_ready(time.monotonic() + config.SERVER_START_S)
    Handler.teacher = group
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
    try:
        server.serve_forever()
    finally:
        group.stop()
