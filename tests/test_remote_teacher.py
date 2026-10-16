"""The teacher served by another box: answers over HTTP, hidden states downloaded once, the serving
box's copy deleted; no GPU, a fake teacher behind the real endpoint."""

from __future__ import annotations

import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from common.config import Settings, settings
from eval_service import servers, teacher_server


class FakeTeacher:
    """Writes each request's 'states' into the serving box's states dir, as vLLM does."""

    def __init__(self, states_dir: Path) -> None:
        self.states_dir, self.n = states_dir, 0

    def complete(self, prompt, max_tokens, seed, force=None):
        self.n += 1
        path = self.states_dir / "gpu0" / f"req{self.n}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(prompt) * 1000)
        Path(str(path) + ".lock").touch()
        return (force[len(prompt) :] if force else [7] * max_tokens), str(path)

    def preemptions(self) -> float:
        return 0.0


@pytest.fixture
def remote(tmp_path, monkeypatch):
    serving, local = tmp_path / "serving", tmp_path / "local"
    serving.mkdir()
    monkeypatch.setattr(servers, "HIDDEN_STATES_DIR", local)
    handler = type(
        "H", (teacher_server.Handler,), {"teacher": FakeTeacher(serving), "states_dir": serving}
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", serving, local
    server.shutdown()


def test_an_answer_and_its_states_come_over_once(remote, tmp_path):
    url, serving, local = remote
    group = servers.RemoteGroup(url, tmp_path)
    assert group.alive() and group.preemptions() == 0.0

    tokens, path = group.complete([1, 2, 3], 4, seed=0)
    assert tokens == [7, 7, 7, 7]
    assert Path(path).parent == local / "remote"
    assert Path(path).read_bytes() == bytes([1, 2, 3]) * 1000
    # the serving box deletes its copy after sending the last byte, so a moment after the download
    deadline = time.monotonic() + 5
    while list(serving.rglob("*.safetensors*")) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not list(serving.rglob("*.safetensors*")), "the serving box keeps no copy"

    forced, _ = group.complete([1, 2], 3, seed=1, force=[1, 2, 5, 6, 7])
    assert forced == [5, 6, 7]

    servers.discard(path)
    assert not Path(path).exists() and not Path(path + ".lock").exists()


def test_only_state_files_inside_the_states_dir_are_served(remote):
    url, _, _ = remote
    for name in ("../../etc/passwd", "gpu0/req1.lock", "missing.safetensors"):
        assert httpx.get(f"{url}/states/{name}").status_code == 404


def test_a_teacher_url_takes_the_teacher_off_this_box(remote, tmp_path, monkeypatch):
    url, _, _ = remote
    monkeypatch.setenv("INGOT_TEACHER_URL", url)
    settings.cache_clear()
    try:
        groups = servers.serve({"teacher": tmp_path})
        assert isinstance(groups["teacher"], servers.RemoteGroup)
    finally:
        servers._groups.clear()
        settings.cache_clear()


def test_gpus_come_from_env(monkeypatch):
    monkeypatch.setenv("INGOT_KING_GPUS", "0,1,2,3")
    monkeypatch.setenv("INGOT_CHALLENGER_GPUS", "4, 5,6,7")
    monkeypatch.setenv("INGOT_TEACHER_GPUS", "")
    s = Settings()
    assert (s.gpus("king"), s.gpus("challenger"), s.gpus("teacher")) == (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (),
    )


def test_gpus_default_to_four_per_king_and_challenger(monkeypatch):
    for name in ("INGOT_TEACHER_GPUS", "INGOT_KING_GPUS", "INGOT_CHALLENGER_GPUS"):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert (s.gpus("teacher"), s.gpus("king"), s.gpus("challenger")) == (
        (),
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    )


def test_a_local_teacher_needs_gpus(tmp_path, monkeypatch):
    for name in ("INGOT_TEACHER_GPUS", "INGOT_TEACHER_URL"):
        monkeypatch.setenv(name, "")
    settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="INGOT_TEACHER_GPUS or INGOT_TEACHER_URL"):
            servers.serve({"teacher": tmp_path})
    finally:
        servers._groups.clear()
        settings.cache_clear()
