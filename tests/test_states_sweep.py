import os
import time

from eval_service import servers


def test_the_sweep_leaves_held_files_and_takes_abandoned_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(servers, "HIDDEN_STATES_DIR", tmp_path)
    old = time.time() - 7200
    paths = {}
    for name in ("held", "abandoned", "fresh"):
        path = tmp_path / "gpu0" / f"{name}.safetensors"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x")
        lock = path.with_name(path.name + ".lock")
        lock.write_text("")
        if name != "fresh":
            for p in (path, lock):
                os.utime(p, (old, old))
        paths[name] = str(path)
    servers.hold(paths["held"])
    try:
        assert servers.sweep(3600) == 1
        assert os.path.exists(paths["held"]) and os.path.exists(paths["held"] + ".lock")
        assert not os.path.exists(paths["abandoned"]) and not os.path.exists(
            paths["abandoned"] + ".lock"
        )
        assert os.path.exists(paths["fresh"])
        servers.discard(paths["held"])  # a discarded file is no longer held
        assert not os.path.exists(paths["held"]) and paths["held"] not in servers._held
    finally:
        servers._held.discard(paths["held"])
