"""The promoter keeps each pair's newest king on disk; an older one goes once its mirror matches."""

from __future__ import annotations

from pathlib import Path

from common import db
from promoter import main as promoter
from tests.conftest import submit


def crown(conn: db.Conn, s, hotkey: str, uid: int) -> str:
    sid = submit(conn, hotkey=hotkey, uid=uid)
    digest = db.one(conn, "SELECT model_hash FROM submissions WHERE id = %s", (sid,))["model_hash"]
    conn.execute(
        "INSERT INTO kings (submission_id, hotkey, uid, model_uri, hf_repo, family, quant)"
        " VALUES (%s, %s, %s, 'u', %s, 'qwen38_27b', 'fp8')",
        (sid, hotkey, uid, f"org/{hotkey}"),
    )
    (s.model_cache_dir / digest).mkdir()
    (s.model_cache_dir / digest / "model.safetensors").write_bytes(b"w")
    (s.model_cache_dir / f"{digest}.verified").write_text("")
    return digest


def on_disk(s, digest: str) -> bool:
    return (s.model_cache_dir / f"{digest}.verified").exists()


def test_an_older_king_goes_once_its_mirror_matches(conn, s, spec):
    old, new = crown(conn, s, "hk-1", 7), crown(conn, s, "hk-2", 8)
    checked: list[str] = []

    def mismatch(_dir: Path, repo: str, _pair: object) -> bool:
        checked.append(repo)
        return False

    promoter.sweep(conn, s, spec, mismatch)
    assert checked == ["org/hk-1"] and on_disk(s, old) and on_disk(s, new)

    promoter.sweep(conn, s, spec, lambda *_: True)
    assert not on_disk(s, old) and not (s.model_cache_dir / old).exists()
    assert on_disk(s, new)


def test_without_the_check_an_older_king_goes_at_once(conn, s, spec):
    old, new = crown(conn, s, "hk-1", 7), crown(conn, s, "hk-2", 8)
    s.verify_mirror_before_delete = False
    promoter.sweep(conn, s, spec, lambda *_: False)
    assert not on_disk(s, old) and on_disk(s, new)
