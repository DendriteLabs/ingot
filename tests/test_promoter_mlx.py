"""A gguf king is mirrored twice: its gguf repo at the coronation, then, as a separate step
that never touches the coronation, an MLX pack of the same weights beside it."""

from __future__ import annotations

from pathlib import Path

import pytest

from common import db, models
from common.gguf import mlx
from promoter import config
from promoter import main as promoter
from promoter.main import card, card_mlx
from tests.conftest import crown_genesis, submit
from tests.test_pipeline import WORKER
from tests.test_promoter_sweep import crown, on_disk


@pytest.fixture
def q2_upload(conn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An upload holding a gguf the promoter takes for PQ2_0; the PTQ1_0 repack and the MLX
    conversion are stubbed, the latter to write a one-byte pack."""
    conn.execute("INSERT INTO workers (name, service) VALUES (%s, %s)", (WORKER, "test"))
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "king.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(promoter, "read_header", lambda _p: None)
    monkeypatch.setattr(promoter, "is_pq2", lambda _g: True)
    monkeypatch.setattr(promoter, "repacked", lambda *_: {})
    monkeypatch.setattr(promoter, "ternary_changes", lambda *_: (0, 128))
    converted: list[tuple[Path, Path, Path]] = []

    def convert(src: Path, template: Path, out: Path) -> int:
        out.mkdir()
        (out / mlx.WEIGHTS).write_bytes(b"w")
        converted.append((src, template, out))
        return 1

    monkeypatch.setattr(mlx, "convert", convert)
    monkeypatch.setattr(mlx, "verify", lambda *_: None)
    return upload, converted


def crown_q2(conn, s, hotkey: str, uid: int) -> str:
    """A crowned q2 king with its gguf mirrored, as run_once leaves it."""
    digest = crown(conn, s, hotkey, uid)
    conn.execute("UPDATE kings SET quant = 'q2' WHERE hotkey = %s", (hotkey,))
    conn.execute("UPDATE submissions SET state = 'CROWNED' WHERE hotkey = %s", (hotkey,))
    return digest


def run_mlx(conn, s, spec, upload: Path, **stubs) -> bool:
    kw = {
        "fetch": lambda _uri, _cache: upload,
        "publish_dir": lambda _d, repo, _c, _p: repo,
        "snapshot": lambda repo, *_a, **_kw: Path("/template"),
        "update_card": lambda _repo, _line: None,
    }
    kw.update(stubs)
    return promoter.mlx_once(conn, s=s, spec=spec, **kw)


def king_of(conn, hotkey: str) -> db.Row:
    return db.one(
        conn,
        "SELECT hf_repo, hf_repo_mlx, mlx_attempts, mlx_error FROM kings WHERE hotkey = %s",
        (hotkey,),
    )


def test_the_coronation_publishes_the_gguf_repo_only(conn, s, spec, q2_upload):
    upload, converted = q2_upload
    crown_genesis(conn)
    sid = submit(conn, hotkey="hk-2", uid=8)
    conn.execute("UPDATE submissions SET state = 'WON', quant = 'q2' WHERE id = %s", (sid,))
    published: list[str] = []

    def publish(_d, repo, text, _p, _e) -> str:
        published.append(repo)
        assert "MLX" not in text
        return repo

    assert promoter.run_once(
        conn,
        worker=WORKER,
        s=s,
        spec=spec,
        fetch=lambda _u, _c: upload,
        publish=publish,
        matches=lambda *_: True,
    )
    assert published == ["ingot-test/Ingot-I-27B-Q2"] and converted == []
    assert king_of(conn, "hk-2")["hf_repo_mlx"] is None
    assert db.one(conn, "SELECT state FROM submissions WHERE id = %s", (sid,))["state"] == "CROWNED"


def test_the_mlx_pack_follows_the_coronation(conn, s, spec, q2_upload):
    upload, converted = q2_upload
    crown_q2(conn, s, "hk-1", 7)
    packs: list[tuple[str, str]] = []
    links: list[tuple[str, str]] = []

    def publish_dir(pack: Path, repo: str, text: str, _pair) -> str:
        assert (pack / mlx.WEIGHTS).exists() and pack.name == "mlx"
        packs.append((repo, text))
        return repo

    assert run_mlx(
        conn,
        s,
        spec,
        upload,
        publish_dir=publish_dir,
        update_card=lambda repo, line: links.append((repo, line)),
    )
    king = king_of(conn, "hk-1")
    assert king["hf_repo_mlx"] == "ingot-test/Ingot-I-27B-MLX-2bit" and king["mlx_attempts"] == 0
    assert [repo for repo, _ in packs] == ["ingot-test/Ingot-I-27B-MLX-2bit"]
    assert "# Ingot-I-27B-MLX-2bit" in packs[0][1]
    assert links == [("org/hk-1", promoter._mlx_link("ingot-test/Ingot-I-27B-MLX-2bit"))]
    src, template, out = converted[0]
    assert src == upload / "king.gguf" and template == Path("/template")
    assert not out.exists(), "the pack goes with its temporary directory"
    assert not run_mlx(conn, s, spec, upload), "nothing is left to do"


def test_a_failed_pack_counts_an_attempt_and_touches_nothing_else(conn, s, spec, q2_upload):
    upload, _ = q2_upload
    crown_q2(conn, s, "hk-1", 7)

    def refuse(*_) -> str:
        raise RuntimeError("hub down")

    for attempt in range(1, config.MAX_ATTEMPTS + 1):
        assert run_mlx(conn, s, spec, upload, publish_dir=refuse)
        king = king_of(conn, "hk-1")
        assert king["hf_repo"] == "org/hk-1" and king["hf_repo_mlx"] is None
        assert king["mlx_attempts"] == attempt and king["mlx_error"] == "hub down"
    assert db.one(conn, "SELECT state FROM submissions")["state"] == "CROWNED"
    assert not run_mlx(conn, s, spec, upload, publish_dir=refuse), "past the limit it is left alone"

    conn.execute("UPDATE kings SET mlx_attempts = 0")
    assert run_mlx(conn, s, spec, upload)
    king = king_of(conn, "hk-1")
    assert king["hf_repo_mlx"] == "ingot-test/Ingot-I-27B-MLX-2bit" and king["mlx_error"] is None


def test_a_gone_upload_is_taken_from_the_mirror(conn, s, spec, q2_upload):
    upload, converted = q2_upload
    crown_q2(conn, s, "hk-1", 7)
    snapshots: list[tuple[str, str | None]] = []

    def fetch(_uri, _cache) -> Path:
        raise FileNotFoundError("expired")

    def snapshot(repo: str, revision: str | None = None, **kw) -> Path:
        snapshots.append((repo, kw.get("allow_patterns")))
        return upload if repo == "org/hk-1" else Path("/template")

    assert run_mlx(conn, s, spec, upload, fetch=fetch, snapshot=snapshot)
    assert snapshots[0] == ("org/hk-1", "*-PQ2_0.gguf") and len(converted) == 1
    assert king_of(conn, "hk-1")["hf_repo_mlx"] == "ingot-test/Ingot-I-27B-MLX-2bit"


def test_mlx_once_skips_kings_that_are_not_gguf(conn, s, spec, q2_upload):
    upload, converted = q2_upload
    crown(conn, s, "hk-1", 7)  # fp8
    assert not run_mlx(conn, s, spec, upload) and converted == []


def test_an_older_gguf_king_stays_until_its_mlx_pack_is_mirrored(conn, s, spec):
    old, new = crown(conn, s, "hk-1", 7), crown(conn, s, "hk-2", 8)
    conn.execute("UPDATE kings SET quant = 'q2', mlx_attempts = %s", (config.MAX_ATTEMPTS,))
    promoter.sweep(conn, s, spec, lambda *_: True)
    assert on_disk(s, old) and on_disk(s, new)

    conn.execute("UPDATE kings SET hf_repo_mlx = 'org/mlx' WHERE hotkey = 'hk-1'")
    promoter.sweep(conn, s, spec, lambda *_: True)
    assert not on_disk(s, old) and on_disk(s, new)


def test_the_mlx_card_and_the_link_from_the_gguf_card(spec):
    row = {"hotkey": "hk", "uid": 1, "id": "sub"}
    q2 = spec.pair("qwen38_27b", "q2")
    text = card_mlx(
        row, "Ingot-I-27B-Q2", q2, [], 8_595_478_424, gguf_repo="ns/Ingot-I-27B-Q2", base_model="t"
    )
    assert text.startswith(f"---\nbase_model: {mlx.TEMPLATE_REPO}\nbase_model_relation: finetune\n")
    assert "library_name: mlx" in text and "Credit belongs to the miner, and to Prism ML" in text
    assert "# Ingot-I-27B-MLX-2bit" in text
    assert (
        "| `model.safetensors` | 2.25 (2-bit codes, fp16 scale and bias per 128 weights) |" in text
    )
    assert "| 8.60 GB |" in text
    assert f"[{mlx.TEMPLATE_REPO}](https://huggingface.co/{mlx.TEMPLATE_REPO})" in text
    assert "mlx_lm.generate --model ns/Ingot-I-27B-MLX-2bit" in text

    gguf = card(row, "Ingot-I-27B-Q2", q2, [], mlx_repo="ns/Ingot-I-27B-MLX-2bit")
    assert "[ns/Ingot-I-27B-MLX-2bit](https://huggingface.co/ns/Ingot-I-27B-MLX-2bit)" in gguf
    assert "MLX" not in card(row, "Ingot-I-27B-Q2", q2, [])


def test_publish_dir_uploads_every_file_of_the_pack(tmp_path, monkeypatch, spec):
    from tests.test_publish import FakeApi, _fake_hub

    _fake_hub(monkeypatch, tmp_path)
    pack = tmp_path / "mlx"
    pack.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (pack / name).write_bytes(b"x")
    models.publish_dir(pack, "ns/Ingot-I-27B-MLX-2bit", "card", spec.pair("qwen38_27b", "q2"))
    assert FakeApi.commits == [
        ["LICENSE", "NOTICE.txt", "README.md", "config.json", "model.safetensors", "tokenizer.json"]
    ]
    assert models.mlx_repo_name("Ingot-IV-27B-Q2") == "Ingot-IV-27B-MLX-2bit"


def test_update_card_appends_the_link_once(tmp_path, monkeypatch):
    from tests.test_publish import FakeApi, _fake_hub

    _fake_hub(monkeypatch, tmp_path)
    (tmp_path / "README.md").write_text("# king\n")
    monkeypatch.setattr(models, "hub_file", lambda *_: tmp_path / "README.md")
    models.update_card("ns/king", "\nlink\n")
    assert FakeApi.commits == [["README.md"]]
    (tmp_path / "README.md").write_text("# king\n\nlink\n")
    models.update_card("ns/king", "\nlink\n")
    assert FakeApi.commits == [["README.md"]], "a card that has the link is left alone"


def test_both_cards_carry_the_speed_tables(spec, tmp_path: Path):
    from promoter import speed

    row = {"hotkey": "hk", "uid": 1, "id": "sub"}
    q2 = spec.pair("qwen38_27b", "q2")
    ptq1 = tmp_path / "Ingot-I-27B-PTQ1_0.gguf"
    ptq1.write_bytes(b"x")
    gguf = card(row, "Ingot-I-27B-Q2", q2, [], {ptq1.name: ptq1})
    assert "\n## Speed\n" in gguf and "### GPU" in gguf and "### Mac" in gguf
    assert gguf.index("## Speed") < gguf.index("## Running it")
    for build, rows in speed.MAC_ROWS:
        for packing, name in (("PQ2_0", "Ingot-I-27B-PQ2_0.gguf"), ("PTQ1_0", ptq1.name)):
            decode, prompt = rows[packing]
            assert f"| {build} | `{name}` | {decode} | {prompt} |" in gguf
    assert f"| {speed.GPU} | `Ingot-I-27B-PQ2_0.gguf` | {speed.GPU_PQ2[0]} |" in gguf
    assert "MLX" not in gguf
    linked = card(
        row, "Ingot-I-27B-Q2", q2, [], {ptq1.name: ptq1}, mlx_repo="ns/Ingot-I-27B-MLX-2bit"
    )
    assert "see the speed table of [ns/Ingot-I-27B-MLX-2bit]" in linked

    plain = card(row, "Ingot-I-27B-Q2", q2, [])
    assert "PTQ1_0" not in plain and "| `Ingot-I-27B-PQ2_0.gguf` |" in plain

    mlx_text = card_mlx(row, "Ingot-I-27B-Q2", q2, [], 8_595_478_424, gguf_repo="ns/Ingot-I-27B-Q2")
    assert "\n## Speed\n" in mlx_text and mlx_text.index("## Speed") < mlx_text.index(
        "## Running it"
    )
    for engine, mode, decode in speed.MLX_ROWS:
        assert f"| {engine} | {mode} | {decode} |" in mlx_text
    assert "see [its card](https://huggingface.co/ns/Ingot-I-27B-Q2)" in mlx_text
