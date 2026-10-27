"""The kings row lands first, so the next eval already faces the new king while the upload
runs. After each coronation the older kings' downloads are deleted, each once its mirror is
checked (verify_mirror_before_delete)."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger as log

from common import db, models, reign, states
from common.config import ChainSpec, Quant, Settings, chain_spec, settings
from common.gguf import read_header
from common.gguf.ptq1 import is_pq2, repack
from common.loop import forever
from promoter import config

Fetch = Callable[[str, Path], Path]
Publish = Callable[[Path, str, str, Quant, dict[str, Path]], str]
Matches = Callable[[Path, str, Quant], bool]


def run_once(
    conn: db.Conn,
    *,
    worker: str,
    s: Settings,
    spec: ChainSpec,
    fetch: Fetch = models.fetch,
    publish: Publish = models.publish,
    matches: Matches = models.mirror_matches,
) -> bool:
    db.heartbeat(conn, worker, queued=states.WON)
    row = db.claim(
        conn, worker=worker, queued=states.WON, running=states.CROWNING, lease_s=config.LEASE_S
    )
    if row is None:
        return False
    try:
        king = db.one(
            conn,
            """
            INSERT INTO kings (submission_id, hotkey, uid, model_uri, family, quant)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (submission_id) DO UPDATE SET hotkey = EXCLUDED.hotkey
            RETURNING id, hf_repo
            """,
            (row["id"], row["hotkey"], row["uid"], row["model_uri"], row["family"], row["quant"]),
        )
        # numbered per pair: the first q2 king is Ingot-I-27B-Q2
        ordinal = db.one(
            conn,
            "SELECT count(*) AS n FROM kings"
            " WHERE submission_id IS NOT NULL AND family = %s AND quant = %s AND id <= %s",
            (row["family"], row["quant"], king["id"]),
        )["n"]
        pair = spec.pair(row["family"], row["quant"])
        title = reign.name(ordinal, pair.king_name)
        if not king["hf_repo"] and s.king_hf_namespace:
            repo_id = f"{s.king_hf_namespace}/{title}"
            model = fetch(row["model_uri"], s.model_cache_dir)
            with tempfile.TemporaryDirectory(dir=s.model_cache_dir) as tmp:
                extra = repacked(model, Path(tmp), title)
                steps = lineage(conn, king["id"], pair)
                text = card(row, title, pair, steps, extra, base_model=spec.teacher_repo)
                publish(model, repo_id, text, pair, extra)
            conn.execute("UPDATE kings SET hf_repo = %s WHERE id = %s", (repo_id, king["id"]))
        elif not king["hf_repo"]:
            log.warning("INGOT_KING_HF_NAMESPACE is empty; {} is not mirrored", title)
    except Exception as exc:
        log.exception("crowning {} failed", row["id"])
        db.fail(
            conn,
            row,
            back_to=states.WON,
            error=f"crowning failed: {exc}",
            max_attempts=config.MAX_ATTEMPTS,
        )
        return True
    db.finish(conn, row["id"], states.CROWNED)
    log.info("crowned {} from submission {} (hotkey {})", title, row["id"], row["hotkey"])
    sweep(conn, s, spec, matches)
    return True


def repacked(model_dir: Path, tmp: Path, title: str) -> dict[str, Path]:
    """The king's 1.75 bpw PTQ1_0 file in tmp when its gguf is ternary PQ2_0, as every q2 king is:
    Ingot-IV-27B-PTQ1_0.gguf beside Ingot-IV-27B-PQ2_0.gguf."""
    gguf = next(iter(sorted(model_dir.rglob("*.gguf"))), None)
    if gguf is None or not is_pq2(read_header(gguf)):
        return {}
    name = models.gguf_name(title, "PTQ1_0")
    weights = repack(gguf, tmp / name)
    log.info("repacked {} as {} ({:,} weights)", gguf.name, name, weights)
    return {name: tmp / name}


def sweep(conn: db.Conn, s: Settings, spec: ChainSpec, matches: Matches) -> None:
    """Deletes the downloads of every king but each pair's newest KEEP_KINGS. With
    verify_mirror_before_delete one is deleted only once its mirror matches it; one that does not
    stays, and is checked again after the next coronation."""
    rows = conn.execute(
        "SELECT k.family, k.quant, k.hf_repo, s.model_hash,"
        " row_number() OVER (PARTITION BY k.family, k.quant ORDER BY k.id DESC) AS rank"
        " FROM kings k JOIN submissions s ON s.id = k.submission_id"
    ).fetchall()
    newest = {r["model_hash"] for r in rows if r["rank"] <= config.KEEP_KINGS}
    for r in rows:
        digest = r["model_hash"]
        dest = s.model_cache_dir / digest
        marker = s.model_cache_dir / f"{digest}.verified"
        if digest in newest or not marker.exists():
            continue
        if s.verify_mirror_before_delete:
            try:
                ok = bool(r["hf_repo"]) and matches(
                    dest, r["hf_repo"], spec.pair(r["family"], r["quant"])
                )
            except Exception:
                log.exception("checking the mirror {} failed", r["hf_repo"])
                ok = False
            if not ok:
                log.warning("king {} kept: its mirror {} does not match", digest, r["hf_repo"])
                continue
        shutil.rmtree(dest, ignore_errors=True)
        marker.unlink(missing_ok=True)
        log.info("deleted older king {} (mirrored at {})", digest, r["hf_repo"])


@dataclass(frozen=True)
class Step:
    """One king's winning duel: its scores and the dethroned king's, on the same samples."""

    kl: float
    kl_king: float
    dist: float
    dist_king: float


def lineage(conn: db.Conn, king_id: int, pair: Quant) -> list[Step]:
    """The winning duels of the pair's kings up to this one, oldest first."""
    rows = conn.execute(
        """
        SELECT e.score_challenger, e.score_king, e.dist_challenger, e.dist_king
        FROM kings k
        JOIN LATERAL (
            SELECT * FROM evals e WHERE e.submission_id = k.submission_id AND e.won
            ORDER BY e.id DESC LIMIT 1
        ) e ON TRUE
        WHERE k.family = %s AND k.quant = %s AND k.id <= %s
        ORDER BY k.id
        """,
        (pair.family, pair.name, king_id),
    ).fetchall()
    return [
        Step(
            *(
                float(r[c])
                for c in ("score_challenger", "score_king", "dist_challenger", "dist_king")
            )
        )
        for r in rows
    ]


def card(
    row: db.Row,
    title: str,
    pair: Quant,
    steps: list[Step],
    extra: dict[str, Path] | None = None,
    *,
    base_model: str | None = None,
) -> str:
    gguf = models.gguf_name(title, pair.king_packing)
    ptq1 = next(iter(extra or {}), None)
    text = _front_matter(pair, base_model) + (
        f"# {title}\n\n"
        "A public mirror of a model that won on Ingot, a Bittensor subnet where miners quantize a "
        "fixed teacher model and the validator crowns the copy that stays closest to it.\n\n"
        f"- Submitted by miner hotkey `{row['hotkey']}` (uid {row['uid']})\n"
        f"- Crowned from submission `{row['id']}`\n\n"
    )
    if ptq1:
        text += (
            f"The weights are the miner's upload, byte for byte (`{gguf}`); `{ptq1}` repacks "
            "them losslessly at 1.75 bits per weight. Credit belongs to the miner.\n"
            + _files(gguf, ptq1, (extra or {})[ptq1].stat().st_size, pair.weight_limit)
        )
    else:
        text += "The weights are the miner's upload, byte for byte. Credit belongs to the miner.\n"
    if steps:
        text += _scores(pair, steps)
    if pair.format == "gguf":
        text += _gguf_card(gguf, pair, ptq1)
    return text


def _front_matter(pair: Quant, base_model: str | None) -> str:
    """The card's YAML header: with it Hugging Face lists the mirror as a quantization of the
    teacher."""
    gguf = pair.format == "gguf"
    lines = ["---"]
    if base_model:
        lines += [f"base_model: {base_model}", "base_model_relation: quantized"]
    if gguf:
        lines.append("library_name: llama.cpp")
    lines += ["pipeline_tag: text-generation", "tags:"]
    lines += [f"- {tag}" for tag in (["gguf"] if gguf else []) + ["ingot", "bittensor"]]
    return "\n".join([*lines, "---", "", ""])


def _files(gguf: str, ptq1: str, ptq1_bytes: int, q2_bytes: int | None) -> str:
    """Both files of a q2 king; a q2 king is genesis's size exactly (pair.weight_limit)."""
    return (
        "\n## Files\n\n"
        "| file | bits per weight | size |\n| --- | --- | --- |\n"
        f"| `{ptq1}` | 1.75 | {_gb(ptq1_bytes)} |\n"
        f"| `{gguf}` | 2.125 (the same ternary weights in a 2-bit container) "
        f"| {_gb(q2_bytes)} |\n\n"
        f"Both files hold the same ternary weights. `{ptq1}` is Ingot's lossless repack of the "
        "miner's file into Prism's PTQ1_0 format: every 128-weight block keeps its codes and fp16 "
        "scale, and every weight decodes to the same value. The same repack of Bonsai 2's PQ2_0 "
        "file reproduces Prism's published PTQ1_0 file byte for byte. The 1.75 bpw file is "
        "smaller; the 2-bit file decodes faster on GPUs.\n"
    )


def _scores(pair: Quant, steps: list[Step]) -> str:
    """Its winning duel, and the gain over genesis chained as the dashboard chains it: each eval
    scores its own samples, so a king's ratio to genesis is the product of every challenger /
    king ratio along the way."""
    last = steps[-1]
    kl = dist = 1.0
    for step in steps:
        kl *= step.kl / step.kl_king
        dist *= step.dist / step.dist_king
    genesis = f"[{pair.genesis_repo}](https://huggingface.co/{pair.genesis_repo})"
    return (
        "\n## How close it is to the teacher\n\n"
        "Per answer token, against the BF16 teacher reading the same answer: KL(teacher ‖ model) "
        "of the next-token distributions, and the hidden-state distance |h − h_teacher| / "
        "|h_teacher| averaged over every 8th layer. Lower is closer.\n\n"
        "| | KL | hidden-state distance |\n| --- | --- | --- |\n"
        f"| this model, in its winning eval | {last.kl:.5f} | {last.dist:.4f} |\n"
        f"| the king it beat, same eval | {last.kl_king:.5f} | {last.dist_king:.4f} |\n"
        f"| closer than genesis {genesis} | {_gain(kl)} | {_gain(dist)} |\n"
    )


def _gain(ratio: float) -> str:
    return f"{100 * (1 - ratio):.1f}%"


def _gb(n: int | None) -> str:
    return "—" if n is None else f"{n / 1e9:.2f} GB"


def _gguf_card(gguf: str, pair: Quant, ptq1: str | None = None) -> str:
    """How to run a gguf king: only the fork the eval scored it with runs these files; and the
    genesis vision projector, which the eval never scored, for image input."""
    from eval_service.config import LLAMA_CPP_COMMIT

    fork = "https://github.com/DendriteLabs/llama.cpp"
    hub = f"https://huggingface.co/{pair.genesis_repo}"
    text = (
        "\n## Running it\n\n"
        f"`{gguf}` is a gguf with the layout, tokenizer and Hadamard rotation of "
        f"[{pair.genesis_repo}]({hub}) (`{pair.genesis_file}`), its weights the miner's. Stock "
        "llama.cpp does not run it; the Dendrite fork of PrismML's llama.cpp does, at the commit "
        "Ingot scored it with, "
        f"[`{LLAMA_CPP_COMMIT[:7]}`]({fork}/commit/{LLAMA_CPP_COMMIT}):\n\n"
        "```\n"
        f"git clone {fork} && cd llama.cpp && git checkout {LLAMA_CPP_COMMIT}\n"
        "cmake -B build -DGGML_CUDA=ON && cmake --build build -j\n"
        f"build/bin/llama-server -m {gguf} -ngl 99 -fa on --jinja\n"
        "```\n"
    )
    if ptq1:
        text += f"\nEither file runs the same way: pass `-m {ptq1}` for the 1.75 bpw one.\n"
    if pair.vision_files:
        first = pair.vision_files[0]
        vrepo, _, vrev = (pair.vision_repo or "").partition("@")
        vrepo, vrev = vrepo or pair.genesis_repo, vrev or pair.genesis_revision
        vhub = f"https://huggingface.co/{vrepo}"
        files = ", ".join(f"[`{f}`]({vhub}/blob/{vrev}/{f})" for f in pair.vision_files)
        text += (
            "\n### With images\n\n"
            "The file is the text model. For image input, add the vision tower of "
            f"[{vrepo}]({vhub}) as a projector ({files}). It was made for genesis, not "
            "for this model's weights, and Ingot scores the text model only, so image quality "
            "is untested:\n\n"
            "```\n"
            f"hf download {vrepo} {first} --revision {vrev}"
            " --local-dir .\n"
            f"build/bin/llama-server -m {gguf} --mmproj {first} -ngl 99 -fa on --jinja\n"
            "```\n"
        )
    return text


def main() -> None:
    s = settings()
    link = db.Link()
    worker = db.register_worker(link.conn(), "promoter")
    log.info("promoter started — mirroring kings to {}", s.king_hf_namespace or "(nowhere)")
    forever(
        lambda: run_once(link.conn(), worker=worker, s=s, spec=chain_spec()),
        idle_s=config.IDLE_S,
        on_stop=lambda: db.offline_quietly(link, worker),
    )
